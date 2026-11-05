#include "yolox_ppe_postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kClassCount = 12;
constexpr int kChannels = 5 + kClassCount;

constexpr std::array<const char*, kClassCount> kClassNames = {
    "helmet", "gloves", "vest", "boots", "goggles", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"};

struct Candidate {
    float left;
    float top;
    float right;
    float bottom;
    float score;
    int class_id;
};

float intersection_over_union(const Candidate& lhs, const Candidate& rhs) {
    const float left = std::max(lhs.left, rhs.left);
    const float top = std::max(lhs.top, rhs.top);
    const float right = std::min(lhs.right, rhs.right);
    const float bottom = std::min(lhs.bottom, rhs.bottom);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float lhs_area = std::max(0.0F, lhs.right - lhs.left) *
                           std::max(0.0F, lhs.bottom - lhs.top);
    const float rhs_area = std::max(0.0F, rhs.right - rhs.left) *
                           std::max(0.0F, rhs.bottom - rhs.top);
    const float union_area = lhs_area + rhs_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

float tensor_value(const rknn_app_context_t& app_ctx, const rknn_tensor_attr& attr,
                   const rknn_output& output, std::size_t index) {
    if (app_ctx.is_quant) {
        if (attr.type == RKNN_TENSOR_INT8) {
            const auto* values = static_cast<const std::int8_t*>(output.buf);
            return (static_cast<float>(values[index]) - static_cast<float>(attr.zp)) * attr.scale;
        }
        if (attr.type == RKNN_TENSOR_UINT8) {
            const auto* values = static_cast<const std::uint8_t*>(output.buf);
            return (static_cast<float>(values[index]) - static_cast<float>(attr.zp)) * attr.scale;
        }
        throw std::runtime_error("quantized YOLOX output must use int8 or uint8 tensors");
    }
    return static_cast<const float*>(output.buf)[index];
}

}  // namespace

const char* ppe_class_to_name(int class_id) noexcept {
    return class_id >= 0 && class_id < kClassCount ? kClassNames[static_cast<std::size_t>(class_id)]
                                                   : "unknown";
}

int post_process_yolox_ppe(rknn_app_context_t* app_ctx, const rknn_output* outputs,
                           const letterbox_t* letter_box, float confidence_threshold,
                           float nms_threshold, int source_width, int source_height,
                           object_detect_result_list* results) {
    if (app_ctx == nullptr || outputs == nullptr || letter_box == nullptr || results == nullptr ||
        app_ctx->output_attrs == nullptr || app_ctx->io_num.n_output != 3 ||
        app_ctx->model_width <= 0 || app_ctx->model_height <= 0 || source_width <= 0 ||
        source_height <= 0 || letter_box->scale <= 0.0F || confidence_threshold < 0.0F ||
        confidence_threshold > 1.0F || nms_threshold < 0.0F || nms_threshold > 1.0F) {
        return -1;
    }

    std::memset(results, 0, sizeof(*results));
    std::vector<Candidate> candidates;
    candidates.reserve(1000);

    for (std::uint32_t output_index = 0; output_index < app_ctx->io_num.n_output; ++output_index) {
        const auto& attr = app_ctx->output_attrs[output_index];
        if (outputs[output_index].buf == nullptr || attr.n_dims != 4 || attr.dims[0] != 1) {
            return -2;
        }

        const bool channels_first = attr.dims[1] == kChannels;
        const bool channels_last = attr.dims[3] == kChannels;
        if (channels_first == channels_last) return -3;
        if ((channels_first && attr.fmt != RKNN_TENSOR_NCHW) ||
            (channels_last && attr.fmt != RKNN_TENSOR_NHWC)) {
            return -4;
        }
        const int grid_height = static_cast<int>(channels_first ? attr.dims[2] : attr.dims[1]);
        const int grid_width = static_cast<int>(channels_first ? attr.dims[3] : attr.dims[2]);
        if (grid_height <= 0 || grid_width <= 0 || app_ctx->model_height % grid_height != 0 ||
            app_ctx->model_width % grid_width != 0 ||
            app_ctx->model_height / grid_height != app_ctx->model_width / grid_width) {
            return -5;
        }
        const int stride = app_ctx->model_height / grid_height;
        const std::size_t plane_size = static_cast<std::size_t>(grid_height) * grid_width;
        const std::size_t values_count = plane_size * kChannels;
        const std::size_t element_size = app_ctx->is_quant ? sizeof(std::int8_t) : sizeof(float);
        if (outputs[output_index].size < values_count * element_size) return -6;

        const auto value_at = [&](int channel, int y, int x) {
            const std::size_t index = channels_first
                ? static_cast<std::size_t>(channel) * plane_size +
                      static_cast<std::size_t>(y) * grid_width + x
                : (static_cast<std::size_t>(y) * grid_width + x) * kChannels + channel;
            return tensor_value(*app_ctx, attr, outputs[output_index], index);
        };

        for (int y = 0; y < grid_height; ++y) {
            for (int x = 0; x < grid_width; ++x) {
                const float objectness = value_at(4, y, x);
                int class_id = 0;
                float class_score = value_at(5, y, x);
                for (int channel = 1; channel < kClassCount; ++channel) {
                    const float score = value_at(5 + channel, y, x);
                    if (score > class_score) {
                        class_score = score;
                        class_id = channel;
                    }
                }
                const float score = objectness * class_score;
                if (!std::isfinite(score) || score < confidence_threshold) continue;

                const float center_x = (value_at(0, y, x) + x) * stride;
                const float center_y = (value_at(1, y, x) + y) * stride;
                const float width = std::exp(value_at(2, y, x)) * stride;
                const float height = std::exp(value_at(3, y, x)) * stride;
                if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
                    !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F || height <= 0.0F) {
                    continue;
                }

                const float left = (center_x - width * 0.5F - letter_box->x_pad) / letter_box->scale;
                const float top = (center_y - height * 0.5F - letter_box->y_pad) / letter_box->scale;
                const float right = (center_x + width * 0.5F - letter_box->x_pad) / letter_box->scale;
                const float bottom = (center_y + height * 0.5F - letter_box->y_pad) / letter_box->scale;
                Candidate candidate{
                    std::clamp(left, 0.0F, static_cast<float>(source_width)),
                    std::clamp(top, 0.0F, static_cast<float>(source_height)),
                    std::clamp(right, 0.0F, static_cast<float>(source_width)),
                    std::clamp(bottom, 0.0F, static_cast<float>(source_height)),
                    score,
                    class_id,
                };
                if (candidate.right > candidate.left && candidate.bottom > candidate.top) {
                    candidates.push_back(candidate);
                }
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.score > rhs.score;
    });
    std::vector<Candidate> selected;
    selected.reserve(std::min<std::size_t>(candidates.size(), OBJ_NUMB_MAX_SIZE));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& kept : selected) {
            if (candidate.class_id == kept.class_id &&
                intersection_over_union(candidate, kept) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        selected.push_back(candidate);
        if (selected.size() == OBJ_NUMB_MAX_SIZE) break;
    }

    results->count = static_cast<int>(selected.size());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const auto& candidate = selected[i];
        results->results[i].box.left = static_cast<int>(std::lround(candidate.left));
        results->results[i].box.top = static_cast<int>(std::lround(candidate.top));
        results->results[i].box.right = static_cast<int>(std::lround(candidate.right));
        results->results[i].box.bottom = static_cast<int>(std::lround(candidate.bottom));
        results->results[i].prop = candidate.score;
        results->results[i].cls_id = candidate.class_id;
    }
    return 0;
}
