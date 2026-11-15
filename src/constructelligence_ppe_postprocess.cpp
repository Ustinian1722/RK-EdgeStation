#include "constructelligence_ppe_postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace edge {
namespace {

constexpr int kSourceClassCount = 14;
constexpr std::array<int, kSourceClassCount> kSourceToProjectClass = {
    5, 0, 6, 2, 11, -1, 1, 3, -1, -1, -1, 4, 7, 8};

struct Candidate {
    int class_id;
    float score;
    float left;
    float top;
    float right;
    float bottom;
};

float intersection_over_union(const Candidate& a, const Candidate& b) {
    const float left = std::max(a.left, b.left);
    const float top = std::max(a.top, b.top);
    const float right = std::min(a.right, b.right);
    const float bottom = std::min(a.bottom, b.bottom);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float area_a = std::max(0.0F, a.right - a.left) * std::max(0.0F, a.bottom - a.top);
    const float area_b = std::max(0.0F, b.right - b.left) * std::max(0.0F, b.bottom - b.top);
    const float union_area = area_a + area_b - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

}  // namespace

std::vector<Detection> decode_constructelligence_ppe(
    const float* output, int channels, int anchors, const letterbox_t& letterbox,
    int source_width, int source_height, float confidence_threshold,
    float nms_threshold) {
    if (output == nullptr || channels != 4 + kSourceClassCount || anchors <= 0 ||
        source_width <= 0 || source_height <= 0 || !std::isfinite(letterbox.scale) ||
        letterbox.scale <= 0.0F || confidence_threshold < 0.0F ||
        confidence_threshold > 1.0F || nms_threshold < 0.0F || nms_threshold > 1.0F) {
        throw std::invalid_argument("invalid Constructelligence PPE output contract");
    }

    const auto value_at = [=](int channel, int anchor) {
        return output[static_cast<std::size_t>(channel) * anchors + anchor];
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(anchors / 5));
    for (int anchor = 0; anchor < anchors; ++anchor) {
        int source_class = 0;
        float score = value_at(4, anchor);
        for (int cls = 1; cls < kSourceClassCount; ++cls) {
            const float candidate_score = value_at(4 + cls, anchor);
            if (candidate_score > score) {
                source_class = cls;
                score = candidate_score;
            }
        }
        if (!std::isfinite(score) || score < confidence_threshold ||
            kSourceToProjectClass[static_cast<std::size_t>(source_class)] < 0) {
            continue;
        }

        const float center_x = value_at(0, anchor);
        const float center_y = value_at(1, anchor);
        const float width = value_at(2, anchor);
        const float height = value_at(3, anchor);
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F || height <= 0.0F) {
            continue;
        }
        Candidate detection{};
        detection.class_id = kSourceToProjectClass[static_cast<std::size_t>(source_class)];
        detection.score = score;
        detection.left = std::clamp((center_x - width * 0.5F - letterbox.x_pad) /
                                         letterbox.scale,
                                     0.0F, static_cast<float>(source_width));
        detection.top = std::clamp((center_y - height * 0.5F - letterbox.y_pad) /
                                       letterbox.scale,
                                   0.0F, static_cast<float>(source_height));
        detection.right = std::clamp((center_x + width * 0.5F - letterbox.x_pad) /
                                         letterbox.scale,
                                     0.0F, static_cast<float>(source_width));
        detection.bottom = std::clamp((center_y + height * 0.5F - letterbox.y_pad) /
                                          letterbox.scale,
                                      0.0F, static_cast<float>(source_height));
        if (detection.right > detection.left && detection.bottom > detection.top) {
            candidates.push_back(detection);
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });
    std::vector<Candidate> kept;
    kept.reserve(std::min<std::size_t>(300, candidates.size()));
    for (const auto& candidate : candidates) {
        const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const Candidate& prior) {
            return prior.class_id == candidate.class_id &&
                   intersection_over_union(prior, candidate) > nms_threshold;
        });
        if (!suppressed) kept.push_back(candidate);
        if (kept.size() == 300) break;
    }

    std::vector<Detection> detections;
    detections.reserve(kept.size());
    for (const auto& item : kept) {
        detections.push_back({item.class_id, item.score,
                              static_cast<int>(std::lround(item.left)),
                              static_cast<int>(std::lround(item.top)),
                              static_cast<int>(std::lround(item.right)),
                              static_cast<int>(std::lround(item.bottom)), "PPE"});
    }
    return detections;
}

}  // namespace edge
