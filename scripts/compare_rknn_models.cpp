// Run the same UINT8 NHWC input through two RKNN models and dump float outputs.
// Build on RK3576:
// g++ -O2 -std=c++17 -Wall -Wextra -Wpedantic \
//   -I src/third_party/rknn_model_zoo_v2.0.0/utils \
//   -I src/third_party/rknn_model_zoo_v2.0.0/yolov8 \
//   scripts/compare_rknn_models.cpp \
//   src/third_party/rknn_model_zoo_v2.0.0/yolov8/postprocess.cc \
//   -o compare-rknn-models -lrknnrt

#include <rknn_api.h>
#include "yolov8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string reference_model;
    std::string candidate_model;
    std::string input;
    std::string output_dir;
    int source_width = 768;
    int source_height = 576;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "compare_rknn_models --reference MODEL.rknn --candidate MODEL.rknn "
                         "--input RGB-U8-NHWC.bin --output-dir DIR "
                         "[--source-width N] [--source-height N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--reference") options.reference_model = value;
        else if (key == "--candidate") options.candidate_model = value;
        else if (key == "--input") options.input = value;
        else if (key == "--output-dir") options.output_dir = value;
        else if (key == "--source-width") options.source_width = std::stoi(value);
        else if (key == "--source-height") options.source_height = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.reference_model.empty() || options.candidate_model.empty() ||
        options.input.empty() || options.output_dir.empty()) {
        throw std::runtime_error("--reference, --candidate, --input and --output-dir are required");
    }
    if (options.source_width < 1 || options.source_height < 1) {
        throw std::runtime_error("source dimensions must be positive");
    }
    return options;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    const auto end = input.tellg();
    if (end <= 0) throw std::runtime_error("empty file: " + path);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("failed to read file: " + path);
    return bytes;
}

const char* tensor_type_name(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32: return "float32";
        case RKNN_TENSOR_FLOAT16: return "float16";
        case RKNN_TENSOR_INT8: return "int8";
        case RKNN_TENSOR_UINT8: return "uint8";
        case RKNN_TENSOR_INT16: return "int16";
        case RKNN_TENSOR_UINT16: return "uint16";
        case RKNN_TENSOR_INT32: return "int32";
        case RKNN_TENSOR_UINT32: return "uint32";
        case RKNN_TENSOR_INT64: return "int64";
        case RKNN_TENSOR_BOOL: return "bool";
        default: return "unknown";
    }
}

const char* quantization_name(rknn_tensor_qnt_type type) {
    switch (type) {
        case RKNN_TENSOR_QNT_NONE: return "none";
        case RKNN_TENSOR_QNT_DFP: return "dfp";
        case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC: return "affine_asymmetric";
        default: return "unknown";
    }
}

using TensorOutputs = std::vector<std::vector<float>>;

class Model {
public:
    explicit Model(const std::string& path) : path_(path), model_bytes_(read_bytes(path)) {
        int status = rknn_init(&context_, model_bytes_.data(),
                               static_cast<std::uint32_t>(model_bytes_.size()), 0, nullptr);
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_init failed for " + path + ": " +
                                                          std::to_string(status));
        try {
            status = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
            if (status != RKNN_SUCC || io_num_.n_input != 1 || io_num_.n_output != 9) {
                throw std::runtime_error("expected one image input and nine YOLO outputs: " + path);
            }
            input_attr_ = {};
            input_attr_.index = 0;
            status = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
            if (status != RKNN_SUCC || input_attr_.n_dims != 4 ||
                input_attr_.fmt != RKNN_TENSOR_NHWC || input_attr_.dims[1] != 640 ||
                input_attr_.dims[2] != 640 || input_attr_.dims[3] != 3) {
                throw std::runtime_error("expected 640x640x3 NHWC model input: " + path);
            }
            output_attrs_.resize(io_num_.n_output);
            for (std::uint32_t i = 0; i < io_num_.n_output; ++i) {
                output_attrs_[i] = {};
                output_attrs_[i].index = i;
                status = rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i],
                                    sizeof(rknn_tensor_attr));
                if (status != RKNN_SUCC) throw std::runtime_error("failed to query output attrs: " + path);
            }
        } catch (...) {
            rknn_destroy(context_);
            context_ = 0;
            throw;
        }
    }

    ~Model() {
        if (context_ != 0) rknn_destroy(context_);
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const std::string& path() const { return path_; }
    const std::vector<std::uint8_t>& model_bytes() const { return model_bytes_; }
    const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }
    const rknn_tensor_attr& input_attr() const { return input_attr_; }
    rknn_input_output_num io_num() const { return io_num_; }

    TensorOutputs run(const std::vector<std::uint8_t>& input_bytes, double* run_ms,
                      double* run_plus_fetch_ms) {
        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<std::uint32_t>(input_bytes.size());
        input.buf = const_cast<std::uint8_t*>(input_bytes.data());
        int status = rknn_inputs_set(context_, 1, &input);
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(status));

        const auto start = Clock::now();
        status = rknn_run(context_, nullptr);
        const auto run_end = Clock::now();
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(status));

        std::vector<rknn_output> outputs(io_num_.n_output);
        for (std::uint32_t i = 0; i < io_num_.n_output; ++i) {
            outputs[i] = {};
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        status = rknn_outputs_get(context_, io_num_.n_output, outputs.data(), nullptr);
        const auto fetch_end = Clock::now();
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(status));

        TensorOutputs copied;
        copied.reserve(outputs.size());
        for (const auto& output : outputs) {
            if (output.buf == nullptr || output.size == 0 || output.size % sizeof(float) != 0) {
                rknn_outputs_release(context_, io_num_.n_output, outputs.data());
                throw std::runtime_error("RKNN returned an invalid float output");
            }
            const auto* begin = static_cast<const float*>(output.buf);
            copied.emplace_back(begin, begin + output.size / sizeof(float));
        }
        status = rknn_outputs_release(context_, io_num_.n_output, outputs.data());
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_release failed: " + std::to_string(status));
        *run_ms = std::chrono::duration<double, std::milli>(run_end - start).count();
        *run_plus_fetch_ms = std::chrono::duration<double, std::milli>(fetch_end - start).count();
        return copied;
    }

private:
    std::string path_;
    std::vector<std::uint8_t> model_bytes_;
    rknn_context context_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
};

void save_outputs(const TensorOutputs& outputs, const Model& model,
                  const Options& options, const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    const int model_width = static_cast<int>(model.input_attr().dims[2]);
    const int model_height = static_cast<int>(model.input_attr().dims[1]);
    const float scale = std::min(static_cast<float>(model_width) / options.source_width,
                                 static_cast<float>(model_height) / options.source_height);
    const int resized_width = static_cast<int>(std::round(options.source_width * scale));
    const int resized_height = static_cast<int>(std::round(options.source_height * scale));
    const int pad_x = (model_width - resized_width) / 2;
    const int pad_y = (model_height - resized_height) / 2;

    std::ofstream manifest(root / "tensor-manifest.txt");
    if (!manifest) throw std::runtime_error("cannot create output tensor manifest");
    manifest << "model=" << std::filesystem::path(model.path()).filename().string() << '\n'
             << "input_layout=NHWC\ninput_type=uint8\ninput_shape="
             << model_height << ',' << model_width << ",3\n"
             << "source_width=" << options.source_width << '\n'
             << "source_height=" << options.source_height << '\n'
             << "letterbox_x_pad=" << pad_x << '\n'
             << "letterbox_y_pad=" << pad_y << '\n'
             << "letterbox_scale=" << std::setprecision(10) << scale << '\n'
             << "output_requested_type=float32\n";

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::ostringstream filename;
        filename << "output-" << std::setw(2) << std::setfill('0') << i << ".f32.bin";
        std::ofstream output(root / filename.str(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(outputs[i].data()),
                     static_cast<std::streamsize>(outputs[i].size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed to write output tensor");
        const auto& attr = model.output_attrs()[i];
        manifest << "output[" << i << "].name=" << attr.name << '\n'
                 << "output[" << i << "].shape=";
        for (std::uint32_t dim = 0; dim < attr.n_dims; ++dim) {
            if (dim != 0) manifest << ',';
            manifest << attr.dims[dim];
        }
        manifest << '\n'
                 << "output[" << i << "].type=" << tensor_type_name(attr.type) << '\n'
                 << "output[" << i << "].quantization=" << quantization_name(attr.qnt_type) << '\n'
                 << "output[" << i << "].scale=" << attr.scale << '\n'
                 << "output[" << i << "].zero_point=" << attr.zp << '\n'
                 << "output[" << i << "].file=" << filename.str() << '\n';
    }
    manifest.flush();
    if (!manifest) throw std::runtime_error("failed to flush tensor manifest");
}

struct Metrics {
    double mean_abs = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    double cosine = 1.0;
};

object_detect_result_list decode_detections(const Model& model, const TensorOutputs& outputs,
                                           int source_width, int source_height, float confidence) {
    rknn_app_context_t app_context{};
    app_context.io_num = model.io_num();
    app_context.output_attrs = const_cast<rknn_tensor_attr*>(model.output_attrs().data());
    app_context.model_width = static_cast<int>(model.input_attr().dims[2]);
    app_context.model_height = static_cast<int>(model.input_attr().dims[1]);
    app_context.is_quant = false;  // rknn_outputs_get(want_float=1) supplies dequantized FP32 buffers.

    std::vector<rknn_output> output_views(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        output_views[i] = {};
        output_views[i].index = static_cast<std::uint32_t>(i);
        output_views[i].want_float = 1;
        output_views[i].buf = const_cast<float*>(outputs[i].data());
        output_views[i].size = static_cast<std::uint32_t>(outputs[i].size() * sizeof(float));
    }

    const float scale = std::min(static_cast<float>(app_context.model_width) / source_width,
                                 static_cast<float>(app_context.model_height) / source_height);
    const int resized_width = static_cast<int>(std::round(source_width * scale));
    const int resized_height = static_cast<int>(std::round(source_height * scale));
    letterbox_t letterbox{};
    letterbox.x_pad = (app_context.model_width - resized_width) / 2;
    letterbox.y_pad = (app_context.model_height - resized_height) / 2;
    letterbox.scale = scale;
    object_detect_result_list detections{};
    const int status = post_process(&app_context, output_views.data(), &letterbox, confidence,
                                    NMS_THRESH, &detections);
    if (status != 0) throw std::runtime_error("RKNN Model Zoo post_process failed");
    return detections;
}

double detection_iou(const object_detect_result& first, const object_detect_result& second) {
    const int left = std::max(first.box.left, second.box.left);
    const int top = std::max(first.box.top, second.box.top);
    const int right = std::min(first.box.right, second.box.right);
    const int bottom = std::min(first.box.bottom, second.box.bottom);
    const double intersection = static_cast<double>(std::max(0, right - left)) *
                                static_cast<double>(std::max(0, bottom - top));
    const double first_area = static_cast<double>(std::max(0, first.box.right - first.box.left)) *
                              static_cast<double>(std::max(0, first.box.bottom - first.box.top));
    const double second_area = static_cast<double>(std::max(0, second.box.right - second.box.left)) *
                               static_cast<double>(std::max(0, second.box.bottom - second.box.top));
    const double union_area = first_area + second_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

void print_detections(std::ostream& output, const object_detect_result_list& detections) {
    output << '[';
    for (int i = 0; i < detections.count; ++i) {
        if (i != 0) output << ',';
        const auto& item = detections.results[i];
        output << "{\"class_id\":" << item.cls_id
               << ",\"confidence\":" << item.prop
               << ",\"box\":[" << item.box.left << ',' << item.box.top << ','
               << item.box.right << ',' << item.box.bottom << "]}";
    }
    output << ']';
}

void print_detection_agreement(std::ostream& output,
                               const object_detect_result_list& reference,
                               const object_detect_result_list& candidate,
                               double match_iou) {
    std::vector<bool> candidate_used(static_cast<std::size_t>(candidate.count), false);
    int matched = 0;
    double sum_iou = 0.0;
    double sum_confidence_delta = 0.0;
    for (int i = 0; i < reference.count; ++i) {
        int best_index = -1;
        double best_iou = -1.0;
        for (int j = 0; j < candidate.count; ++j) {
            if (candidate_used[static_cast<std::size_t>(j)] ||
                reference.results[i].cls_id != candidate.results[j].cls_id) continue;
            const double overlap = detection_iou(reference.results[i], candidate.results[j]);
            if (overlap > best_iou) {
                best_iou = overlap;
                best_index = j;
            }
        }
        if (best_index >= 0 && best_iou >= match_iou) {
            candidate_used[static_cast<std::size_t>(best_index)] = true;
            ++matched;
            sum_iou += best_iou;
            sum_confidence_delta += std::abs(
                static_cast<double>(reference.results[i].prop) -
                static_cast<double>(candidate.results[best_index].prop));
        }
    }
    const int denominator = reference.count + candidate.count;
    const double agreement = denominator == 0 ? 1.0 : 2.0 * matched / denominator;
    output << "{\"reference_count\":" << reference.count
           << ",\"candidate_count\":" << candidate.count
           << ",\"matched_count\":" << matched
           << ",\"unmatched_reference_count\":" << (reference.count - matched)
           << ",\"unmatched_candidate_count\":" << (candidate.count - matched)
           << ",\"set_f1_agreement\":" << agreement
           << ",\"matched_mean_iou\":"
           << (matched == 0 ? 0.0 : sum_iou / matched)
           << ",\"matched_mean_absolute_confidence_delta\":"
           << (matched == 0 ? 0.0 : sum_confidence_delta / matched) << '}';
}

Metrics compare(const std::vector<float>& first, const std::vector<float>& second) {
    if (first.size() != second.size()) throw std::runtime_error("output shape/size mismatch between models");
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double maximum = 0.0;
    double dot = 0.0;
    double norm_first = 0.0;
    double norm_second = 0.0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const double a = first[i];
        const double b = second[i];
        const double delta = std::abs(a - b);
        sum_abs += delta;
        sum_sq += delta * delta;
        maximum = std::max(maximum, delta);
        dot += a * b;
        norm_first += a * a;
        norm_second += b * b;
    }
    const double count = static_cast<double>(first.size());
    return {sum_abs / count, std::sqrt(sum_sq / count), maximum,
            dot / (std::sqrt(norm_first * norm_second) + 1e-12)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto input = read_bytes(options.input);
        constexpr std::size_t expected_bytes = 640U * 640U * 3U;
        if (input.size() != expected_bytes) throw std::runtime_error("input tensor must be 640x640x3 uint8");

        Model reference(options.reference_model);
        Model candidate(options.candidate_model);
        if (reference.input_attr().n_dims != candidate.input_attr().n_dims ||
            !std::equal(reference.input_attr().dims,
                        reference.input_attr().dims + reference.input_attr().n_dims,
                        candidate.input_attr().dims)) {
            throw std::runtime_error("model input tensor shapes differ");
        }
        for (std::size_t i = 0; i < reference.output_attrs().size(); ++i) {
            const auto& a = reference.output_attrs()[i];
            const auto& b = candidate.output_attrs()[i];
            if (a.n_dims != b.n_dims || !std::equal(a.dims, a.dims + a.n_dims, b.dims)) {
                throw std::runtime_error("model output tensor shape mismatch at index " + std::to_string(i));
            }
        }

        double reference_run_ms = 0.0;
        double reference_total_ms = 0.0;
        double candidate_run_ms = 0.0;
        double candidate_total_ms = 0.0;
        const TensorOutputs reference_outputs = reference.run(input, &reference_run_ms, &reference_total_ms);
        const TensorOutputs candidate_outputs = candidate.run(input, &candidate_run_ms, &candidate_total_ms);
        const std::filesystem::path root(options.output_dir);
        save_outputs(reference_outputs, reference, options, root / "reference");
        save_outputs(candidate_outputs, candidate, options, root / "candidate");

        std::cout << std::fixed << std::setprecision(9)
                  << "{\"type\":\"rknn_model_comparison\",\"input_bytes\":" << input.size()
                  << ",\"reference_model_bytes\":" << reference.model_bytes().size()
                  << ",\"candidate_model_bytes\":" << candidate.model_bytes().size()
                  << ",\"reference_input_type\":\""
                  << tensor_type_name(reference.input_attr().type) << "\""
                  << ",\"candidate_input_type\":\""
                  << tensor_type_name(candidate.input_attr().type) << "\""
                  << ",\"reference_input_size\":" << reference.input_attr().size
                  << ",\"candidate_input_size\":" << candidate.input_attr().size
                  << ",\"reference_run_ms\":" << reference_run_ms
                  << ",\"reference_run_plus_fetch_ms\":" << reference_total_ms
                  << ",\"candidate_run_ms\":" << candidate_run_ms
                  << ",\"candidate_run_plus_fetch_ms\":" << candidate_total_ms
                  << ",\"outputs\":[";
        for (std::size_t i = 0; i < reference_outputs.size(); ++i) {
            if (i != 0) std::cout << ',';
            const Metrics metrics = compare(reference_outputs[i], candidate_outputs[i]);
            std::cout << "{\"index\":" << i << ",\"elements\":" << reference_outputs[i].size()
                      << ",\"mean_abs_difference\":" << metrics.mean_abs
                      << ",\"rmse\":" << metrics.rmse
                      << ",\"max_abs_difference\":" << metrics.max_abs
                      << ",\"cosine_similarity\":" << metrics.cosine << '}';
        }
        std::cout << "],\"detection_comparisons\":[";
        const std::vector<float> confidence_thresholds{0.25F, 0.345F, 0.50F};
        for (std::size_t i = 0; i < confidence_thresholds.size(); ++i) {
            const float threshold = confidence_thresholds[i];
            const auto reference_detections = decode_detections(
                reference, reference_outputs, options.source_width, options.source_height, threshold);
            const auto candidate_detections = decode_detections(
                candidate, candidate_outputs, options.source_width, options.source_height, threshold);
            if (i != 0) std::cout << ',';
            std::cout << "{\"confidence_threshold\":" << threshold << ",\"agreement\":";
            print_detection_agreement(std::cout, reference_detections, candidate_detections, 0.50);
            std::cout << ",\"reference_detections\":";
            print_detections(std::cout, reference_detections);
            std::cout << ",\"candidate_detections\":";
            print_detections(std::cout, candidate_detections);
            std::cout << '}';
        }
        std::cout << "],\"note\":\"One paired inference per model for diagnostics; run timings are single observations, not latency statistics. Detection sets use the same vendored C++ postprocess source as the worker and are consistency evidence, not labeled accuracy.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RKNN model comparison error: " << error.what() << '\n';
        return 2;
    }
}
