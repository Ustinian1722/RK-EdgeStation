#include <rknn_api.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open model: " + path);
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error("model is empty: " + path);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read complete model: " + path);
    }
    return bytes;
}

double ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    const auto index = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

std::string dims_string(const rknn_tensor_attr& attr) {
    std::string result = "[";
    for (std::uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i != 0) result += ',';
        result += std::to_string(attr.dims[i]);
    }
    return result + ']';
}

struct LetterboxInfo {
    double scale = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};

struct CandidateDetection {
    int class_id = -1;
    float score = 0.0F;
    float left = 0.0F;
    float top = 0.0F;
    float right = 0.0F;
    float bottom = 0.0F;
};

std::vector<std::uint8_t> make_input(const cv::Mat& bgr, int width, int height,
                                     bool channels_first, LetterboxInfo& info) {
    const double scale = std::min(static_cast<double>(width) / bgr.cols,
                                  static_cast<double>(height) / bgr.rows);
    const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    cv::Mat resized_bgr;
    cv::resize(bgr, resized_bgr, cv::Size(resized_width, resized_height), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::Mat boxed_bgr(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
    const int pad_x = (width - resized_width) / 2;
    const int pad_y = (height - resized_height) / 2;
    info = {scale, pad_x, pad_y};
    resized_bgr.copyTo(boxed_bgr(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    cv::Mat boxed_rgb;
    cv::cvtColor(boxed_bgr, boxed_rgb, cv::COLOR_BGR2RGB);
    if (!channels_first) {
        if (!boxed_rgb.isContinuous()) boxed_rgb = boxed_rgb.clone();
        const auto* begin = boxed_rgb.ptr<std::uint8_t>();
        return {begin, begin + boxed_rgb.total() * boxed_rgb.elemSize()};
    }
    std::vector<std::uint8_t> planar(static_cast<std::size_t>(width) * height * 3);
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    for (int y = 0; y < height; ++y) {
        const auto* row = boxed_rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const auto offset = static_cast<std::size_t>(y) * width + x;
            planar[offset] = row[x][0];
            planar[plane + offset] = row[x][1];
            planar[plane * 2 + offset] = row[x][2];
        }
    }
    return planar;
}

float box_iou(const CandidateDetection& first, const CandidateDetection& second) {
    const float left = std::max(first.left, second.left);
    const float top = std::max(first.top, second.top);
    const float right = std::min(first.right, second.right);
    const float bottom = std::min(first.bottom, second.bottom);
    const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float area_first = std::max(0.0F, first.right - first.left) *
                             std::max(0.0F, first.bottom - first.top);
    const float area_second = std::max(0.0F, second.right - second.left) *
                              std::max(0.0F, second.bottom - second.top);
    const float union_area = area_first + area_second - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<CandidateDetection> decode_yolov8(const float* values,
                                               const rknn_tensor_attr& attr,
                                               int width, int height,
                                               const LetterboxInfo& letterbox,
                                               const std::map<int, int>& class_map,
                                               float nms_threshold) {
    if (values == nullptr || attr.n_dims != 3 || attr.dims[0] != 1) {
        throw std::runtime_error("YOLOv8 candidate output must be one rank-3 tensor");
    }
    const bool channels_first = attr.dims[1] < attr.dims[2];
    const int channels = static_cast<int>(channels_first ? attr.dims[1] : attr.dims[2]);
    const int anchors = static_cast<int>(channels_first ? attr.dims[2] : attr.dims[1]);
    if (channels < 5 || anchors <= 0) throw std::runtime_error("invalid YOLOv8 output dimensions");
    const int class_count = channels - 4;
    const auto value_at = [&](int channel, int anchor) -> float {
        return channels_first
            ? values[static_cast<std::size_t>(channel) * anchors + anchor]
            : values[static_cast<std::size_t>(anchor) * channels + channel];
    };
    std::vector<CandidateDetection> candidates;
    candidates.reserve(static_cast<std::size_t>(anchors / 4));
    for (int anchor = 0; anchor < anchors; ++anchor) {
        int source_class = 0;
        float score = value_at(4, anchor);
        for (int cls = 1; cls < class_count; ++cls) {
            const float candidate = value_at(4 + cls, anchor);
            if (candidate > score) {
                score = candidate;
                source_class = cls;
            }
        }
        if (!std::isfinite(score) || score < 0.001F) continue;
        const auto mapping = class_map.find(source_class);
        if (mapping == class_map.end()) continue;
        const float center_x = value_at(0, anchor);
        const float center_y = value_at(1, anchor);
        const float box_width = value_at(2, anchor);
        const float box_height = value_at(3, anchor);
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(box_width) || !std::isfinite(box_height) ||
            box_width <= 0.0F || box_height <= 0.0F) {
            continue;
        }
        CandidateDetection detection;
        detection.class_id = mapping->second;
        detection.score = score;
        detection.left = std::clamp(static_cast<float>((center_x - box_width * 0.5F - letterbox.pad_x) /
                                                       letterbox.scale), 0.0F, static_cast<float>(width));
        detection.top = std::clamp(static_cast<float>((center_y - box_height * 0.5F - letterbox.pad_y) /
                                                      letterbox.scale), 0.0F, static_cast<float>(height));
        detection.right = std::clamp(static_cast<float>((center_x + box_width * 0.5F - letterbox.pad_x) /
                                                        letterbox.scale), 0.0F, static_cast<float>(width));
        detection.bottom = std::clamp(static_cast<float>((center_y + box_height * 0.5F - letterbox.pad_y) /
                                                         letterbox.scale), 0.0F, static_cast<float>(height));
        if (detection.right > detection.left && detection.bottom > detection.top) {
            candidates.push_back(detection);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    std::vector<CandidateDetection> kept;
    kept.reserve(300);
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (candidate.class_id == selected.class_id && box_iou(candidate, selected) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
        if (kept.size() == 300) break;
    }
    return kept;
}

std::map<int, int> parse_class_map(const std::string& value) {
    std::map<int, int> mapping;
    std::stringstream input(value);
    for (std::string item; std::getline(input, item, ',');) {
        const auto colon = item.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("class map must use src:canonical pairs");
        mapping.emplace(std::stoi(item.substr(0, colon)), std::stoi(item.substr(colon + 1)));
    }
    return mapping;
}

void write_prediction(std::ostream& stream, std::size_t index, const std::string& name,
                      int width, int height, const std::vector<CandidateDetection>& detections) {
    stream << std::fixed << std::setprecision(6)
           << "{\"index\":" << index << ",\"file_name\":\"" << name
           << "\",\"width\":" << width << ",\"height\":" << height << ",\"detections\":[";
    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (i != 0) stream << ',';
        const auto& item = detections[i];
        stream << "{\"class_id\":" << item.class_id << ",\"score\":" << item.score
               << ",\"bbox_xyxy\":[" << item.left << ',' << item.top << ','
               << item.right << ',' << item.bottom << "]}";
    }
    stream << "]}\n";
}

double timeval_seconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1.0e6;
}

long peak_rss_kb() {
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
    return usage.ru_maxrss;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
                throw std::invalid_argument("invalid argument near " + key);
            }
            args[key] = argv[++i];
        }
        if (!args.count("--model") || !args.count("--output-prefix") ||
            (args.count("--video") == args.count("--image-manifest"))) {
            std::cerr << "usage: rknn_yolov8_raw_bench --model MODEL.rknn "
                         "(--video INPUT.mp4 | --image-manifest ROOT_AND_FILES.txt) "
                         "--output-prefix PREFIX [--class-map src:target,...] [--nms-iou X]\n";
            return 2;
        }
        const std::string model_path = args.at("--model");
        const std::string prefix = args.at("--output-prefix");
        const bool video_mode = args.count("--video") != 0;
        const std::string input_path = video_mode ? args.at("--video") : args.at("--image-manifest");
        const auto class_map = args.count("--class-map")
            ? parse_class_map(args.at("--class-map")) : std::map<int, int>{};
        const float nms_threshold = args.count("--nms-iou") ? std::stof(args.at("--nms-iou")) : 0.45F;
        if (nms_threshold < 0.0F || nms_threshold > 1.0F) {
            throw std::invalid_argument("--nms-iou must be between 0 and 1");
        }
        const auto model_bytes = read_file(model_path);
        rknn_context context = 0;
        int status = rknn_init(&context, const_cast<std::uint8_t*>(model_bytes.data()),
                               static_cast<std::uint32_t>(model_bytes.size()), 0, nullptr);
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_init failed: " + std::to_string(status));
        struct ContextGuard {
            rknn_context value;
            ~ContextGuard() { if (value != 0) rknn_destroy(value); }
        } context_guard{context};

        rknn_input_output_num io{};
        status = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        if (status != RKNN_SUCC || io.n_input != 1 || io.n_output == 0) {
            throw std::runtime_error("unsupported input/output count; query=" + std::to_string(status));
        }
        rknn_tensor_attr input_attr{};
        input_attr.index = 0;
        status = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
        if (status != RKNN_SUCC || input_attr.n_dims != 4) {
            throw std::runtime_error("unsupported RKNN input tensor; query=" + std::to_string(status));
        }
        int width = 0, height = 0;
        bool channels_first = false;
        if (input_attr.fmt == RKNN_TENSOR_NHWC && input_attr.dims[0] == 1 && input_attr.dims[3] == 3) {
            height = static_cast<int>(input_attr.dims[1]);
            width = static_cast<int>(input_attr.dims[2]);
        } else if (input_attr.fmt == RKNN_TENSOR_NCHW && input_attr.dims[0] == 1 && input_attr.dims[1] == 3) {
            channels_first = true;
            height = static_cast<int>(input_attr.dims[2]);
            width = static_cast<int>(input_attr.dims[3]);
        } else {
            throw std::runtime_error("only fixed 1x3xHxW or 1xHxWx3 image inputs are supported");
        }
        if (width <= 0 || height <= 0) throw std::runtime_error("invalid model image size");

        std::vector<rknn_tensor_attr> output_attrs(io.n_output);
        for (std::uint32_t i = 0; i < io.n_output; ++i) {
            output_attrs[i] = {};
            output_attrs[i].index = i;
            status = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
            if (status != RKNN_SUCC) throw std::runtime_error("query output tensor failed: " + std::to_string(status));
        }

        cv::VideoCapture capture;
        std::ifstream manifest;
        std::string image_root;
        std::vector<std::string> image_names;
        double nominal_fps = 0.0;
        if (video_mode) {
            capture.open(input_path);
            if (!capture.isOpened()) throw std::runtime_error("cannot open video: " + input_path);
            nominal_fps = capture.get(cv::CAP_PROP_FPS);
        } else {
            manifest.open(input_path);
            if (!manifest || !std::getline(manifest, image_root) || image_root.empty()) {
                throw std::runtime_error("image manifest must start with an image root path");
            }
            for (std::string name; std::getline(manifest, name);) {
                if (!name.empty() && name.back() == '\r') name.pop_back();
                if (!name.empty() && name.front() != '#') image_names.push_back(name);
            }
            if (image_names.empty()) throw std::runtime_error("image manifest has no images");
        }
        std::vector<double> prep_ms, input_ms, run_ms, fetch_ms;
        std::ofstream predictions(prefix + ".predictions.jsonl");
        if (!predictions) throw std::runtime_error("cannot create prediction JSONL");
        std::uint64_t frame_count = 0;
        std::size_t image_index = 0;
        const auto wall_start = Clock::now();
        struct rusage cpu_start{};
        getrusage(RUSAGE_SELF, &cpu_start);
        while (true) {
            cv::Mat bgr;
            std::string name;
            if (video_mode) {
                if (!capture.read(bgr)) break;
                std::ostringstream generated_name;
                generated_name << "frame-" << std::setw(6) << std::setfill('0') << frame_count << ".jpg";
                name = generated_name.str();
            } else {
                if (image_index >= image_names.size()) break;
                name = image_names[image_index++];
                bgr = cv::imread(image_root + "/" + name, cv::IMREAD_COLOR);
                if (bgr.empty()) throw std::runtime_error("cannot decode image: " + image_root + "/" + name);
            }
            LetterboxInfo letterbox{};
            const auto prep_start = Clock::now();
            auto input_bytes = make_input(bgr, width, height, channels_first, letterbox);
            prep_ms.push_back(ms(Clock::now() - prep_start));

            rknn_input input{};
            input.index = 0;
            input.type = RKNN_TENSOR_UINT8;
            input.fmt = channels_first ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
            input.size = static_cast<std::uint32_t>(input_bytes.size());
            input.buf = input_bytes.data();
            auto started = Clock::now();
            status = rknn_inputs_set(context, 1, &input);
            input_ms.push_back(ms(Clock::now() - started));
            if (status != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(status));

            started = Clock::now();
            status = rknn_run(context, nullptr);
            run_ms.push_back(ms(Clock::now() - started));
            if (status != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(status));

            std::vector<rknn_output> outputs(io.n_output);
            for (std::uint32_t i = 0; i < io.n_output; ++i) {
                outputs[i] = {};
                outputs[i].index = i;
                const auto type = output_attrs[i].type;
                const bool quantized = type == RKNN_TENSOR_INT8 || type == RKNN_TENSOR_UINT8;
                outputs[i].want_float = quantized ? 0 : 1;
            }
            started = Clock::now();
            status = rknn_outputs_get(context, io.n_output, outputs.data(), nullptr);
            fetch_ms.push_back(ms(Clock::now() - started));
            if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(status));
            if (io.n_output != 1) throw std::runtime_error("released YOLOv8 evaluator expects one raw output tensor");
            const auto& output_attr = output_attrs[0];
            std::vector<float> dequantized;
            const float* output_values = nullptr;
            const bool quantized_output = output_attr.type == RKNN_TENSOR_INT8 ||
                                          output_attr.type == RKNN_TENSOR_UINT8;
            if (quantized_output) {
                if (output_attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
                    output_attr.scale <= 0.0F || outputs[0].size < output_attr.n_elems) {
                    throw std::runtime_error("unsupported quantized output scale/zero-point contract");
                }
                dequantized.resize(output_attr.n_elems);
                if (output_attr.type == RKNN_TENSOR_INT8) {
                    const auto* quantized_values = static_cast<const std::int8_t*>(outputs[0].buf);
                    for (std::size_t i = 0; i < dequantized.size(); ++i) {
                        dequantized[i] = (static_cast<int>(quantized_values[i]) - output_attr.zp) * output_attr.scale;
                    }
                } else {
                    const auto* quantized_values = static_cast<const std::uint8_t*>(outputs[0].buf);
                    for (std::size_t i = 0; i < dequantized.size(); ++i) {
                        dequantized[i] = (static_cast<int>(quantized_values[i]) - output_attr.zp) * output_attr.scale;
                    }
                }
                output_values = dequantized.data();
            } else {
                if (outputs[0].size < output_attr.n_elems * sizeof(float)) {
                    throw std::runtime_error("float output buffer is smaller than queried tensor size");
                }
                output_values = static_cast<const float*>(outputs[0].buf);
            }
            if (frame_count == 0) {
                std::ofstream input_file(prefix + ".input0.rgb-or-chw-u8.bin", std::ios::binary);
                input_file.write(reinterpret_cast<const char*>(input_bytes.data()),
                                 static_cast<std::streamsize>(input_bytes.size()));
                if (!input_file) throw std::runtime_error("cannot write first model input tensor");
                for (std::uint32_t i = 0; i < io.n_output; ++i) {
                    const std::string path = prefix + ".output" + std::to_string(i) + ".f32.bin";
                    std::ofstream output_file(path, std::ios::binary);
                    const std::size_t element_count = output_attrs[i].n_elems;
                    output_file.write(reinterpret_cast<const char*>(i == 0 ? output_values : nullptr),
                                      static_cast<std::streamsize>(element_count * sizeof(float)));
                    if (!output_file) throw std::runtime_error("cannot write float output tensor: " + path);
                }
            }
            std::map<int, int> active_map = class_map;
            if (active_map.empty()) {
                const int channels = static_cast<int>(output_attrs[0].dims[1] < output_attrs[0].dims[2]
                    ? output_attrs[0].dims[1] : output_attrs[0].dims[2]);
                for (int cls = 0; cls < channels - 4; ++cls) active_map.emplace(cls, cls);
            }
            const auto detections = decode_yolov8(output_values, output_attr,
                bgr.cols, bgr.rows, letterbox, active_map, nms_threshold);
            write_prediction(predictions, static_cast<std::size_t>(frame_count), name,
                             bgr.cols, bgr.rows, detections);
            status = rknn_outputs_release(context, io.n_output, outputs.data());
            if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_release failed: " + std::to_string(status));
            ++frame_count;
        }
        predictions.flush();
        if (!predictions) throw std::runtime_error("failed writing predictions JSONL");
        const auto wall_seconds = std::chrono::duration<double>(Clock::now() - wall_start).count();
        struct rusage cpu_end{};
        getrusage(RUSAGE_SELF, &cpu_end);
        const double cpu_seconds = timeval_seconds(cpu_end.ru_utime) + timeval_seconds(cpu_end.ru_stime) -
                                   timeval_seconds(cpu_start.ru_utime) - timeval_seconds(cpu_start.ru_stime);
        const long cpu_count = std::max(1L, sysconf(_SC_NPROCESSORS_ONLN));

        std::ofstream report(prefix + ".json");
        if (!report) throw std::runtime_error("cannot open report output");
        report << std::fixed << std::setprecision(4)
               << "{\n  \"scope\":\"isolated-rknn-offline-no-pacing\",\n"
               << "  \"model\":\"" << model_path << "\",\n"
               << "  \"input\":\"" << input_path << "\",\n"
               << "  \"mode\":\"" << (video_mode ? "video" : "image-manifest") << "\",\n"
               << "  \"confidence_floor\":0.001,\n"
               << "  \"nms_iou\":" << nms_threshold << ",\n"
               << "  \"frames\":" << frame_count << ",\n"
               << "  \"nominal_fps\":" << nominal_fps << ",\n"
               << "  \"model_input\":{\"dims\":\"" << dims_string(input_attr)
               << "\",\"format\":" << input_attr.fmt << ",\"type\":" << input_attr.type
               << ",\"qnt_type\":" << input_attr.qnt_type << ",\"scale\":" << input_attr.scale
               << ",\"zero_point\":" << input_attr.zp << ",\"size\":" << input_attr.size << ",\"width\":" << width
               << ",\"height\":" << height << ",\"color_order\":\"RGB\",\"input_pixels\":\"uint8 0..255; Toolkit2 mean=0 std=255\"},\n"
               << "  \"outputs\":[";
        for (std::uint32_t i = 0; i < io.n_output; ++i) {
            if (i != 0) report << ',';
            report << "{\"index\":" << i << ",\"dims\":\"" << dims_string(output_attrs[i])
                   << "\",\"format\":" << output_attrs[i].fmt << ",\"type\":" << output_attrs[i].type
                   << ",\"qnt_type\":" << output_attrs[i].qnt_type
                   << ",\"scale\":" << output_attrs[i].scale
                   << ",\"zero_point\":" << output_attrs[i].zp
                   << ",\"elements\":" << output_attrs[i].n_elems << '}';
        }
        report << "],\n  \"timing_ms\":{\"preprocess_p50\":" << percentile(prep_ms, 0.50)
               << ",\"preprocess_p95\":" << percentile(prep_ms, 0.95)
               << ",\"input_set_p50\":" << percentile(input_ms, 0.50)
               << ",\"input_set_p95\":" << percentile(input_ms, 0.95)
               << ",\"run_p50\":" << percentile(run_ms, 0.50)
               << ",\"run_p95\":" << percentile(run_ms, 0.95)
               << ",\"fetch_p50\":" << percentile(fetch_ms, 0.50)
               << ",\"fetch_p95\":" << percentile(fetch_ms, 0.95) << "},\n"
               << "  \"wall_seconds\":" << wall_seconds << ",\n"
               << "  \"throughput_fps\":" << (wall_seconds > 0.0 ? frame_count / wall_seconds : 0.0) << ",\n"
               << "  \"process_cpu_core_percent\":" << (wall_seconds > 0.0 ? cpu_seconds / wall_seconds * 100.0 : 0.0) << ",\n"
               << "  \"process_cpu_system_percent\":" << (wall_seconds > 0.0 ? cpu_seconds / wall_seconds * 100.0 / cpu_count : 0.0) << ",\n"
               << "  \"process_cpu_seconds\":" << cpu_seconds << ",\n"
               << "  \"peak_rss_kb\":" << peak_rss_kb() << "\n}\n";
        std::cout << "model input_dims=" << dims_string(input_attr) << " input_fmt=" << input_attr.fmt
                  << " input_type=" << input_attr.type << " input_size=" << input_attr.size
                  << " output_count=" << io.n_output << " frames=" << frame_count << '\n';
        std::cout << "report=" << prefix << ".json throughput_fps="
                  << (wall_seconds > 0.0 ? frame_count / wall_seconds : 0.0)
                  << " run_p50_ms=" << percentile(run_ms, 0.50)
                  << " run_p95_ms=" << percentile(run_ms, 0.95) << '\n';
        return frame_count > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "rknn_yolov8_raw_bench: " << error.what() << '\n';
        return 1;
    }
}
