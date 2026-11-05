#include "bounded_queue.hpp"
#include "ppe_events.hpp"
#include "roi_events.hpp"
#include "tracker.hpp"
#include "yolov8.h"
#include "yolox_ppe_postprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

volatile std::sig_atomic_t shutdown_requested = 0;
std::mutex stdout_mutex;

void request_shutdown(int) { shutdown_requested = 1; }

bool is_rtsp_source(const std::string& source) {
    std::string prefix = source.substr(0, std::min<std::size_t>(source.size(), 8));
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return prefix.rfind("rtsp://", 0) == 0 || prefix.rfind("rtsps://", 0) == 0;
}

bool wait_for_retry(std::chrono::milliseconds duration) {
    const auto end = Clock::now() + duration;
    while (!shutdown_requested && Clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return !shutdown_requested;
}

std::vector<int> rtsp_open_parameters() {
    return {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
            cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000};
}

struct Frame {
    cv::Mat image;
    std::uint64_t sequence = 0;
    double pts_ms = 0.0;
    std::uint64_t stream_epoch = 0;
    Clock::time_point decoded_at;
};

struct Roi {
    double x = 0.0;
    double y = 0.0;
    double width = 1.0;
    double height = 1.0;

    bool contains(const edge::Detection& detection, int frame_width, int frame_height) const {
        const double center_x = (detection.left + detection.right) * 0.5 / frame_width;
        const double center_y = (detection.top + detection.bottom) * 0.5 / frame_height;
        return center_x >= x && center_x <= x + width && center_y >= y && center_y <= y + height;
    }
};

struct InferenceResult {
    std::vector<edge::Detection> detections;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

enum class ModelFamily { YoloV8Coco, YoloXPpe };

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec;
                } else {
                    out << static_cast<char>(ch);
                }
        }
    }
    return out.str();
}

std::string utc_timestamp(std::chrono::system_clock::time_point now) {
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&raw, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setw(3) << std::setfill('0') << millis.count() << 'Z';
    return out.str();
}

std::string basename(const std::string& path) { return fs::path(path).filename().string(); }

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

int read_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            int value = 0;
            status >> value;
            return value;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open model file: " + path);
    }
    const auto end = input.tellg();
    if (end <= 0 || static_cast<std::uint64_t>(end) > 0x7fffffffULL) {
        throw std::runtime_error("invalid model file size");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error("failed to read complete model file");
    }
    return bytes;
}

class RknnYoloEngine {
public:
    explicit RknnYoloEngine(const std::string& model_path, std::string raw_dump_dir,
                            ModelFamily model_family)
        : raw_dump_dir_(std::move(raw_dump_dir)), model_family_(model_family) {
        try {
            auto model = read_file(model_path);
            int ret = rknn_init(&context_.rknn_ctx, model.data(),
                                static_cast<std::uint32_t>(model.size()), 0, nullptr);
            if (ret < 0) {
                throw std::runtime_error("rknn_init failed with code " + std::to_string(ret));
            }

            ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM,
                             &context_.io_num, sizeof(context_.io_num));
            if (ret != RKNN_SUCC || context_.io_num.n_input != 1 || context_.io_num.n_output == 0 ||
                (model_family_ == ModelFamily::YoloXPpe && context_.io_num.n_output != 3)) {
                throw std::runtime_error("unsupported RKNN input/output contract");
            }

            input_attrs_.resize(context_.io_num.n_input);
            output_attrs_.resize(context_.io_num.n_output);
            for (std::uint32_t i = 0; i < context_.io_num.n_input; ++i) {
                input_attrs_[i] = {};
                input_attrs_[i].index = i;
                ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_INPUT_ATTR,
                                 &input_attrs_[i], sizeof(rknn_tensor_attr));
                if (ret != RKNN_SUCC) {
                    throw std::runtime_error("failed to query RKNN input tensor");
                }
            }
            for (std::uint32_t i = 0; i < context_.io_num.n_output; ++i) {
                output_attrs_[i] = {};
                output_attrs_[i].index = i;
                ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR,
                                 &output_attrs_[i], sizeof(rknn_tensor_attr));
                if (ret != RKNN_SUCC) {
                    throw std::runtime_error("failed to query RKNN output tensor");
                }
                if (model_family_ == ModelFamily::YoloXPpe) {
                    const auto& output = output_attrs_[i];
                    const bool nchw = output.n_dims == 4 && output.dims[0] == 1 &&
                                      output.dims[1] == 17 && output.fmt == RKNN_TENSOR_NCHW;
                    const bool nhwc = output.n_dims == 4 && output.dims[0] == 1 &&
                                      output.dims[3] == 17 && output.fmt == RKNN_TENSOR_NHWC;
                    if (!nchw && !nhwc) {
                        throw std::runtime_error("YOLOX-PPE requires three 17-channel NCHW/NHWC output maps");
                    }
                }
            }

            context_.input_attrs = input_attrs_.data();
            context_.output_attrs = output_attrs_.data();
            const auto& input = input_attrs_.front();
            if (input.n_dims != 4 || input.fmt != RKNN_TENSOR_NHWC) {
                throw std::runtime_error("only one NHWC image input is supported in P1");
            }
            context_.model_height = static_cast<int>(input.dims[1]);
            context_.model_width = static_cast<int>(input.dims[2]);
            context_.model_channel = static_cast<int>(input.dims[3]);
            if (context_.model_channel != 3 || context_.model_width <= 0 || context_.model_height <= 0) {
                throw std::runtime_error("invalid RKNN image input dimensions");
            }
            context_.is_quant = output_attrs_.front().qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                output_attrs_.front().type == RKNN_TENSOR_INT8;
            if (!raw_dump_dir_.empty() && context_.is_quant) {
                throw std::runtime_error("raw RKNN output dump currently supports fp models only");
            }
            model_name_ = basename(model_path);
            std::cerr << "RKNN model=" << model_name_
                      << " input=" << context_.model_width << 'x' << context_.model_height
                      << " channels=" << context_.model_channel
                      << " outputs=" << context_.io_num.n_output
                      << " quantized=" << (context_.is_quant ? "true" : "false")
                      << " family=" << (model_family_ == ModelFamily::YoloXPpe ? "yolox-ppe" : "yolov8-coco")
                      << '\n';
        } catch (...) {
            if (context_.rknn_ctx != 0) {
                rknn_destroy(context_.rknn_ctx);
                context_.rknn_ctx = 0;
            }
            throw;
        }
    }

    ~RknnYoloEngine() {
        if (context_.rknn_ctx != 0) {
            const int ret = rknn_destroy(context_.rknn_ctx);
            if (ret != RKNN_SUCC) {
                std::cerr << "rknn_destroy returned " << ret << '\n';
            }
        }
    }

    RknnYoloEngine(const RknnYoloEngine&) = delete;
    RknnYoloEngine& operator=(const RknnYoloEngine&) = delete;

    InferenceResult infer(const cv::Mat& bgr, float confidence) {
        InferenceResult result;
        const auto prep_start = Clock::now();
        const float scale = std::min(
            static_cast<float>(context_.model_width) / static_cast<float>(bgr.cols),
            static_cast<float>(context_.model_height) / static_cast<float>(bgr.rows));
        const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
        const int pad_x = (context_.model_width - resized_width) / 2;
        const int pad_y = (context_.model_height - resized_height) / 2;

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat padded(context_.model_height, context_.model_width, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
        cv::Mat rgb;
        const cv::Mat* model_input = &padded;
        const char* input_channel_order = "BGR";
        if (model_family_ == ModelFamily::YoloV8Coco) {
            cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
            model_input = &rgb;
            input_channel_order = "RGB";
        }

        letterbox_t letterbox{};
        letterbox.x_pad = pad_x;
        letterbox.y_pad = pad_y;
        letterbox.scale = scale;
        result.preprocess_ms = milliseconds(Clock::now() - prep_start);

        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<std::uint32_t>(model_input->total() * model_input->elemSize());
        input.buf = model_input->data;
        int ret = rknn_inputs_set(context_.rknn_ctx, context_.io_num.n_input, &input);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_inputs_set failed with code " + std::to_string(ret));
        }

        const auto inference_start = Clock::now();
        ret = rknn_run(context_.rknn_ctx, nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_run failed with code " + std::to_string(ret));
        }
        std::vector<rknn_output> outputs(context_.io_num.n_output);
        for (std::uint32_t i = 0; i < context_.io_num.n_output; ++i) {
            outputs[i] = {};
            outputs[i].index = i;
            outputs[i].want_float = context_.is_quant ? 0 : 1;
        }
        ret = rknn_outputs_get(context_.rknn_ctx, context_.io_num.n_output, outputs.data(), nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_outputs_get failed with code " + std::to_string(ret));
        }
        struct OutputRelease {
            rknn_context context;
            std::uint32_t count;
            rknn_output* outputs;
            ~OutputRelease() { rknn_outputs_release(context, count, outputs); }
        } output_release{context_.rknn_ctx, context_.io_num.n_output, outputs.data()};
        result.inference_ms = milliseconds(Clock::now() - inference_start);
        dump_first_outputs(*model_input, outputs, bgr.cols, bgr.rows, letterbox,
                           input_channel_order);

        const auto post_start = Clock::now();
        object_detect_result_list raw{};
        if (model_family_ == ModelFamily::YoloXPpe) {
            ret = post_process_yolox_ppe(&context_, outputs.data(), &letterbox, confidence,
                                         NMS_THRESH, bgr.cols, bgr.rows, &raw);
        } else {
            ret = post_process(&context_, outputs.data(), &letterbox, confidence, NMS_THRESH, &raw);
        }
        if (ret != 0) {
            throw std::runtime_error("YOLO postprocess failed with code " + std::to_string(ret));
        }
        result.postprocess_ms = milliseconds(Clock::now() - post_start);

        result.detections.reserve(raw.count);
        for (int i = 0; i < raw.count; ++i) {
            const auto& item = raw.results[i];
            const int left = std::clamp(item.box.left, 0, bgr.cols);
            const int top = std::clamp(item.box.top, 0, bgr.rows);
            const int right = std::clamp(item.box.right, left, bgr.cols);
            const int bottom = std::clamp(item.box.bottom, top, bgr.rows);
            const char* label = model_family_ == ModelFamily::YoloXPpe
                ? ppe_class_to_name(item.cls_id)
                : coco_cls_to_name(item.cls_id);
            result.detections.push_back({item.cls_id, item.prop, left, top, right, bottom, label});
        }
        dump_first_detections(result.detections);
        return result;
    }

private:
    void dump_first_outputs(const cv::Mat& model_input, const std::vector<rknn_output>& outputs,
                            int source_width, int source_height, const letterbox_t& letterbox,
                            const char* input_channel_order) {
        if (raw_dump_dir_.empty() || raw_dumped_) return;
        fs::create_directories(raw_dump_dir_);

        const fs::path root(raw_dump_dir_);
        const std::string input_filename = std::string("input-") +
            (model_family_ == ModelFamily::YoloXPpe ? "bgr" : "rgb") + "-u8.bin";
        const fs::path input_path = root / input_filename;
        std::ofstream input_file(input_path, std::ios::binary);
        const auto input_bytes = model_input.total() * model_input.elemSize();
        input_file.write(reinterpret_cast<const char*>(model_input.data),
                         static_cast<std::streamsize>(input_bytes));
        if (!input_file) throw std::runtime_error("failed to write raw RKNN input tensor");

        std::ofstream manifest(root / "tensor-manifest.txt");
        if (!manifest) throw std::runtime_error("failed to create raw RKNN tensor manifest");
        manifest << "model=" << model_name_ << '\n'
                 << "input_layout=NHWC\ninput_type=uint8\ninput_shape="
                 << context_.model_height << ',' << context_.model_width << ','
                 << context_.model_channel << '\n'
                 << "input_channel_order=" << input_channel_order << '\n'
                 << "input_file=" << input_path.filename().string() << '\n'
                 << "input_bytes=" << input_bytes << '\n'
                 << "source_width=" << source_width << '\n'
                 << "source_height=" << source_height << '\n'
                 << "letterbox_x_pad=" << letterbox.x_pad << '\n'
                 << "letterbox_y_pad=" << letterbox.y_pad << '\n'
                 << "letterbox_scale=" << std::setprecision(10) << letterbox.scale << '\n'
                 << "output_requested_type=float32\n";

        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].buf == nullptr || outputs[i].size == 0 || outputs[i].size % sizeof(float) != 0) {
                throw std::runtime_error("raw RKNN output is empty or not float32");
            }
            std::ostringstream filename;
            filename << "output-" << std::setw(2) << std::setfill('0') << i << ".f32.bin";
            std::ofstream output_file(root / filename.str(), std::ios::binary);
            output_file.write(static_cast<const char*>(outputs[i].buf), outputs[i].size);
            if (!output_file) throw std::runtime_error("failed to write raw RKNN output tensor");

            const auto& attr = output_attrs_[i];
            manifest << "output[" << i << "].name=" << attr.name << '\n'
                     << "output[" << i << "].shape=";
            for (std::uint32_t dim = 0; dim < attr.n_dims; ++dim) {
                if (dim != 0) manifest << ',';
                manifest << attr.dims[dim];
            }
            manifest << '\n'
                     << "output[" << i << "].type=" << get_type_string(attr.type) << '\n'
                     << "output[" << i << "].format=" << get_format_string(attr.fmt) << '\n'
                     << "output[" << i << "].quantization=" << get_qnt_type_string(attr.qnt_type) << '\n'
                     << "output[" << i << "].scale=" << attr.scale << '\n'
                     << "output[" << i << "].zero_point=" << attr.zp << '\n'
                     << "output[" << i << "].file=" << filename.str() << '\n'
                     << "output[" << i << "].bytes=" << outputs[i].size << '\n';
        }
        manifest.flush();
        if (!manifest) throw std::runtime_error("failed to flush raw RKNN tensor manifest");
        raw_dumped_ = true;
    }

    void dump_first_detections(const std::vector<edge::Detection>& detections) {
        if (raw_dump_dir_.empty() || detections_dumped_) return;
        std::ofstream output(fs::path(raw_dump_dir_) / "rknn-detections.csv");
        if (!output) throw std::runtime_error("failed to create raw RKNN detection dump");
        output << "class_id,label,confidence,left,top,right,bottom\n";
        output << std::fixed << std::setprecision(7);
        for (const auto& detection : detections) {
            output << detection.class_id << ',' << (detection.label ? detection.label : "unknown")
                   << ',' << detection.confidence << ',' << detection.left << ',' << detection.top
                   << ',' << detection.right << ',' << detection.bottom << '\n';
        }
        output.flush();
        if (!output) throw std::runtime_error("failed to flush raw RKNN detection dump");
        detections_dumped_ = true;
    }

    rknn_app_context_t context_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::string model_name_;
    std::string raw_dump_dir_;
    ModelFamily model_family_;
    bool raw_dumped_ = false;
    bool detections_dumped_ = false;
};

struct Options {
    std::string model;
    std::string video;
    std::string state_dir = "state";
    Roi roi;
    std::size_t queue_capacity = 1;
    edge::QueuePolicy queue_policy = edge::QueuePolicy::Latest;
    bool loop_video = false;
    float confidence = 0.25F;
    std::chrono::milliseconds dwell_threshold{5000};
    std::size_t roi_debounce_frames = 2;
    std::string raw_dump_dir;
    std::string model_family = "yolov8-coco";
    edge::NormalizedZone danger_zone;
    edge::NormalizedZone conflict_zone;
    std::size_t ppe_confirmation_frames = 3;
    std::chrono::milliseconds ppe_cooldown{30000};
    std::chrono::milliseconds conflict_dwell{1000};
};

edge::NormalizedZone parse_zone(const std::string& value, const char* option_name) {
    edge::NormalizedZone zone;
    std::stringstream input(value);
    char comma1 = 0, comma2 = 0, comma3 = 0;
    if (!(input >> zone.x >> comma1 >> zone.y >> comma2 >> zone.width >> comma3 >> zone.height) ||
        comma1 != ',' || comma2 != ',' || comma3 != ',') {
        throw std::invalid_argument(std::string(option_name) + " must be x,y,width,height in normalized coordinates");
    }
    return zone;
}

Options parse_args(int argc, char** argv) {
    Options options;
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "rk_edge_worker --model MODEL.rknn --video INPUT.mp4 "
                         "[--state-dir state] [--roi x,y,w,h] [--queue-capacity N] "
                         "[--queue-policy latest|fifo] [--loop-video] [--confidence X] "
                         "[--dwell-ms N] [--roi-debounce-frames N] [--dump-first-rknn-output DIR] "
                         "[--model-family yolov8-coco|yolox-ppe] [--danger-zone x,y,w,h] "
                         "[--conflict-zone x,y,w,h] [--ppe-confirm-frames N] "
                         "[--ppe-cooldown-ms N] [--conflict-dwell-ms N]\n";
            std::exit(0);
        }
        if (key == "--loop-video") {
            options.loop_video = true;
            continue;
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::invalid_argument("invalid argument near " + key);
        }
        args[key] = argv[++i];
    }
    if (!args.count("--model") || !args.count("--video")) {
        throw std::invalid_argument("--model and --video are required");
    }
    options.model = args.at("--model");
    options.video = args.at("--video");
    if (args.count("--state-dir")) options.state_dir = args.at("--state-dir");
    if (args.count("--queue-capacity")) options.queue_capacity = std::stoul(args.at("--queue-capacity"));
    if (args.count("--queue-policy")) {
        const auto& policy = args.at("--queue-policy");
        if (policy == "latest") options.queue_policy = edge::QueuePolicy::Latest;
        else if (policy == "fifo") options.queue_policy = edge::QueuePolicy::Fifo;
        else throw std::invalid_argument("--queue-policy must be latest or fifo");
    }
    if (args.count("--confidence")) options.confidence = std::stof(args.at("--confidence"));
    if (args.count("--dwell-ms")) options.dwell_threshold = std::chrono::milliseconds(std::stoll(args.at("--dwell-ms")));
    if (args.count("--roi-debounce-frames")) options.roi_debounce_frames = std::stoul(args.at("--roi-debounce-frames"));
    if (args.count("--dump-first-rknn-output")) options.raw_dump_dir = args.at("--dump-first-rknn-output");
    if (args.count("--model-family")) options.model_family = args.at("--model-family");
    if (args.count("--danger-zone")) options.danger_zone = parse_zone(args.at("--danger-zone"), "--danger-zone");
    if (args.count("--conflict-zone")) options.conflict_zone = parse_zone(args.at("--conflict-zone"), "--conflict-zone");
    if (args.count("--ppe-confirm-frames")) options.ppe_confirmation_frames = std::stoul(args.at("--ppe-confirm-frames"));
    if (args.count("--ppe-cooldown-ms")) options.ppe_cooldown = std::chrono::milliseconds(std::stoll(args.at("--ppe-cooldown-ms")));
    if (args.count("--conflict-dwell-ms")) options.conflict_dwell = std::chrono::milliseconds(std::stoll(args.at("--conflict-dwell-ms")));
    if (args.count("--roi")) {
        std::stringstream input(args.at("--roi"));
        char comma1 = 0, comma2 = 0, comma3 = 0;
        if (!(input >> options.roi.x >> comma1 >> options.roi.y >> comma2 >>
              options.roi.width >> comma3 >> options.roi.height) ||
            comma1 != ',' || comma2 != ',' || comma3 != ',') {
            throw std::invalid_argument("--roi must be x,y,width,height in normalized coordinates");
        }
    }
    if (options.queue_capacity == 0 || options.confidence < 0.0F || options.confidence > 1.0F ||
        options.dwell_threshold.count() <= 0 || options.roi_debounce_frames == 0 ||
        options.roi_debounce_frames > 10000 ||
        options.roi.x < 0.0 || options.roi.y < 0.0 || options.roi.width <= 0.0 ||
        options.roi.height <= 0.0 || options.roi.x + options.roi.width > 1.0 ||
        options.roi.y + options.roi.height > 1.0) {
        throw std::invalid_argument("invalid queue, confidence, ROI, dwell, or debounce configuration");
    }
    if (options.model_family != "yolov8-coco" && options.model_family != "yolox-ppe") {
        throw std::invalid_argument("--model-family must be yolov8-coco or yolox-ppe");
    }
    if (options.ppe_confirmation_frames == 0 || options.ppe_confirmation_frames > 10000 ||
        options.ppe_cooldown.count() < 0 || options.conflict_dwell.count() < 0) {
        throw std::invalid_argument("PPE confirmation, cooldown, and conflict dwell settings are invalid");
    }
    return options;
}

void emit_metric(std::uint64_t processed, std::uint64_t detections, std::uint64_t decoded,
                 const edge::BoundedQueue<Frame>& queue, Clock::time_point started,
                 double preprocess_ms, double inference_ms, double postprocess_ms,
                 double queue_wait_ms, double e2e_ms, const std::deque<double>& latency_window,
                 std::size_t roi_occupancy_count, std::uint64_t cumulative_roi_entries,
                 std::uint64_t safety_event_count, std::uint64_t ppe_violation_count,
                 std::uint64_t danger_zone_count, std::uint64_t conflict_count,
                 const edge::PPEStatusCounts& ppe_status_counts,
                 std::uint64_t reconnect_attempts, std::uint64_t stream_epoch,
                 const char* source_type, const char* source_status, const char* status) {
    std::lock_guard<std::mutex> output_lock(stdout_mutex);
    std::vector<double> sorted(latency_window.begin(), latency_window.end());
    std::sort(sorted.begin(), sorted.end());
    auto percentile = [&sorted](double fraction) {
        if (sorted.empty()) return 0.0;
        const auto index = static_cast<std::size_t>(std::ceil(fraction * sorted.size())) - 1;
        return sorted[std::min(index, sorted.size() - 1)];
    };
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const double fps = elapsed > 0.0 ? processed / elapsed : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "{\"type\":\"metric\",\"status\":\"" << status
              << "\",\"processed_frames\":" << processed
              << ",\"decoded_frames\":" << decoded
              << ",\"detections\":" << detections
              << ",\"roi_occupancy_count\":" << roi_occupancy_count
              << ",\"cumulative_roi_entries\":" << cumulative_roi_entries
              << ",\"safety_event_count\":" << safety_event_count
              << ",\"ppe_violation_count\":" << ppe_violation_count
              << ",\"danger_zone_intrusion_count\":" << danger_zone_count
              << ",\"forklift_person_cooccurrence_count\":" << conflict_count
              << ",\"ppe_compliant_person_count\":" << ppe_status_counts.compliant_people
              << ",\"ppe_violation_person_count\":" << ppe_status_counts.violation_people
              << ",\"ppe_unknown_person_count\":" << ppe_status_counts.unknown_people
              << ",\"queue_capacity\":" << queue.capacity()
              << ",\"queue_policy\":\"" << edge::queue_policy_name(queue.policy()) << "\""
              << ",\"queue_depth\":" << queue.size()
              << ",\"dropped_frames\":" << queue.dropped_count()
              << ",\"throughput_fps\":" << fps
              << ",\"last_preprocess_ms\":" << preprocess_ms
              << ",\"last_inference_ms\":" << inference_ms
              << ",\"last_postprocess_ms\":" << postprocess_ms
              << ",\"last_queue_wait_ms\":" << queue_wait_ms
              << ",\"last_end_to_end_ms\":" << e2e_ms
              << ",\"p50_end_to_end_ms\":" << percentile(0.50)
              << ",\"p95_end_to_end_ms\":" << percentile(0.95)
              << ",\"latency_window_frames\":" << sorted.size()
              << ",\"process_rss_kb\":" << read_rss_kb()
              << ",\"source_type\":\"" << source_type << "\""
              << ",\"source_status\":\"" << source_status << "\""
              << ",\"reconnect_attempts\":" << reconnect_attempts
              << ",\"stream_epoch\":" << stream_epoch << "}\n" << std::flush;
}

void emit_source_status(const char* source_status, std::uint64_t reconnect_attempts,
                        std::uint64_t stream_epoch) {
    std::lock_guard<std::mutex> output_lock(stdout_mutex);
    std::cout << "{\"type\":\"source_status\",\"source_type\":\"rtsp\",\"source_status\":\""
              << source_status << "\",\"reconnect_attempts\":" << reconnect_attempts
              << ",\"stream_epoch\":" << stream_epoch << "}\n" << std::flush;
}

std::string event_json(const edge::RoiEventTransition& event, std::uint64_t frame_sequence,
                       double pts_ms, const std::string& evidence, const std::string& model,
                       const std::string& event_id, const std::string& timestamp,
                       const char* source_id) {
    const auto& d = event.detection;
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "{\"type\":\"event\",\"schema_version\":1,\"event_id\":\""
        << json_escape(event_id) << "\",\"event_type\":\"" << edge::roi_event_type_name(event.type)
        << "\",\"state\":\"" << edge::roi_event_state_name(event.type)
        << "\",\"transition_reason\":\"" << edge::roi_event_reason(event.type)
        << "\",\"timestamp_utc\":\""
        << json_escape(timestamp) << "\",\"source_id\":\"" << source_id
        << "\",\"frame_sequence\":"
        << frame_sequence << ",\"source_pts_ms\":" << pts_ms
        << ",\"track_id\":" << event.track_id << ",\"class_id\":" << d.class_id
        << ",\"label\":\"" << json_escape(d.label ? d.label : "unknown")
        << "\",\"confidence\":" << d.confidence << ",\"box\":["
        << d.left << ',' << d.top << ',' << d.right << ',' << d.bottom
        << "],\"roi\":\"configured\",\"roi_occupancy_count\":" << event.occupancy_count
        << ",\"cumulative_roi_entries\":" << event.cumulative_entries
        << ",\"model\":\"" << json_escape(model)
        << "\",\"evidence\":\"" << json_escape(evidence) << "\"}";
    return out.str();
}

std::string safety_event_json(const edge::PPEEventTransition& event, std::uint64_t frame_sequence,
                              double pts_ms, const std::string& evidence, const std::string& model,
                              const std::string& event_id, const std::string& timestamp,
                              const char* source_id) {
    const auto& d = event.detection;
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "{\"type\":\"event\",\"schema_version\":1,\"event_id\":\""
        << json_escape(event_id) << "\",\"event_type\":\"" << json_escape(event.event_type)
        << "\",\"state\":\"active\",\"transition_reason\":\""
        << json_escape(event.reason) << "\",\"severity\":\"" << json_escape(event.severity)
        << "\",\"timestamp_utc\":\"" << json_escape(timestamp)
        << "\",\"source_id\":\"" << source_id << "\",\"frame_sequence\":"
        << frame_sequence << ",\"source_pts_ms\":" << pts_ms
        << ",\"track_id\":" << event.track_id << ",\"related_track_id\":";
    if (event.related_track_id.has_value()) out << *event.related_track_id;
    else out << "null";
    out << ",\"person_association\":\"" << json_escape(event.person_association)
        << "\",\"ppe_status\":\"" << json_escape(event.ppe_status)
        << "\",\"confirmation_frames\":" << event.confirmation_frames
        << ",\"class_id\":" << d.class_id << ",\"label\":\""
        << json_escape(d.label ? d.label : "unknown") << "\",\"confidence\":" << d.confidence
        << ",\"box\":[" << d.left << ',' << d.top << ',' << d.right << ',' << d.bottom
        << "],\"model\":\"" << json_escape(model)
        << "\",\"evidence\":\"" << json_escape(evidence) << "\"}";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, request_shutdown);
        std::signal(SIGTERM, request_shutdown);
        const Options options = parse_args(argc, argv);
        const bool rtsp_source = is_rtsp_source(options.video);
        if (!fs::exists(options.model)) throw std::runtime_error("model file does not exist: " + options.model);
        if (!rtsp_source && !fs::exists(options.video)) {
            throw std::runtime_error("video file does not exist: " + options.video);
        }
        fs::create_directories(fs::path(options.state_dir) / "evidence");

        const ModelFamily model_family = options.model_family == "yolox-ppe"
            ? ModelFamily::YoloXPpe
            : ModelFamily::YoloV8Coco;
        if (model_family == ModelFamily::YoloV8Coco && init_post_process() != 0) {
            throw std::runtime_error("failed to load COCO labels from ./model/coco_80_labels_list.txt");
        }
        int exit_code = 0;
        {
            RknnYoloEngine engine(options.model, options.raw_dump_dir, model_family);
            edge::BoundedQueue<Frame> queue(options.queue_capacity, options.queue_policy);
            edge::PPEEventConfig ppe_event_config;
            ppe_event_config.danger_zone = options.danger_zone;
            ppe_event_config.conflict_zone = options.conflict_zone;
            ppe_event_config.confirmation_frames = options.ppe_confirmation_frames;
            ppe_event_config.cooldown = options.ppe_cooldown;
            ppe_event_config.conflict_dwell = options.conflict_dwell;
            std::atomic<std::uint64_t> decoded{0};
            std::atomic<std::uint64_t> reconnect_attempts{0};
            std::atomic<std::uint64_t> stream_epoch{0};
            std::string producer_error;
            const auto started = Clock::now();
            const auto run_started_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            std::thread producer([&] {
                try {
                    cv::VideoCapture capture;
                    auto open_source = [&]() {
                        if (rtsp_source) {
                            const auto parameters = rtsp_open_parameters();
                            return capture.open(options.video, cv::CAP_FFMPEG, parameters);
                        }
                        return capture.open(options.video);
                    };
                    if (rtsp_source) {
                        // OpenCV's FFmpeg backend reads this process-level option for RTSP transport.
                        ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
                    }
                    if (!rtsp_source && !open_source()) {
                        producer_error = "OpenCV could not open the video source";
                        queue.close();
                        return;
                    }
                    const double source_fps = rtsp_source ? 0.0 : capture.get(cv::CAP_PROP_FPS);
                    const bool pace_source = !rtsp_source && std::isfinite(source_fps) &&
                                             source_fps > 0.0 && source_fps <= 240.0;
                    const auto frame_period = pace_source
                        ? std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / source_fps))
                        : Clock::duration::zero();
                    auto next_frame_at = Clock::now();
                    if (rtsp_source) {
                        std::cerr << "video source_type=rtsp pacing=disabled reconnect=enabled\n";
                    } else {
                        std::cerr << "video source_type=local-file source_fps=" << source_fps
                                  << " pacing=" << (pace_source ? "media-rate" : "disabled")
                                  << " looping=" << (options.loop_video ? "enabled" : "disabled") << '\n';
                    }
                    cv::Mat frame;
                    std::uint64_t sequence = 0;
                    std::uint64_t local_stream_epoch = 0;
                    double last_pts_ms = -1.0;
                    bool ever_decoded = false;
                    bool awaiting_reconnect = false;
                    std::chrono::milliseconds retry_delay{500};
                    while (!shutdown_requested) {
                        if (rtsp_source && !capture.isOpened()) {
                            bool opened = false;
                            try {
                                opened = open_source();
                            } catch (const std::exception&) {
                                opened = false;
                            }
                            if (!opened) {
                                capture.release();
                                const auto attempt = reconnect_attempts.fetch_add(1) + 1;
                                emit_source_status(ever_decoded ? "reconnecting" : "connecting",
                                                   attempt, local_stream_epoch);
                                std::cerr << "RTSP reconnect attempt=" << attempt << '\n';
                                if (!wait_for_retry(retry_delay)) break;
                                retry_delay = std::min(retry_delay * 2, std::chrono::milliseconds(5000));
                                continue;
                            }
                        }

                        bool frame_read = false;
                        try {
                            frame_read = capture.read(frame) && !frame.empty();
                        } catch (const std::exception&) {
                            frame_read = false;
                        }
                        if (!frame_read) {
                            if (rtsp_source) {
                                capture.release();
                                awaiting_reconnect = ever_decoded;
                                const auto attempt = reconnect_attempts.fetch_add(1) + 1;
                                emit_source_status(ever_decoded ? "reconnecting" : "connecting",
                                                   attempt, local_stream_epoch);
                                std::cerr << "RTSP reconnect attempt=" << attempt << '\n';
                                if (!wait_for_retry(retry_delay)) break;
                                retry_delay = std::min(retry_delay * 2, std::chrono::milliseconds(5000));
                                continue;
                            }
                            if (!options.loop_video) break;
                            capture.release();
                            if (!capture.open(options.video)) {
                                producer_error = "OpenCV could not reopen the looped video source";
                                break;
                            }
                            if (!capture.read(frame) || frame.empty()) {
                                producer_error = "looped video source returned no frame after reopen";
                                break;
                            }
                        }
                        if (rtsp_source) {
                            if (awaiting_reconnect) {
                                ++local_stream_epoch;
                                stream_epoch.store(local_stream_epoch);
                                awaiting_reconnect = false;
                                retry_delay = std::chrono::milliseconds(500);
                                std::cerr << "RTSP stream reconnected epoch=" << local_stream_epoch << '\n';
                                emit_source_status("connected", reconnect_attempts.load(),
                                                   local_stream_epoch);
                            }
                            if (!ever_decoded) {
                                emit_source_status("connected", reconnect_attempts.load(),
                                                   local_stream_epoch);
                            }
                            ever_decoded = true;
                        }
                        ++sequence;
                        double pts_ms = rtsp_source
                            ? milliseconds(Clock::now() - started)
                            : capture.get(cv::CAP_PROP_POS_MSEC);
                        if (!rtsp_source && (!std::isfinite(pts_ms) || pts_ms < 0.0 ||
                            (sequence > 1 && pts_ms <= last_pts_ms))) {
                            pts_ms = pace_source
                                ? static_cast<double>(sequence - 1) * 1000.0 / source_fps
                                : milliseconds(Clock::now() - started);
                        }
                        if (!rtsp_source && sequence > 1 && pts_ms <= last_pts_ms) {
                            pts_ms = last_pts_ms + (pace_source ? 1000.0 / source_fps : 1.0);
                        }
                        Frame item;
                        item.image = frame.clone();
                        item.sequence = sequence;
                        item.pts_ms = pts_ms;
                        item.stream_epoch = local_stream_epoch;
                        last_pts_ms = pts_ms;
                        item.decoded_at = Clock::now();
                        if (queue.push(std::move(item)) == edge::QueuePushResult::Closed) break;
                        decoded.store(sequence);
                        if (pace_source) {
                            next_frame_at += frame_period;
                            const auto now = Clock::now();
                            if (next_frame_at > now) {
                                std::this_thread::sleep_until(next_frame_at);
                            } else {
                                next_frame_at = now;
                            }
                        }
                    }
                    queue.close();
                } catch (const std::exception& error) {
                    producer_error = rtsp_source ? "RTSP capture failed unexpectedly" : error.what();
                    queue.close();
                }
            });

            edge::IoUTracker tracker;
            edge::RoiEventMachine roi_event_machine(options.dwell_threshold,
                                                    options.roi_debounce_frames);
            edge::PPEEventMachine ppe_event_machine(ppe_event_config);
            edge::PPEStatusCounts ppe_status_counts;
            std::uint64_t processed_stream_epoch = 0;
            std::deque<double> latency_window;
            std::uint64_t processed = 0;
            std::uint64_t detection_total = 0;
            std::uint64_t event_total = 0;
            std::size_t roi_occupancy_count = 0;
            std::uint64_t cumulative_roi_entries = 0;
            double roi_timeline_ms = -1.0;
            double last_preprocess = 0.0, last_inference = 0.0, last_postprocess = 0.0;
            double last_queue_wait = 0.0, last_e2e = 0.0;
            Frame frame;
            while (queue.pop(frame)) {
                if (frame.stream_epoch != processed_stream_epoch) {
                    tracker.reset();
                    roi_event_machine.reset_stream();
                    ppe_event_machine.reset_stream();
                    roi_occupancy_count = 0;
                    processed_stream_epoch = frame.stream_epoch;
                    std::cerr << "stream state reset epoch=" << processed_stream_epoch << '\n';
                }
                const auto process_start = Clock::now();
                last_queue_wait = milliseconds(process_start - frame.decoded_at);
                try {
                    auto result = engine.infer(frame.image, options.confidence);
                    last_preprocess = result.preprocess_ms;
                    last_inference = result.inference_ms;
                    last_postprocess = result.postprocess_ms;
                    detection_total += result.detections.size();
                    const auto tracked = tracker.update(result.detections);
                    cv::Mat annotated = frame.image.clone();
                    std::vector<edge::RoiObservation> observations;
                    observations.reserve(tracked.size());

                    for (const auto& item : tracked) {
                        const auto& d = item.detection;
                        const bool inside = options.roi.contains(d, frame.image.cols, frame.image.rows);
                        observations.push_back({item.track_id, d, inside});
                        cv::rectangle(annotated, cv::Point(d.left, d.top), cv::Point(d.right, d.bottom),
                                      cv::Scalar(40, 220, 40), 2);
                        std::ostringstream label;
                        label << (d.label ? d.label : "object") << ' ' << std::fixed << std::setprecision(2)
                              << d.confidence << " T" << item.track_id;
                        cv::putText(annotated, label.str(), cv::Point(std::max(0, d.left), std::max(18, d.top - 4)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40, 220, 40), 1);
                    }
                    const auto active = tracker.active_ids();
                    double event_time_ms = frame.pts_ms;
                    if (!std::isfinite(event_time_ms) || event_time_ms < 0.0 ||
                        event_time_ms <= roi_timeline_ms) {
                        event_time_ms = milliseconds(frame.decoded_at - started);
                    }
                    event_time_ms = std::max(event_time_ms, roi_timeline_ms);
                    roi_timeline_ms = event_time_ms;
                    const auto event_time = Clock::time_point{} +
                        std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double, std::milli>(event_time_ms));
                    auto transitions = roi_event_machine.update(observations, active, event_time);
                    std::vector<edge::PPEEventTransition> safety_transitions;
                    if (model_family == ModelFamily::YoloXPpe) {
                        safety_transitions = ppe_event_machine.update(
                            tracked, active, frame.image.cols, frame.image.rows, event_time);
                    }
                    ppe_status_counts = ppe_event_machine.person_status_counts();
                    roi_occupancy_count = roi_event_machine.occupancy_count();
                    cumulative_roi_entries = roi_event_machine.cumulative_entries();
                    std::vector<std::pair<edge::RoiEventTransition, std::string>> pending_events;
                    pending_events.reserve(transitions.size());
                    for (const auto& event : transitions) {
                        const auto now = std::chrono::system_clock::now();
                        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();
                        const std::string event_id = "evt-" + std::to_string(now_ms) + "-" +
                                                     std::to_string(frame.sequence) + "-" +
                                                     std::to_string(event.track_id) + "-" +
                                                     edge::roi_event_type_name(event.type);
                        pending_events.emplace_back(event, event_id);
                    }
                    std::vector<std::pair<edge::PPEEventTransition, std::string>> pending_safety_events;
                    pending_safety_events.reserve(safety_transitions.size());
                    for (const auto& event : safety_transitions) {
                        const auto now = std::chrono::system_clock::now();
                        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();
                        const std::string related_id = event.related_track_id.has_value()
                            ? std::to_string(*event.related_track_id)
                            : "unknown";
                        const std::string event_id = "evt-" + std::to_string(now_ms) + "-" +
                                                     std::to_string(frame.sequence) + "-" +
                                                     std::to_string(event.track_id) + "-" +
                                                     related_id + "-" + event.event_type;
                        pending_safety_events.emplace_back(event, event_id);
                    }

                    if (!pending_events.empty() || !pending_safety_events.empty()) {
                        const std::string filename = "frame-" + std::to_string(run_started_ms) + "-" +
                                                     std::to_string(frame.sequence) + ".jpg";
                        const fs::path disk_path = fs::path(options.state_dir) / "evidence" / filename;
                        if (!cv::imwrite(disk_path.string(), annotated)) {
                            throw std::runtime_error("failed to write event evidence image");
                        }
                        const std::string evidence = (fs::path("evidence") / filename).generic_string();
                        for (const auto& pending : pending_events) {
                            const std::string event_id = pending.second;
                            const auto timestamp = utc_timestamp(std::chrono::system_clock::now());
                            const auto payload = event_json(
                                pending.first, frame.sequence, frame.pts_ms, evidence,
                                basename(options.model), event_id, timestamp,
                                rtsp_source ? "rtsp" : "local-file");
                            {
                                std::lock_guard<std::mutex> output_lock(stdout_mutex);
                                std::cout << payload << '\n' << std::flush;
                            }
                            ++event_total;
                        }
                        for (const auto& pending : pending_safety_events) {
                            const auto timestamp = utc_timestamp(std::chrono::system_clock::now());
                            const auto payload = safety_event_json(
                                pending.first, frame.sequence, frame.pts_ms, evidence,
                                basename(options.model), pending.second, timestamp,
                                rtsp_source ? "rtsp" : "local-file");
                            {
                                std::lock_guard<std::mutex> output_lock(stdout_mutex);
                                std::cout << payload << '\n' << std::flush;
                            }
                            ++event_total;
                        }
                    }

                    ++processed;
                    last_e2e = milliseconds(Clock::now() - frame.decoded_at);
                    latency_window.push_back(last_e2e);
                    if (latency_window.size() > 300) latency_window.pop_front();
                    if (processed == 1 || processed % 30 == 0) {
                        emit_metric(processed, detection_total, decoded.load(), queue, started,
                                    last_preprocess, last_inference, last_postprocess,
                                    last_queue_wait, last_e2e, latency_window,
                                    roi_occupancy_count, cumulative_roi_entries,
                                    ppe_event_machine.total_events(),
                                    ppe_event_machine.ppe_violations(),
                                    ppe_event_machine.danger_zone_intrusions(),
                                    ppe_event_machine.forklift_person_cooccurrences(),
                                    ppe_status_counts,
                                    reconnect_attempts.load(), stream_epoch.load(),
                                    rtsp_source ? "rtsp" : "local-file",
                                    rtsp_source ? "connected" : "not-applicable", "running");
                    }
                } catch (const std::exception& error) {
                    std::cerr << "frame processing failed at sequence " << frame.sequence
                              << ": " << error.what() << '\n';
                    exit_code = 4;
                    queue.close();
                    break;
                }
            }
            producer.join();
            if (decoded.load() == 0 && !shutdown_requested) {
                std::cerr << "video input failed: "
                          << (producer_error.empty() ? "no frames decoded" : producer_error) << '\n';
                exit_code = 2;
            } else if (!producer_error.empty() && !shutdown_requested) {
                std::cerr << "video producer stopped after input error: " << producer_error << '\n';
                exit_code = 3;
            }
            emit_metric(processed, detection_total, decoded.load(), queue, started,
                        last_preprocess, last_inference, last_postprocess,
                        last_queue_wait, last_e2e, latency_window,
                        roi_occupancy_count, cumulative_roi_entries,
                        ppe_event_machine.total_events(),
                        ppe_event_machine.ppe_violations(),
                        ppe_event_machine.danger_zone_intrusions(),
                        ppe_event_machine.forklift_person_cooccurrences(),
                        ppe_status_counts,
                        reconnect_attempts.load(), stream_epoch.load(),
                        rtsp_source ? "rtsp" : "local-file",
                        rtsp_source ? "connected" : "not-applicable",
                        exit_code == 0 ? "completed" : "error");
            std::cerr << "worker completed: decoded=" << decoded.load()
                      << " processed=" << processed << " detections=" << detection_total
                      << " events=" << event_total << " dropped=" << queue.dropped_count() << '\n';
        }
        deinit_post_process();
        return exit_code;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        deinit_post_process();
        return 1;
    }
}
