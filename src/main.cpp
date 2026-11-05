#include "bounded_queue.hpp"
#include "roi_events.hpp"
#include "tracker.hpp"
#include "yolov8.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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

struct Frame {
    cv::Mat image;
    std::uint64_t sequence = 0;
    double pts_ms = 0.0;
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
    explicit RknnYoloEngine(const std::string& model_path) {
        try {
            auto model = read_file(model_path);
            int ret = rknn_init(&context_.rknn_ctx, model.data(),
                                static_cast<std::uint32_t>(model.size()), 0, nullptr);
            if (ret < 0) {
                throw std::runtime_error("rknn_init failed with code " + std::to_string(ret));
            }

            ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM,
                             &context_.io_num, sizeof(context_.io_num));
            if (ret != RKNN_SUCC || context_.io_num.n_input != 1 || context_.io_num.n_output == 0) {
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
            model_name_ = basename(model_path);
            std::cerr << "RKNN model=" << model_name_
                      << " input=" << context_.model_width << 'x' << context_.model_height
                      << " channels=" << context_.model_channel
                      << " outputs=" << context_.io_num.n_output
                      << " quantized=" << (context_.is_quant ? "true" : "false") << '\n';
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
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

        letterbox_t letterbox{};
        letterbox.x_pad = pad_x;
        letterbox.y_pad = pad_y;
        letterbox.scale = scale;
        result.preprocess_ms = milliseconds(Clock::now() - prep_start);

        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<std::uint32_t>(rgb.total() * rgb.elemSize());
        input.buf = rgb.data;
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

        const auto post_start = Clock::now();
        object_detect_result_list raw{};
        ret = post_process(&context_, outputs.data(), &letterbox, confidence, NMS_THRESH, &raw);
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
            result.detections.push_back({item.cls_id, item.prop, left, top, right, bottom,
                                         coco_cls_to_name(item.cls_id)});
        }
        return result;
    }

private:
    rknn_app_context_t context_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::string model_name_;
};

struct Options {
    std::string model;
    std::string video;
    std::string state_dir = "state";
    Roi roi;
    std::size_t queue_capacity = 1;
    float confidence = 0.25F;
    std::chrono::milliseconds dwell_threshold{5000};
    std::size_t roi_debounce_frames = 2;
};

Options parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "rk_edge_worker --model MODEL.rknn --video INPUT.mp4 "
                         "[--state-dir state] [--roi x,y,w,h] [--queue-capacity N] [--confidence X] "
                         "[--dwell-ms N] [--roi-debounce-frames N]\n";
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::invalid_argument("invalid argument near " + key);
        }
        args[key] = argv[++i];
    }
    Options options;
    if (!args.count("--model") || !args.count("--video")) {
        throw std::invalid_argument("--model and --video are required");
    }
    options.model = args.at("--model");
    options.video = args.at("--video");
    if (args.count("--state-dir")) options.state_dir = args.at("--state-dir");
    if (args.count("--queue-capacity")) options.queue_capacity = std::stoul(args.at("--queue-capacity"));
    if (args.count("--confidence")) options.confidence = std::stof(args.at("--confidence"));
    if (args.count("--dwell-ms")) options.dwell_threshold = std::chrono::milliseconds(std::stoll(args.at("--dwell-ms")));
    if (args.count("--roi-debounce-frames")) options.roi_debounce_frames = std::stoul(args.at("--roi-debounce-frames"));
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
    return options;
}

void emit_metric(std::uint64_t processed, std::uint64_t detections, std::uint64_t decoded,
                 const edge::BoundedLatestQueue<Frame>& queue, Clock::time_point started,
                 double preprocess_ms, double inference_ms, double postprocess_ms,
                 double queue_wait_ms, double e2e_ms, const std::deque<double>& latency_window,
                 std::size_t roi_occupancy_count, std::uint64_t cumulative_roi_entries,
                 const char* status) {
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
              << ",\"queue_capacity\":" << queue.capacity()
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
              << ",\"process_rss_kb\":" << read_rss_kb() << "}\n" << std::flush;
}

std::string event_json(const edge::RoiEventTransition& event, std::uint64_t frame_sequence,
                       double pts_ms, const std::string& evidence, const std::string& model,
                       const std::string& event_id, const std::string& timestamp) {
    const auto& d = event.detection;
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "{\"type\":\"event\",\"schema_version\":1,\"event_id\":\""
        << json_escape(event_id) << "\",\"event_type\":\"" << edge::roi_event_type_name(event.type)
        << "\",\"state\":\"" << edge::roi_event_state_name(event.type)
        << "\",\"transition_reason\":\"" << edge::roi_event_reason(event.type)
        << "\",\"timestamp_utc\":\""
        << json_escape(timestamp) << "\",\"source_id\":\"local-file\",\"frame_sequence\":"
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (!fs::exists(options.model)) throw std::runtime_error("model file does not exist: " + options.model);
        if (!fs::exists(options.video)) throw std::runtime_error("video file does not exist: " + options.video);
        fs::create_directories(fs::path(options.state_dir) / "evidence");

        if (init_post_process() != 0) throw std::runtime_error("failed to load COCO labels from ./model/coco_80_labels_list.txt");
        int exit_code = 0;
        {
            RknnYoloEngine engine(options.model);
            edge::BoundedLatestQueue<Frame> queue(options.queue_capacity);
            std::atomic<std::uint64_t> decoded{0};
            std::string producer_error;
            const auto started = Clock::now();
            const auto run_started_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            std::thread producer([&] {
                try {
                    cv::VideoCapture capture(options.video);
                    if (!capture.isOpened()) {
                        producer_error = "OpenCV could not open the video source";
                        queue.close();
                        return;
                    }
                    const double source_fps = capture.get(cv::CAP_PROP_FPS);
                    const bool pace_source = std::isfinite(source_fps) && source_fps > 0.0 && source_fps <= 240.0;
                    const auto frame_period = pace_source
                        ? std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / source_fps))
                        : Clock::duration::zero();
                    auto next_frame_at = Clock::now();
                    std::cerr << "video source_fps=" << source_fps
                              << " pacing=" << (pace_source ? "media-rate" : "disabled") << '\n';
                    cv::Mat frame;
                    std::uint64_t sequence = 0;
                    double last_pts_ms = -1.0;
                    while (capture.read(frame)) {
                        if (frame.empty()) continue;
                        ++sequence;
                        double pts_ms = capture.get(cv::CAP_PROP_POS_MSEC);
                        if (!std::isfinite(pts_ms) || pts_ms < 0.0 ||
                            (sequence > 1 && pts_ms <= last_pts_ms)) {
                            pts_ms = pace_source
                                ? static_cast<double>(sequence - 1) * 1000.0 / source_fps
                                : milliseconds(Clock::now() - started);
                        }
                        if (sequence > 1 && pts_ms <= last_pts_ms) {
                            pts_ms = last_pts_ms + (pace_source ? 1000.0 / source_fps : 1.0);
                        }
                        Frame item;
                        item.image = frame.clone();
                        item.sequence = sequence;
                        item.pts_ms = pts_ms;
                        last_pts_ms = pts_ms;
                        item.decoded_at = Clock::now();
                        if (!queue.push(std::move(item))) break;
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
                    producer_error = error.what();
                    queue.close();
                }
            });

            edge::IoUTracker tracker;
            edge::RoiEventMachine roi_event_machine(options.dwell_threshold,
                                                    options.roi_debounce_frames);
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

                    if (!pending_events.empty()) {
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
                            std::cout << event_json(pending.first, frame.sequence, frame.pts_ms,
                                                    evidence, basename(options.model), event_id, timestamp)
                                      << '\n' << std::flush;
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
                                    roi_occupancy_count, cumulative_roi_entries, "running");
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
            if (decoded.load() == 0) {
                std::cerr << "video input failed: "
                          << (producer_error.empty() ? "no frames decoded" : producer_error) << '\n';
                exit_code = 2;
            } else if (!producer_error.empty()) {
                std::cerr << "video producer stopped after input error: " << producer_error << '\n';
            }
            emit_metric(processed, detection_total, decoded.load(), queue, started,
                        last_preprocess, last_inference, last_postprocess,
                        last_queue_wait, last_e2e, latency_window,
                        roi_occupancy_count, cumulative_roi_entries,
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
