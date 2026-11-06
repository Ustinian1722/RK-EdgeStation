#include "bounded_queue.hpp"
#include "ppe_events.hpp"
#include "preprocess.hpp"
#include "roi_events.hpp"
#include "tracker.hpp"
#include "video_source.hpp"
#include "yolov8.h"
#include "yolox_ppe_postprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
#include <unistd.h>

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

struct Frame {
    edge::DecodedFrame media;
    std::uint64_t sequence = 0;
    double pts_ms = 0.0;
    std::uint64_t stream_epoch = 0;
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
    double input_set_ms = 0.0;
    double input_sync_ms = 0.0;
    double rknn_run_ms = 0.0;
    double output_fetch_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    std::uint64_t model_input_bytes = 0;
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

int read_peak_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmHWM:") {
            int value = 0;
            status >> value;
            return value;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return read_rss_kb();
}

struct ProcessCpuUsage {
    double core_percent = 0.0;
    double system_percent = 0.0;
    double sample_seconds = 0.0;
};

ProcessCpuUsage read_process_cpu_usage() {
    std::ifstream stat("/proc/self/stat");
    std::string line;
    if (!std::getline(stat, line)) return {};
    const auto command_end = line.rfind(')');
    if (command_end == std::string::npos || command_end + 2 >= line.size()) return {};
    std::istringstream fields_stream(line.substr(command_end + 2));
    std::vector<std::string> fields;
    for (std::string field; fields_stream >> field;) fields.push_back(std::move(field));
    if (fields.size() <= 12) return {};
    std::uint64_t ticks = 0;
    try {
        ticks = std::stoull(fields[11]) + std::stoull(fields[12]);
    } catch (const std::exception&) {
        return {};
    }

    static bool have_start = false;
    static std::uint64_t start_ticks = 0;
    static auto start_at = Clock::now();
    const auto now = Clock::now();
    const long ticks_per_second = ::sysconf(_SC_CLK_TCK);
    const long online_cpus = ::sysconf(_SC_NPROCESSORS_ONLN);
    if (!have_start || ticks_per_second <= 0) {
        have_start = true;
        start_ticks = ticks;
        start_at = now;
        return {};
    }
    const double elapsed = std::chrono::duration<double>(now - start_at).count();
    const std::uint64_t elapsed_ticks = ticks >= start_ticks ? ticks - start_ticks : 0;
    if (elapsed <= 0.0) return {};
    ProcessCpuUsage usage;
    usage.sample_seconds = elapsed;
    usage.core_percent = static_cast<double>(elapsed_ticks) /
                         static_cast<double>(ticks_per_second) / elapsed * 100.0;
    usage.system_percent = online_cpus > 0
        ? usage.core_percent / static_cast<double>(online_cpus)
        : usage.core_percent;
    return usage;
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
                            ModelFamily model_family, bool shared_input)
        : raw_dump_dir_(std::move(raw_dump_dir)), model_family_(model_family),
          shared_input_(shared_input) {
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
            if (shared_input_) {
                native_input_attr_ = {};
                native_input_attr_.index = 0;
                ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR,
                                 &native_input_attr_, sizeof(native_input_attr_));
                if (ret != RKNN_SUCC) {
                    throw std::runtime_error("RKNN_QUERY_NATIVE_INPUT_ATTR failed with code " +
                                             std::to_string(ret));
                }
                if (native_input_attr_.fmt != RKNN_TENSOR_NHWC ||
                    native_input_attr_.dims[0] != 1 || native_input_attr_.dims[3] != 3 ||
                    native_input_attr_.dims[1] != input_attrs_.front().dims[1] ||
                    native_input_attr_.dims[2] != input_attrs_.front().dims[2] ||
                    native_input_attr_.size_with_stride == 0) {
                    throw std::runtime_error("shared input mode requires a native NHWC image input");
                }
                // The deployed preprocessing contract supplies raw uint8 RGB/BGR.
                // This matches Rockchip's zero-copy image-input example; it does
                // not claim the Runtime avoids all internal conversions.
                native_input_attr_.type = RKNN_TENSOR_UINT8;
                input_memory_ = rknn_create_mem(context_.rknn_ctx,
                                                native_input_attr_.size_with_stride);
                if (input_memory_ == nullptr || input_memory_->virt_addr == nullptr ||
                    input_memory_->fd < 0 ||
                    input_memory_->size < static_cast<std::uint64_t>(native_input_attr_.w_stride == 0
                        ? context_.model_width : native_input_attr_.w_stride) *
                        context_.model_height * 3) {
                    throw std::runtime_error("rknn_create_mem returned an invalid shared image buffer");
                }
                ret = rknn_set_io_mem(context_.rknn_ctx, input_memory_, &native_input_attr_);
                if (ret != RKNN_SUCC) {
                    throw std::runtime_error("rknn_set_io_mem failed with code " + std::to_string(ret));
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
            model_name_ = basename(model_path);
            std::cerr << "RKNN model=" << model_name_
                      << " input=" << context_.model_width << 'x' << context_.model_height
                      << " channels=" << context_.model_channel
                      << " input_type=" << get_type_string(input.type)
                      << " input_format=" << get_format_string(input.fmt)
                      << " input_quantization=" << get_qnt_type_string(input.qnt_type)
                      << " input_scale=" << input.scale << " input_zero_point=" << input.zp
                      << " outputs=" << context_.io_num.n_output
                      << " quantized=" << (context_.is_quant ? "true" : "false")
                      << " family=" << (model_family_ == ModelFamily::YoloXPpe ? "yolox-ppe" : "yolov8-coco")
                      << " input_mode=" << (shared_input_ ? "shared-mem" : "inputs-set")
                      << (shared_input_ ? " input_mem_fd=" + std::to_string(input_memory_->fd) +
                                             " input_mem_size=" + std::to_string(input_memory_->size)
                                       : "")
                      << '\n';
        } catch (...) {
            if (input_memory_ != nullptr && context_.rknn_ctx != 0) {
                rknn_destroy_mem(context_.rknn_ctx, input_memory_);
                input_memory_ = nullptr;
            }
            if (context_.rknn_ctx != 0) {
                rknn_destroy(context_.rknn_ctx);
                context_.rknn_ctx = 0;
            }
            throw;
        }
    }

    ~RknnYoloEngine() {
        if (input_memory_ != nullptr && context_.rknn_ctx != 0) {
            const int ret = rknn_destroy_mem(context_.rknn_ctx, input_memory_);
            if (ret != RKNN_SUCC) std::cerr << "rknn_destroy_mem returned " << ret << '\n';
            input_memory_ = nullptr;
        }
        if (context_.rknn_ctx != 0) {
            const int ret = rknn_destroy(context_.rknn_ctx);
            if (ret != RKNN_SUCC) {
                std::cerr << "rknn_destroy returned " << ret << '\n';
            }
        }
    }

    RknnYoloEngine(const RknnYoloEngine&) = delete;
    RknnYoloEngine& operator=(const RknnYoloEngine&) = delete;

    InferenceResult infer(const edge::DecodedFrame& frame, float confidence,
                          edge::PreprocessBackend preprocess_backend) {
        InferenceResult result;
        const bool output_rgb = model_family_ == ModelFamily::YoloV8Coco;
        edge::PreprocessResult preprocessed;
        if (shared_input_) {
            const std::uint32_t w_stride = native_input_attr_.w_stride == 0
                ? static_cast<std::uint32_t>(context_.model_width) : native_input_attr_.w_stride;
            cv::Mat destination(context_.model_height, context_.model_width, CV_8UC3,
                                input_memory_->virt_addr,
                                static_cast<std::size_t>(w_stride) * 3);
            preprocessed = edge::preprocess_frame_into(
                frame, context_.model_width, context_.model_height, output_rgb,
                preprocess_backend, std::move(destination),
                frame.dma_buf_fd >= 0 ? input_memory_->fd : -1, input_memory_->size);
            const auto sync_start = Clock::now();
            const rknn_mem_sync_mode sync_mode = frame.dma_buf_fd >= 0
                ? RKNN_MEMORY_SYNC_BIDIRECTIONAL : RKNN_MEMORY_SYNC_TO_DEVICE;
            const int sync_ret = rknn_mem_sync(context_.rknn_ctx, input_memory_, sync_mode);
            if (sync_ret != RKNN_SUCC) {
                throw std::runtime_error("rknn_mem_sync failed with code " + std::to_string(sync_ret));
            }
            result.input_sync_ms = milliseconds(Clock::now() - sync_start);
        } else {
            preprocessed = edge::preprocess_frame(
                frame, context_.model_width, context_.model_height, output_rgb, preprocess_backend);
        }
        result.preprocess_ms = preprocessed.elapsed_ms;
        result.model_input_bytes = preprocessed.model_input_bytes;
        const cv::Mat& model_input = preprocessed.model_input;

        int ret = RKNN_SUCC;
        if (!shared_input_) {
            rknn_input input{};
            input.index = 0;
            input.type = RKNN_TENSOR_UINT8;
            input.fmt = RKNN_TENSOR_NHWC;
            input.size = static_cast<std::uint32_t>(model_input.total() * model_input.elemSize());
            input.buf = model_input.data;
            const auto input_set_start = Clock::now();
            ret = rknn_inputs_set(context_.rknn_ctx, context_.io_num.n_input, &input);
            if (ret != RKNN_SUCC) {
                throw std::runtime_error("rknn_inputs_set failed with code " + std::to_string(ret));
            }
            result.input_set_ms = milliseconds(Clock::now() - input_set_start);
        }

        const auto inference_start = Clock::now();
        ret = rknn_run(context_.rknn_ctx, nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_run failed with code " + std::to_string(ret));
        }
        result.rknn_run_ms = milliseconds(Clock::now() - inference_start);
        const auto output_fetch_start = Clock::now();
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
        result.output_fetch_ms = milliseconds(Clock::now() - output_fetch_start);
        struct OutputRelease {
            rknn_context context;
            std::uint32_t count;
            rknn_output* outputs;
            ~OutputRelease() { rknn_outputs_release(context, count, outputs); }
        } output_release{context_.rknn_ctx, context_.io_num.n_output, outputs.data()};
        result.inference_ms = result.rknn_run_ms + result.output_fetch_ms;
        dump_first_outputs(model_input, outputs, frame.width, frame.height, frame.sequence,
                           frame.pts_ms, preprocessed.letterbox, output_rgb ? "RGB" : "BGR");

        const auto post_start = Clock::now();
        object_detect_result_list raw{};
        if (model_family_ == ModelFamily::YoloXPpe) {
            ret = post_process_yolox_ppe(&context_, outputs.data(), &preprocessed.letterbox,
                                         confidence, NMS_THRESH, frame.width, frame.height, &raw);
        } else {
            ret = post_process(&context_, outputs.data(), &preprocessed.letterbox,
                               confidence, NMS_THRESH, &raw);
        }
        if (ret != 0) {
            throw std::runtime_error("YOLO postprocess failed with code " + std::to_string(ret));
        }
        result.postprocess_ms = milliseconds(Clock::now() - post_start);

        result.detections.reserve(raw.count);
        for (int i = 0; i < raw.count; ++i) {
            const auto& item = raw.results[i];
            const int left = std::clamp(item.box.left, 0, frame.width);
            const int top = std::clamp(item.box.top, 0, frame.height);
            const int right = std::clamp(item.box.right, left, frame.width);
            const int bottom = std::clamp(item.box.bottom, top, frame.height);
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
                            int source_width, int source_height, std::uint64_t source_sequence,
                            double source_pts_ms, const letterbox_t& letterbox,
                            const char* input_channel_order) {
        if (raw_dump_dir_.empty() || raw_dumped_) return;
        fs::create_directories(raw_dump_dir_);

        const fs::path root(raw_dump_dir_);
        const std::string input_filename = std::string("input-") +
            (model_family_ == ModelFamily::YoloXPpe ? "bgr" : "rgb") + "-u8.bin";
        const fs::path input_path = root / input_filename;
        std::ofstream input_file(input_path, std::ios::binary);
        const auto input_bytes = model_input.total() * model_input.elemSize();
        const auto row_bytes = static_cast<std::streamsize>(
            static_cast<std::size_t>(model_input.cols) * model_input.elemSize());
        for (int row = 0; row < model_input.rows; ++row) {
            input_file.write(reinterpret_cast<const char*>(model_input.ptr(row)), row_bytes);
        }
        if (!input_file) throw std::runtime_error("failed to write raw RKNN input tensor");

        std::ofstream manifest(root / "tensor-manifest.txt");
        if (!manifest) throw std::runtime_error("failed to create raw RKNN tensor manifest");
        manifest << "model=" << model_name_ << '\n'
                 << "input_layout=NHWC\ninput_type=uint8\ninput_shape="
                 << context_.model_height << ',' << context_.model_width << ','
                 << context_.model_channel << '\n'
                 << "input_model_type=" << get_type_string(input_attrs_.front().type) << '\n'
                 << "input_model_format=" << get_format_string(input_attrs_.front().fmt) << '\n'
                 << "input_quantization=" << get_qnt_type_string(input_attrs_.front().qnt_type) << '\n'
                 << "input_scale=" << input_attrs_.front().scale << '\n'
                 << "input_zero_point=" << input_attrs_.front().zp << '\n'
                 << "input_channel_order=" << input_channel_order << '\n'
                 << "input_file=" << input_path.filename().string() << '\n'
                 << "input_bytes=" << input_bytes << '\n'
                 << "source_width=" << source_width << '\n'
                 << "source_height=" << source_height << '\n'
                 << "source_sequence=" << source_sequence << '\n'
                 << "source_pts_ms=" << std::setprecision(12) << source_pts_ms << '\n'
                 << "letterbox_x_pad=" << letterbox.x_pad << '\n'
                 << "letterbox_y_pad=" << letterbox.y_pad << '\n'
                 << "letterbox_scale=" << std::setprecision(10) << letterbox.scale << '\n'
                 << "output_requested_type=" << (context_.is_quant ? "native" : "float32") << '\n';

        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].buf == nullptr || outputs[i].size == 0) {
                throw std::runtime_error("raw RKNN output is empty");
            }
            std::ostringstream filename;
            filename << "output-" << std::setw(2) << std::setfill('0') << i
                     << (context_.is_quant ? ".native.bin" : ".f32.bin");
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
                     << "output[" << i << "].qnt_type=" << get_qnt_type_string(attr.qnt_type) << '\n'
                     << "output[" << i << "].scale=" << attr.scale << '\n'
                     << "output[" << i << "].zero_point=" << attr.zp << '\n'
                     << "output[" << i << "].format=" << get_format_string(attr.fmt) << '\n'
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
    rknn_tensor_attr native_input_attr_{};
    rknn_tensor_mem* input_memory_ = nullptr;
    std::string model_name_;
    std::string raw_dump_dir_;
    ModelFamily model_family_;
    bool shared_input_ = false;
    bool raw_dumped_ = false;
    bool detections_dumped_ = false;
};

struct Options {
    std::string model;
    std::string video;
    edge::VideoBackend video_backend = edge::VideoBackend::OpenCvFfmpeg;
    edge::PreprocessBackend preprocess_backend = edge::PreprocessBackend::Cpu;
    std::string input_pacing = "media";
    std::string video_memory_mode = "copy";
    std::string rknn_input_mode = "api";
    std::string track_log_path;
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
                         "[--video-backend opencv|gstreamer-mpp] [--preprocess-backend cpu|rga] "
                         "[--video-memory-mode copy|dmabuf] [--rknn-input-mode api|shared] "
                         "[--track-log PATH] "
                         "[--input-pacing media|none] "
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
    if (args.count("--video-backend")) options.video_backend = edge::parse_video_backend(args.at("--video-backend"));
    if (args.count("--preprocess-backend")) options.preprocess_backend = edge::parse_preprocess_backend(args.at("--preprocess-backend"));
    if (args.count("--input-pacing")) options.input_pacing = args.at("--input-pacing");
    if (args.count("--video-memory-mode")) options.video_memory_mode = args.at("--video-memory-mode");
    if (args.count("--rknn-input-mode")) options.rknn_input_mode = args.at("--rknn-input-mode");
    if (args.count("--track-log")) options.track_log_path = args.at("--track-log");
    if (options.input_pacing != "media" && options.input_pacing != "none") {
        throw std::invalid_argument("--input-pacing must be media or none");
    }
    if (options.video_memory_mode != "copy" && options.video_memory_mode != "dmabuf") {
        throw std::invalid_argument("--video-memory-mode must be copy or dmabuf");
    }
    if (options.rknn_input_mode != "api" && options.rknn_input_mode != "shared") {
        throw std::invalid_argument("--rknn-input-mode must be api or shared");
    }
    if (options.video_memory_mode == "dmabuf" &&
        (options.video_backend != edge::VideoBackend::GStreamerMpp ||
         options.preprocess_backend != edge::PreprocessBackend::Rga ||
         options.rknn_input_mode != "shared")) {
        throw std::invalid_argument("DMA-BUF video mode requires GStreamer/MPP, RGA, and shared RKNN input");
    }
    if (options.rknn_input_mode == "shared" && options.preprocess_backend != edge::PreprocessBackend::Rga) {
        throw std::invalid_argument("shared RKNN input mode currently requires RGA preprocessing");
    }
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

struct StageSamples {
    std::deque<double> decode;
    std::deque<double> preprocess;
    std::deque<double> input_set;
    std::deque<double> input_sync;
    std::deque<double> rknn_run;
    std::deque<double> output_fetch;
    std::deque<double> inference;
    std::deque<double> postprocess;
    std::deque<double> queue_wait;
    std::deque<double> end_to_end;

    static void append(std::deque<double>& samples, double value) {
        samples.push_back(value);
        if (samples.size() > 300) samples.pop_front();
    }

    void add(double decode_ms, double preprocess_ms, double input_set_ms, double input_sync_ms,
             double run_ms, double fetch_ms, double postprocess_ms,
             double queue_wait_ms, double end_to_end_ms) {
        append(decode, decode_ms);
        append(preprocess, preprocess_ms);
        append(input_set, input_set_ms);
        append(input_sync, input_sync_ms);
        append(rknn_run, run_ms);
        append(output_fetch, fetch_ms);
        append(inference, run_ms + fetch_ms);
        append(postprocess, postprocess_ms);
        append(queue_wait, queue_wait_ms);
        append(end_to_end, end_to_end_ms);
    }
};

double percentile(const std::deque<double>& samples, double fraction) {
    if (samples.empty()) return 0.0;
    std::vector<double> sorted(samples.begin(), samples.end());
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(std::ceil(fraction * sorted.size())) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

struct RuntimeMetricSnapshot {
    edge::VideoBackend video_backend = edge::VideoBackend::OpenCvFfmpeg;
    edge::PreprocessBackend preprocess_backend = edge::PreprocessBackend::Cpu;
    std::string input_pacing = "media";
    std::string video_memory_mode = "copy";
    std::string rknn_input_mode = "api";
    double decode_ms = 0.0;
    double preprocess_ms = 0.0;
    double input_set_ms = 0.0;
    double input_sync_ms = 0.0;
    double rknn_run_ms = 0.0;
    double output_fetch_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double queue_wait_ms = 0.0;
    double end_to_end_ms = 0.0;
    std::uint64_t backend_dropped_frames = 0;
    std::uint64_t source_copy_bytes_total = 0;
    std::uint64_t source_copy_bytes_last = 0;
    std::uint64_t model_input_bytes_last = 0;
    std::uint64_t reconnect_stale_frames_dropped = 0;
    double nominal_input_fps = 0.0;
};

void emit_metric(std::uint64_t processed, std::uint64_t detections, std::uint64_t decoded,
                 const edge::BoundedQueue<Frame>& queue, Clock::time_point started,
                 const StageSamples& samples, const RuntimeMetricSnapshot& runtime,
                 std::size_t roi_occupancy_count, std::uint64_t cumulative_roi_entries,
                 std::uint64_t safety_event_count, std::uint64_t ppe_violation_count,
                 std::uint64_t danger_zone_count, std::uint64_t conflict_count,
                 const edge::PPEStatusCounts& ppe_status_counts,
                 std::uint64_t reconnect_attempts, std::uint64_t stream_epoch,
                 const char* source_type, const char* source_status, const char* status) {
    std::lock_guard<std::mutex> output_lock(stdout_mutex);
    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const double fps = elapsed > 0.0 ? processed / elapsed : 0.0;
    const ProcessCpuUsage cpu_usage = read_process_cpu_usage();
    std::cout << std::fixed << std::setprecision(3)
              << "{\"type\":\"metric\",\"status\":\"" << status
              << "\",\"processed_frames\":" << processed
              << ",\"decoded_frames\":" << decoded
              << ",\"nominal_input_fps\":" << runtime.nominal_input_fps
              << ",\"source_delivery_fps\":" << (elapsed > 0.0 ? decoded / elapsed : 0.0)
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
              << ",\"backend_dropped_frames\":" << runtime.backend_dropped_frames
              << ",\"reconnect_stale_frames_dropped\":" << runtime.reconnect_stale_frames_dropped
              << ",\"throughput_fps\":" << fps
              << ",\"last_decode_ms\":" << runtime.decode_ms
              << ",\"last_preprocess_ms\":" << runtime.preprocess_ms
              << ",\"last_rknn_input_set_ms\":" << runtime.input_set_ms
              << ",\"last_rknn_input_sync_ms\":" << runtime.input_sync_ms
              << ",\"last_rknn_run_ms\":" << runtime.rknn_run_ms
              << ",\"last_rknn_output_fetch_ms\":" << runtime.output_fetch_ms
              << ",\"last_inference_ms\":" << runtime.inference_ms
              << ",\"last_postprocess_ms\":" << runtime.postprocess_ms
              << ",\"last_queue_wait_ms\":" << runtime.queue_wait_ms
              << ",\"last_end_to_end_ms\":" << runtime.end_to_end_ms
              << ",\"p50_decode_ms\":" << percentile(samples.decode, 0.50)
              << ",\"p95_decode_ms\":" << percentile(samples.decode, 0.95)
              << ",\"p50_preprocess_ms\":" << percentile(samples.preprocess, 0.50)
              << ",\"p95_preprocess_ms\":" << percentile(samples.preprocess, 0.95)
              << ",\"p50_rknn_input_set_ms\":" << percentile(samples.input_set, 0.50)
              << ",\"p95_rknn_input_set_ms\":" << percentile(samples.input_set, 0.95)
              << ",\"p50_rknn_input_sync_ms\":" << percentile(samples.input_sync, 0.50)
              << ",\"p95_rknn_input_sync_ms\":" << percentile(samples.input_sync, 0.95)
              << ",\"p50_rknn_run_ms\":" << percentile(samples.rknn_run, 0.50)
              << ",\"p95_rknn_run_ms\":" << percentile(samples.rknn_run, 0.95)
              << ",\"p50_rknn_output_fetch_ms\":" << percentile(samples.output_fetch, 0.50)
              << ",\"p95_rknn_output_fetch_ms\":" << percentile(samples.output_fetch, 0.95)
              << ",\"p50_inference_ms\":" << percentile(samples.inference, 0.50)
              << ",\"p95_inference_ms\":" << percentile(samples.inference, 0.95)
              << ",\"p50_postprocess_ms\":" << percentile(samples.postprocess, 0.50)
              << ",\"p95_postprocess_ms\":" << percentile(samples.postprocess, 0.95)
              << ",\"p50_queue_wait_ms\":" << percentile(samples.queue_wait, 0.50)
              << ",\"p95_queue_wait_ms\":" << percentile(samples.queue_wait, 0.95)
              << ",\"p50_end_to_end_ms\":" << percentile(samples.end_to_end, 0.50)
              << ",\"p95_end_to_end_ms\":" << percentile(samples.end_to_end, 0.95)
              << ",\"metric_window_frames\":" << samples.end_to_end.size()
              << ",\"video_backend\":\"" << edge::video_backend_name(runtime.video_backend) << "\""
              << ",\"preprocess_backend\":\"" << edge::preprocess_backend_name(runtime.preprocess_backend) << "\""
              << ",\"input_pacing\":\"" << runtime.input_pacing << "\""
              << ",\"video_memory_mode\":\"" << runtime.video_memory_mode << "\""
              << ",\"rknn_input_mode\":\"" << runtime.rknn_input_mode << "\""
              << ",\"source_copy_bytes_total\":" << runtime.source_copy_bytes_total
              << ",\"last_source_copy_bytes\":" << runtime.source_copy_bytes_last
              << ",\"model_input_bytes\":" << runtime.model_input_bytes_last
              << ",\"process_rss_kb\":" << read_rss_kb()
              << ",\"process_peak_rss_kb\":" << read_peak_rss_kb()
              << ",\"process_cpu_core_percent\":" << cpu_usage.core_percent
              << ",\"process_cpu_system_percent\":" << cpu_usage.system_percent
              << ",\"process_cpu_sample_seconds\":" << cpu_usage.sample_seconds
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
        std::ofstream track_log;
        if (!options.track_log_path.empty()) {
            const fs::path track_path(options.track_log_path);
            if (!track_path.parent_path().empty()) fs::create_directories(track_path.parent_path());
            track_log.open(track_path, std::ios::out | std::ios::trunc);
            if (!track_log) throw std::runtime_error("could not open track log: " + options.track_log_path);
        }

        const ModelFamily model_family = options.model_family == "yolox-ppe"
            ? ModelFamily::YoloXPpe
            : ModelFamily::YoloV8Coco;
        if (model_family == ModelFamily::YoloV8Coco && init_post_process() != 0) {
            throw std::runtime_error("failed to load COCO labels from ./model/coco_80_labels_list.txt");
        }
        int exit_code = 0;
        {
            RknnYoloEngine engine(options.model, options.raw_dump_dir, model_family,
                                  options.rknn_input_mode == "shared");
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
            std::atomic<bool> source_connected{false};
            std::atomic<std::uint64_t> backend_dropped{0};
            std::atomic<std::uint64_t> source_copy_bytes{0};
            std::atomic<std::uint64_t> stale_epoch_frames_dropped{0};
            std::atomic<double> nominal_input_fps{0.0};
            std::string producer_error;
            const auto started = Clock::now();
            const auto run_started_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            std::thread producer([&] {
                try {
                    std::unique_ptr<edge::VideoSource> source;
                    std::uint64_t sequence = 0;
                    std::uint64_t local_stream_epoch = 0;
                    double last_pts_ms = -1.0;
                    bool ever_decoded = false;
                    bool awaiting_reconnect = false;
                    double source_fps = 0.0;
                    bool source_mode_logged = false;
                    auto next_frame_at = Clock::now();
                    std::uint64_t accounted_backend_drops = 0;
                    std::chrono::milliseconds retry_delay{500};

                    auto account_backend_drops = [&] {
                        if (!source) return;
                        const auto current = source->dropped_frames();
                        if (current > accounted_backend_drops) {
                            backend_dropped.fetch_add(current - accounted_backend_drops);
                            accounted_backend_drops = current;
                        }
                    };
                    auto close_source = [&] {
                        account_backend_drops();
                        if (source) source->close();
                        source.reset();
                        accounted_backend_drops = 0;
                    };
                    auto open_source = [&](std::string& error) {
                        source = edge::make_video_source(
                            options.video_backend, options.video_memory_mode == "dmabuf");
                        if (!source->open(options.video, options.queue_policy, error)) {
                            source->close();
                            source.reset();
                            return false;
                        }
                        accounted_backend_drops = 0;
                        source_fps = source->nominal_fps();
                        nominal_input_fps.store(source_fps);
                        return true;
                    };

                    while (!shutdown_requested) {
                        if (!source) {
                            std::string open_error;
                            if (!open_source(open_error)) {
                                if (!rtsp_source) {
                                    producer_error = open_error.empty() ? "video source could not be opened" : open_error;
                                    break;
                                }
                                source_connected.store(false);
                                const auto attempt = reconnect_attempts.fetch_add(1) + 1;
                                emit_source_status(ever_decoded ? "reconnecting" : "connecting",
                                                   attempt, local_stream_epoch);
                                std::cerr << "RTSP " << edge::video_backend_name(options.video_backend)
                                          << " open attempt=" << attempt << " failed: " << open_error << '\n';
                                if (!wait_for_retry(retry_delay)) break;
                                retry_delay = std::min(retry_delay * 2, std::chrono::milliseconds(5000));
                                continue;
                            }
                        }

                        edge::DecodedFrame decoded_frame;
                        std::string read_error;
                        edge::FrameReadStatus read_status = edge::FrameReadStatus::Error;
                        try {
                            read_status = source->read(decoded_frame, read_error);
                        } catch (const std::exception& exception) {
                            read_error = exception.what();
                            read_status = edge::FrameReadStatus::Error;
                        }
                        account_backend_drops();
                        if (read_status == edge::FrameReadStatus::Timeout) continue;
                        if (read_status == edge::FrameReadStatus::EndOfStream ||
                            read_status == edge::FrameReadStatus::Error) {
                            if (rtsp_source) {
                                awaiting_reconnect = ever_decoded;
                                source_connected.store(false);
                                close_source();
                                if (!read_error.empty()) {
                                    std::cerr << "RTSP source reconnecting after: " << read_error << '\n';
                                }
                                const auto attempt = reconnect_attempts.fetch_add(1) + 1;
                                emit_source_status(ever_decoded ? "reconnecting" : "connecting",
                                                   attempt, local_stream_epoch);
                                std::cerr << "RTSP reconnect attempt=" << attempt << '\n';
                                if (!wait_for_retry(retry_delay)) break;
                                retry_delay = std::min(retry_delay * 2, std::chrono::milliseconds(5000));
                                continue;
                            }
                            if (read_status == edge::FrameReadStatus::Error) {
                                producer_error = read_error.empty() ? "video source read failed" : read_error;
                                break;
                            }
                            if (!options.loop_video) break;
                            close_source();
                            std::string open_error;
                            if (!open_source(open_error)) {
                                producer_error = "could not reopen looped video: " + open_error;
                                break;
                            }
                            ++local_stream_epoch;
                            stream_epoch.store(local_stream_epoch);
                            last_pts_ms = -1.0;
                            next_frame_at = Clock::now();
                            std::cerr << "local video loop restarted epoch=" << local_stream_epoch << '\n';
                            continue;
                        }

                        if (!rtsp_source && source_fps <= 0.0) {
                            source_fps = source->nominal_fps();
                            nominal_input_fps.store(source_fps);
                        }
                        if (!source_mode_logged) {
                            if (rtsp_source) {
                                std::cerr << "video source_type=rtsp backend=" << source->name()
                                          << " pacing=source-clock reconnect=enabled\n";
                            } else {
                                std::cerr << "video source_type=local-file backend=" << source->name()
                                          << " source_fps=" << source_fps
                                          << " pacing=" << (options.input_pacing == "media" ? "media-rate" : "disabled")
                                          << " looping=" << (options.loop_video ? "enabled" : "disabled") << '\n';
                            }
                            source_mode_logged = true;
                        }
                        if (rtsp_source) {
                            if (awaiting_reconnect) {
                                ++local_stream_epoch;
                                stream_epoch.store(local_stream_epoch);
                                awaiting_reconnect = false;
                                source_connected.store(true);
                                retry_delay = std::chrono::milliseconds(500);
                                std::cerr << "RTSP stream reconnected epoch=" << local_stream_epoch << '\n';
                                emit_source_status("connected", reconnect_attempts.load(), local_stream_epoch);
                            }
                            if (!ever_decoded) {
                                source_connected.store(true);
                                emit_source_status("connected", reconnect_attempts.load(), local_stream_epoch);
                            }
                            ever_decoded = true;
                        }
                        ++sequence;
                        double pts_ms = rtsp_source
                            ? milliseconds(decoded_frame.decoded_at - started)
                            : decoded_frame.pts_ms;
                        if (!std::isfinite(pts_ms) || pts_ms < 0.0 ||
                            (sequence > 1 && pts_ms <= last_pts_ms)) {
                            pts_ms = !rtsp_source && source_fps > 0.0
                                ? static_cast<double>(sequence - 1) * 1000.0 / source_fps
                                : milliseconds(decoded_frame.decoded_at - started);
                        }
                        if (!rtsp_source && sequence > 1 && pts_ms <= last_pts_ms) {
                            pts_ms = last_pts_ms + (source_fps > 0.0 ? 1000.0 / source_fps : 1.0);
                        }
                        Frame item;
                        item.media = std::move(decoded_frame);
                        item.media.sequence = sequence;
                        item.media.stream_epoch = local_stream_epoch;
                        item.media.pts_ms = pts_ms;
                        item.sequence = sequence;
                        item.pts_ms = pts_ms;
                        item.stream_epoch = local_stream_epoch;
                        last_pts_ms = pts_ms;
                        source_copy_bytes.fetch_add(item.media.source_copy_bytes);
                        if (queue.push(std::move(item)) == edge::QueuePushResult::Closed) break;
                        decoded.store(sequence);
                        if (!rtsp_source && options.input_pacing == "media" && source_fps > 0.0) {
                            const auto frame_period = std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(1.0 / source_fps));
                            next_frame_at += frame_period;
                            const auto now = Clock::now();
                            if (next_frame_at > now) std::this_thread::sleep_until(next_frame_at);
                            else next_frame_at = now;
                        }
                    }
                    close_source();
                    queue.close();
                } catch (const std::exception& error) {
                    producer_error = rtsp_source ? "RTSP source failed unexpectedly" : error.what();
                    queue.close();
                }
            });

            edge::IoUTracker tracker;
            edge::RoiEventMachine roi_event_machine(options.dwell_threshold,
                                                    options.roi_debounce_frames);
            edge::PPEEventMachine ppe_event_machine(ppe_event_config);
            edge::PPEStatusCounts ppe_status_counts;
            std::uint64_t processed_stream_epoch = 0;
            StageSamples stage_samples;
            RuntimeMetricSnapshot runtime;
            runtime.video_backend = options.video_backend;
            runtime.preprocess_backend = options.preprocess_backend;
            runtime.input_pacing = rtsp_source ? "source-clock" : options.input_pacing;
            runtime.video_memory_mode = options.video_memory_mode;
            runtime.rknn_input_mode = options.rknn_input_mode;
            std::uint64_t processed = 0;
            std::uint64_t detection_total = 0;
            std::uint64_t event_total = 0;
            std::size_t roi_occupancy_count = 0;
            std::uint64_t cumulative_roi_entries = 0;
            double roi_timeline_ms = -1.0;
            Frame frame;
            auto report_metric = [&](const char* status) {
                runtime.backend_dropped_frames = backend_dropped.load();
                runtime.source_copy_bytes_total = source_copy_bytes.load();
                runtime.reconnect_stale_frames_dropped = stale_epoch_frames_dropped.load();
                runtime.nominal_input_fps = nominal_input_fps.load();
                emit_metric(processed, detection_total, decoded.load(), queue, started,
                            stage_samples, runtime,
                            roi_occupancy_count, cumulative_roi_entries,
                            ppe_event_machine.total_events(), ppe_event_machine.ppe_violations(),
                            ppe_event_machine.danger_zone_intrusions(),
                            ppe_event_machine.forklift_person_cooccurrences(), ppe_status_counts,
                            reconnect_attempts.load(), stream_epoch.load(),
                            rtsp_source ? "rtsp" : "local-file",
                            rtsp_source
                                ? (source_connected.load() ? "connected" : "reconnecting")
                                : "not-applicable",
                            status);
            };
            while (queue.pop(frame)) {
                if (frame.stream_epoch < stream_epoch.load()) {
                    stale_epoch_frames_dropped.fetch_add(1);
                    continue;
                }
                if (frame.stream_epoch != processed_stream_epoch) {
                    tracker.reset();
                    roi_event_machine.reset_stream();
                    ppe_event_machine.reset_stream();
                    roi_occupancy_count = 0;
                    processed_stream_epoch = frame.stream_epoch;
                    std::cerr << "stream state reset epoch=" << processed_stream_epoch << '\n';
                }
                const auto process_start = Clock::now();
                runtime.queue_wait_ms = milliseconds(process_start - frame.media.decoded_at);
                try {
                    auto result = engine.infer(frame.media, options.confidence, options.preprocess_backend);
                    runtime.decode_ms = frame.media.decode_ms;
                    runtime.preprocess_ms = result.preprocess_ms;
                    runtime.input_set_ms = result.input_set_ms;
                    runtime.input_sync_ms = result.input_sync_ms;
                    runtime.rknn_run_ms = result.rknn_run_ms;
                    runtime.output_fetch_ms = result.output_fetch_ms;
                    runtime.inference_ms = result.inference_ms;
                    runtime.postprocess_ms = result.postprocess_ms;
                    runtime.source_copy_bytes_last = frame.media.source_copy_bytes;
                    runtime.model_input_bytes_last = result.model_input_bytes;
                    detection_total += result.detections.size();
                    const auto tracked = tracker.update(result.detections);
                    if (track_log.is_open()) {
                        track_log << std::fixed << std::setprecision(4)
                                  << "{\"frame_sequence\":" << frame.sequence
                                  << ",\"source_pts_ms\":" << frame.pts_ms
                                  << ",\"detections\":[";
                        for (std::size_t index = 0; index < tracked.size(); ++index) {
                            if (index != 0) track_log << ',';
                            const auto& item = tracked[index];
                            const auto& d = item.detection;
                            track_log << "{\"track_id\":" << item.track_id
                                      << ",\"class_id\":" << d.class_id
                                      << ",\"confidence\":" << d.confidence
                                      << ",\"box\":[" << d.left << ',' << d.top << ','
                                      << d.right << ',' << d.bottom << ']'
                                      << ",\"new_track\":" << (item.created ? "true" : "false")
                                      << '}';
                        }
                        track_log << "]}\n";
                        if (!track_log) throw std::runtime_error("failed writing track log");
                    }
                    std::vector<edge::RoiObservation> observations;
                    observations.reserve(tracked.size());

                    for (const auto& item : tracked) {
                        const auto& d = item.detection;
                        const bool inside = options.roi.contains(d, frame.media.width, frame.media.height);
                        observations.push_back({item.track_id, d, inside});
                    }
                    const auto active = tracker.active_ids();
                    double event_time_ms = frame.pts_ms;
                    if (!std::isfinite(event_time_ms) || event_time_ms < 0.0 ||
                        event_time_ms <= roi_timeline_ms) {
                        event_time_ms = milliseconds(frame.media.decoded_at - started);
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
                            tracked, active, frame.media.width, frame.media.height, event_time);
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
                        cv::Mat source_bgr = edge::frame_to_bgr(frame.media);
                        cv::Mat annotated = source_bgr.clone();
                        for (const auto& item : tracked) {
                            const auto& d = item.detection;
                            cv::rectangle(annotated, cv::Point(d.left, d.top), cv::Point(d.right, d.bottom),
                                          cv::Scalar(40, 220, 40), 2);
                            std::ostringstream label;
                            label << (d.label ? d.label : "object") << ' ' << std::fixed
                                  << std::setprecision(2) << d.confidence << " T" << item.track_id;
                            cv::putText(annotated, label.str(),
                                        cv::Point(std::max(0, d.left), std::max(18, d.top - 4)),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(40, 220, 40), 1);
                        }
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
                    runtime.end_to_end_ms = milliseconds(Clock::now() - frame.media.decoded_at);
                    stage_samples.add(runtime.decode_ms, runtime.preprocess_ms, runtime.input_set_ms,
                                      runtime.input_sync_ms,
                                      runtime.rknn_run_ms, runtime.output_fetch_ms,
                                      runtime.postprocess_ms, runtime.queue_wait_ms,
                                      runtime.end_to_end_ms);
                    if (processed == 1 || processed % 30 == 0) {
                        report_metric("running");
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
            report_metric(exit_code == 0 ? "completed" : "error");
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
