#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#ifdef EDGE_HAS_RGA
#include <rga/im2d.hpp>
#include <rga/rga.h>
#endif

namespace edge {
namespace {

using Clock = std::chrono::steady_clock;

cv::Mat source_bgr(const DecodedFrame& frame) {
    if (!frame.bgr.empty()) return frame.bgr;
    if (!frame.nv12.empty()) return nv12_to_bgr(frame.nv12, frame.width, frame.height);
    throw std::invalid_argument("decoded frame has no BGR or NV12 pixels");
}

void set_letterbox(const DecodedFrame& frame, int model_width, int model_height,
                   PreprocessResult& result, int& resized_width, int& resized_height,
                   int& pad_x, int& pad_y) {
    if (frame.width <= 0 || frame.height <= 0 || model_width <= 0 || model_height <= 0) {
        throw std::invalid_argument("source and model dimensions must be positive");
    }
    const float scale = std::min(static_cast<float>(model_width) / frame.width,
                                 static_cast<float>(model_height) / frame.height);
    resized_width = std::max(1, static_cast<int>(std::round(frame.width * scale)));
    resized_height = std::max(1, static_cast<int>(std::round(frame.height * scale)));
    pad_x = (model_width - resized_width) / 2;
    pad_y = (model_height - resized_height) / 2;
    result.letterbox.x_pad = pad_x;
    result.letterbox.y_pad = pad_y;
    result.letterbox.scale = scale;
}

PreprocessResult preprocess_cpu(const DecodedFrame& frame, int model_width, int model_height,
                                bool output_rgb) {
    const auto start = Clock::now();
    PreprocessResult result;
    int resized_width = 0, resized_height = 0, pad_x = 0, pad_y = 0;
    set_letterbox(frame, model_width, model_height, result,
                  resized_width, resized_height, pad_x, pad_y);
    const cv::Mat bgr = source_bgr(frame);
    if (bgr.type() != CV_8UC3 || bgr.cols != frame.width || bgr.rows != frame.height) {
        throw std::invalid_argument("CPU preprocessing requires a source-sized BGR uint8 frame");
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0,
               cv::INTER_LINEAR);
    cv::Mat padded(model_height, model_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
    if (output_rgb) cv::cvtColor(padded, result.model_input, cv::COLOR_BGR2RGB);
    else result.model_input = std::move(padded);
    result.model_input_bytes = static_cast<std::uint64_t>(result.model_input.total()) *
                               result.model_input.elemSize();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}

#ifdef EDGE_HAS_RGA
void require_rga(IM_STATUS status, const char* operation) {
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        const char* message = imStrError_t(status);
        throw std::runtime_error(std::string(operation) + " failed: " +
            (message == nullptr ? std::to_string(status) : message));
    }
}

PreprocessResult preprocess_rga(const DecodedFrame& frame, int model_width, int model_height,
                                bool output_rgb) {
    const auto start = Clock::now();
    PreprocessResult result;
    int resized_width = 0, resized_height = 0, pad_x = 0, pad_y = 0;
    set_letterbox(frame, model_width, model_height, result,
                  resized_width, resized_height, pad_x, pad_y);
    const bool use_nv12 = !frame.nv12.empty();
    if (!use_nv12 && (frame.bgr.empty() || frame.bgr.type() != CV_8UC3 ||
                      frame.bgr.cols != frame.width || frame.bgr.rows != frame.height)) {
        throw std::invalid_argument("RGA preprocessing requires a source-sized BGR or packed NV12 frame");
    }
    if (use_nv12 && (frame.nv12.type() != CV_8UC1 || frame.nv12.cols != frame.width ||
                     frame.nv12.rows != frame.height * 3 / 2 || !frame.nv12.isContinuous())) {
        throw std::invalid_argument("RGA preprocessing requires canonical contiguous NV12");
    }

    const int output_format = output_rgb ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888;
    result.model_input.create(model_height, model_width, CV_8UC3);
    result.model_input.setTo(cv::Scalar(114, 114, 114));
    const rga_buffer_t source = use_nv12
        ? wrapbuffer_virtualaddr(frame.nv12.data, frame.width, frame.height,
                                 RK_FORMAT_YCbCr_420_SP, frame.width, frame.height)
        : wrapbuffer_virtualaddr(frame.bgr.data, frame.width, frame.height,
                                 RK_FORMAT_BGR_888, frame.width, frame.height);
    const rga_buffer_t destination = wrapbuffer_virtualaddr(
        result.model_input.data, model_width, model_height, output_format,
        model_width, model_height);
    const im_rect source_rect{0, 0, frame.width, frame.height};
    const im_rect destination_rect{pad_x, pad_y, resized_width, resized_height};
    const im_rect empty_rect{};
    const rga_buffer_t empty_buffer{};
    im_opt_t operation_options{};
    operation_options.interp = INTER_LINEAR;
    require_rga(improcess(source, destination, empty_buffer, source_rect, destination_rect,
                          empty_rect, -1, nullptr, &operation_options, IM_SYNC),
                "RGA NV12/BGR resize, color, and letterbox");
    result.model_input_bytes = static_cast<std::uint64_t>(result.model_input.total()) *
                               result.model_input.elemSize();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}
#endif

}  // namespace

PreprocessResult preprocess_frame(const DecodedFrame& frame, int model_width, int model_height,
                                  bool output_rgb, PreprocessBackend backend) {
    switch (backend) {
        case PreprocessBackend::Cpu:
            return preprocess_cpu(frame, model_width, model_height, output_rgb);
        case PreprocessBackend::Rga:
#ifdef EDGE_HAS_RGA
            return preprocess_rga(frame, model_width, model_height, output_rgb);
#else
            throw std::runtime_error("RGA support was not built; configure with EDGE_ENABLE_RGA=ON");
#endif
    }
    throw std::invalid_argument("unknown preprocessing backend");
}

PreprocessBackend parse_preprocess_backend(const std::string& value) {
    if (value == "cpu") return PreprocessBackend::Cpu;
    if (value == "rga") return PreprocessBackend::Rga;
    throw std::invalid_argument("preprocessing backend must be cpu or rga");
}

const char* preprocess_backend_name(PreprocessBackend backend) noexcept {
    switch (backend) {
        case PreprocessBackend::Cpu: return "cpu";
        case PreprocessBackend::Rga: return "rga";
    }
    return "unknown";
}

}  // namespace edge
