#pragma once

#include "video_source.hpp"
#include "yolov8.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

namespace edge {

enum class PreprocessBackend { Cpu, Rga };

struct PreprocessResult {
    cv::Mat model_input;
    letterbox_t letterbox{};
    double elapsed_ms = 0.0;
    // Tensor payload passed to rknn_inputs_set; not a count of all internal copies.
    std::uint64_t model_input_bytes = 0;
};

PreprocessResult preprocess_frame(const DecodedFrame& frame, int model_width, int model_height,
                                  bool output_rgb, PreprocessBackend backend);
// Writes the prepared tensor into caller-owned memory. With an RKNN DMA-BUF fd
// and a retained MPP DMA-BUF frame, RGA can import both handles and avoid the
// CPU-visible NV12/model-input pixel copies. The destination cv::Mat is a view;
// its storage must outlive inference.
PreprocessResult preprocess_frame_into(const DecodedFrame& frame, int model_width, int model_height,
                                       bool output_rgb, PreprocessBackend backend,
                                       cv::Mat destination, int destination_fd = -1,
                                       std::uint64_t destination_size = 0);
PreprocessBackend parse_preprocess_backend(const std::string& value);
const char* preprocess_backend_name(PreprocessBackend backend) noexcept;

}  // namespace edge
