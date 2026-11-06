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
PreprocessBackend parse_preprocess_backend(const std::string& value);
const char* preprocess_backend_name(PreprocessBackend backend) noexcept;

}  // namespace edge
