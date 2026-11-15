#pragma once

#include "tracker.hpp"

#include "image_utils.h"

#include <vector>

namespace edge {

// Decode the released Constructelligence v0.3 YOLOv8 output contract:
// [1, 18, 8400] = cx, cy, w, h, then 14 class probabilities.
// Only source classes that map to this application's existing PPE taxonomy
// are returned; unsupported classes are intentionally ignored.
std::vector<Detection> decode_constructelligence_ppe(
    const float* output, int channels, int anchors, const letterbox_t& letterbox,
    int source_width, int source_height, float confidence_threshold,
    float nms_threshold);

}  // namespace edge
