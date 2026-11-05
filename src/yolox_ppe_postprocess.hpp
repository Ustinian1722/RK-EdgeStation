#pragma once

#include "yolov8.h"

const char* ppe_class_to_name(int class_id) noexcept;

int post_process_yolox_ppe(rknn_app_context_t* app_ctx, const rknn_output* outputs,
                           const letterbox_t* letter_box, float confidence_threshold,
                           float nms_threshold, int source_width, int source_height,
                           object_detect_result_list* results);
