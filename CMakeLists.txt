cmake_minimum_required(VERSION 3.16)
project(RKEdgeStation VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_path(RKNN_INCLUDE_DIR rknn_api.h REQUIRED)
find_library(RKNN_LIBRARY NAMES rknnrt REQUIRED)

option(EDGE_ENABLE_GSTREAMER "Build the GStreamer/MPP video source" ON)
option(EDGE_ENABLE_RGA "Build RGA preprocessing support" ON)
if(EDGE_ENABLE_GSTREAMER)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    pkg_check_modules(GSTREAMER_ALLOCATORS REQUIRED IMPORTED_TARGET
        gstreamer-allocators-1.0)
endif()
if(EDGE_ENABLE_RGA)
    find_path(RGA_INCLUDE_DIR rga/im2d.hpp REQUIRED)
    find_library(RGA_LIBRARY NAMES rga REQUIRED)
endif()

if(EDGE_ENABLE_GSTREAMER AND EDGE_ENABLE_RGA)
    add_executable(gst_dmabuf_probe tools/gst_dmabuf_probe.cpp)
    target_include_directories(gst_dmabuf_probe PRIVATE ${RGA_INCLUDE_DIR})
    target_link_libraries(gst_dmabuf_probe PRIVATE
        PkgConfig::GSTREAMER PkgConfig::GSTREAMER_ALLOCATORS ${RGA_LIBRARY})
    target_compile_options(gst_dmabuf_probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(rknn_shared_input_probe
        tools/rknn_shared_input_probe.cpp
        src/preprocess.cpp
        src/video_source.cpp
        src/yolox_ppe_postprocess.cpp
    )
    target_include_directories(rknn_shared_input_probe PRIVATE
        ${RKNN_INCLUDE_DIR}
        ${RGA_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        src
        src/third_party/rknn_model_zoo_v2.0.0/utils
        src/third_party/rknn_model_zoo_v2.0.0/yolov8
    )
    target_compile_definitions(rknn_shared_input_probe PRIVATE
        EDGE_HAS_GSTREAMER=1 EDGE_HAS_RGA=1)
    target_link_libraries(rknn_shared_input_probe PRIVATE
        ${OpenCV_LIBS} ${RKNN_LIBRARY} ${RGA_LIBRARY}
        PkgConfig::GSTREAMER PkgConfig::GSTREAMER_ALLOCATORS pthread)
    target_compile_options(rknn_shared_input_probe PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(rknn_shared_input_probe PROPERTIES BUILD_RPATH "/lib;/usr/lib/aarch64-linux-gnu")
endif()

if(EDGE_ENABLE_GSTREAMER)
    add_executable(gst_mpp_lifecycle_probe
        tools/gst_mpp_lifecycle_probe.cpp
        src/video_source.cpp
    )
    target_include_directories(gst_mpp_lifecycle_probe PRIVATE ${OpenCV_INCLUDE_DIRS} src)
    target_compile_definitions(gst_mpp_lifecycle_probe PRIVATE EDGE_HAS_GSTREAMER=1)
    target_link_libraries(gst_mpp_lifecycle_probe PRIVATE
        ${OpenCV_LIBS} PkgConfig::GSTREAMER PkgConfig::GSTREAMER_ALLOCATORS pthread)
    target_compile_options(gst_mpp_lifecycle_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(rk_edge_worker
    src/main.cpp
    src/tracker.cpp
    src/roi_events.cpp
    src/ppe_events.cpp
    src/preprocess.cpp
    src/constructelligence_ppe_postprocess.cpp
    src/yolox_ppe_postprocess.cpp
    src/video_source.cpp
    src/third_party/rknn_model_zoo_v2.0.0/yolov8/postprocess.cc
)

add_executable(create_static_video tools/create_static_video.cpp)
target_include_directories(create_static_video PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(create_static_video PRIVATE ${OpenCV_LIBS})
target_compile_options(create_static_video PRIVATE -Wall -Wextra -Wpedantic)

# Offline quality evaluation for the deployed PPE RKNN variants. This target
# reuses the production CPU preprocessing and YOLOX postprocessor while keeping
# dataset evaluation out of the live worker.
add_executable(ppe_rknn_dataset_eval
    tools/ppe_rknn_dataset_eval.cpp
    src/preprocess.cpp
    src/video_source.cpp
    src/yolox_ppe_postprocess.cpp
)
target_include_directories(ppe_rknn_dataset_eval PRIVATE
    ${RKNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src
    src/third_party/rknn_model_zoo_v2.0.0/utils
    src/third_party/rknn_model_zoo_v2.0.0/yolov8
)
target_link_libraries(ppe_rknn_dataset_eval PRIVATE ${OpenCV_LIBS} ${RKNN_LIBRARY} pthread)
target_compile_options(ppe_rknn_dataset_eval PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(ppe_rknn_dataset_eval PROPERTIES BUILD_RPATH "/lib;/usr/lib/aarch64-linux-gnu")

# Generic raw-output probe for released YOLOv8-style RKNN models. This is a
# model-selection tool, not a Worker backend: it measures Runtime compatibility
# and emits the first output tensor for numerical comparison against ONNX.
add_executable(rknn_yolov8_raw_bench tools/rknn_yolov8_raw_bench.cpp)
target_include_directories(rknn_yolov8_raw_bench PRIVATE
    ${RKNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(rknn_yolov8_raw_bench PRIVATE ${OpenCV_LIBS} ${RKNN_LIBRARY})
target_compile_options(rknn_yolov8_raw_bench PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rknn_yolov8_raw_bench PROPERTIES BUILD_RPATH "/lib;/usr/lib/aarch64-linux-gnu")

# Replays host-exported candidate detections through the production tracker and
# PPE state machine so model comparisons use the same event confirmation and
# cooldown semantics as the Worker without changing the live inference path.
add_executable(replay_ppe_prediction_events
    tools/replay_ppe_prediction_events.cpp
    src/tracker.cpp
    src/ppe_events.cpp
)
target_include_directories(replay_ppe_prediction_events PRIVATE src)
target_compile_options(replay_ppe_prediction_events PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(rk_edge_worker PRIVATE
    ${RKNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src
    src/third_party/rknn_model_zoo_v2.0.0/utils
    src/third_party/rknn_model_zoo_v2.0.0/yolov8
)
target_link_libraries(rk_edge_worker PRIVATE ${OpenCV_LIBS} ${RKNN_LIBRARY} pthread)
if(EDGE_ENABLE_GSTREAMER)
    target_compile_definitions(rk_edge_worker PRIVATE EDGE_HAS_GSTREAMER=1)
    target_link_libraries(rk_edge_worker PRIVATE
        PkgConfig::GSTREAMER PkgConfig::GSTREAMER_ALLOCATORS)
endif()
if(EDGE_ENABLE_RGA)
    target_compile_definitions(rk_edge_worker PRIVATE EDGE_HAS_RGA=1)
    target_include_directories(rk_edge_worker PRIVATE ${RGA_INCLUDE_DIR})
    target_link_libraries(rk_edge_worker PRIVATE ${RGA_LIBRARY})
endif()
target_compile_options(rk_edge_worker PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rk_edge_worker PROPERTIES BUILD_RPATH "/lib;/usr/lib/aarch64-linux-gnu")

include(CTest)
if(BUILD_TESTING)
    add_executable(edge_core_tests tests/cpp_core_tests.cpp src/tracker.cpp src/roi_events.cpp src/ppe_events.cpp)
    target_include_directories(edge_core_tests PRIVATE src)
    target_compile_options(edge_core_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME edge_core_tests COMMAND edge_core_tests)

    add_executable(constructelligence_ppe_postprocess_tests
        tests/constructelligence_ppe_postprocess_tests.cpp
        src/constructelligence_ppe_postprocess.cpp
    )
    target_include_directories(constructelligence_ppe_postprocess_tests PRIVATE
        ${RKNN_INCLUDE_DIR}
        src
        src/third_party/rknn_model_zoo_v2.0.0/utils
    )
    target_compile_options(constructelligence_ppe_postprocess_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME constructelligence_ppe_postprocess_tests COMMAND constructelligence_ppe_postprocess_tests)

    add_executable(edge_media_tests
        tests/media_pipeline_tests.cpp
        src/preprocess.cpp
        src/video_source.cpp
    )
    target_include_directories(edge_media_tests PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${RKNN_INCLUDE_DIR}
        src
        src/third_party/rknn_model_zoo_v2.0.0/utils
        src/third_party/rknn_model_zoo_v2.0.0/yolov8
    )
    target_link_libraries(edge_media_tests PRIVATE ${OpenCV_LIBS})
    target_compile_options(edge_media_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME edge_media_tests COMMAND edge_media_tests)
endif()
