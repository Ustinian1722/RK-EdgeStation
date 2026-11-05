cmake_minimum_required(VERSION 3.16)
project(RKEdgeStation VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_path(RKNN_INCLUDE_DIR rknn_api.h REQUIRED)
find_library(RKNN_LIBRARY NAMES rknnrt REQUIRED)

add_executable(rk_edge_worker
    src/main.cpp
    src/tracker.cpp
    src/roi_events.cpp
    src/third_party/rknn_model_zoo_v2.0.0/yolov8/postprocess.cc
)

add_executable(create_static_video tools/create_static_video.cpp)
target_include_directories(create_static_video PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(create_static_video PRIVATE ${OpenCV_LIBS})
target_compile_options(create_static_video PRIVATE -Wall -Wextra -Wpedantic)

target_include_directories(rk_edge_worker PRIVATE
    ${RKNN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    src
    src/third_party/rknn_model_zoo_v2.0.0/utils
    src/third_party/rknn_model_zoo_v2.0.0/yolov8
)
target_link_libraries(rk_edge_worker PRIVATE ${OpenCV_LIBS} ${RKNN_LIBRARY} pthread)
target_compile_options(rk_edge_worker PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(rk_edge_worker PROPERTIES BUILD_RPATH "/lib;/usr/lib/aarch64-linux-gnu")

include(CTest)
if(BUILD_TESTING)
    add_executable(edge_core_tests tests/cpp_core_tests.cpp src/tracker.cpp src/roi_events.cpp)
    target_include_directories(edge_core_tests PRIVATE src)
    target_compile_options(edge_core_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME edge_core_tests COMMAND edge_core_tests)

    add_executable(edge_media_tests tests/media_pipeline_tests.cpp)
    target_include_directories(edge_media_tests PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(edge_media_tests PRIVATE ${OpenCV_LIBS})
    target_compile_options(edge_media_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME edge_media_tests COMMAND edge_media_tests)
endif()
