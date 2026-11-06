#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "preprocess.hpp"
#include "video_source.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;

    const auto nonce = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    const fs::path video_path = fs::temp_directory_path() /
                                ("rk_edge_station_media_test_" + std::to_string(nonce) + ".mp4");
    struct RemoveFile {
        fs::path path;
        ~RemoveFile() {
            std::error_code ignored;
            fs::remove(path, ignored);
        }
    } cleanup{video_path};
    constexpr int width = 320;
    constexpr int height = 240;
    constexpr int expected_frames = 60;
    constexpr double expected_fps = 30.0;
    const int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    cv::VideoWriter writer(video_path.string(), codec, expected_fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "MP4 writer unavailable for codec mp4v\n";
        return 1;
    }
    for (int i = 0; i < expected_frames; ++i) {
        cv::Mat frame(height, width, CV_8UC3,
                      cv::Scalar((i * 7) % 256, (i * 13) % 256, (i * 19) % 256));
        writer.write(frame);
    }
    writer.release();

    cv::VideoCapture capture(video_path.string());
    if (!capture.isOpened()) {
        std::cerr << "OpenCV could not reopen generated MP4\n";
        return 1;
    }
    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || std::abs(fps - expected_fps) > 0.5 ||
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)) != width ||
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)) != height) {
        std::cerr << "MP4 metadata mismatch: fps=" << fps
                  << " width=" << capture.get(cv::CAP_PROP_FRAME_WIDTH)
                  << " height=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << '\n';
        return 1;
    }

    int decoded_frames = 0;
    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty() || frame.cols != width || frame.rows != height) {
            std::cerr << "decoded MP4 frame has invalid dimensions\n";
            return 1;
        }
        ++decoded_frames;
    }
    capture.release();

    if (decoded_frames != expected_frames) {
        std::cerr << "decoded " << decoded_frames << " frames; expected "
                  << expected_frames << '\n';
        return 1;
    }

    cv::Mat black_nv12(height * 3 / 2, width, CV_8UC1);
    black_nv12.rowRange(0, height).setTo(16);
    black_nv12.rowRange(height, height * 3 / 2).setTo(128);
    const cv::Mat black_bgr = edge::nv12_to_bgr(black_nv12, width, height);
    if (black_bgr.size() != cv::Size(width, height) || black_bgr.type() != CV_8UC3 ||
        cv::mean(black_bgr)[0] > 2.0 || cv::mean(black_bgr)[1] > 2.0 ||
        cv::mean(black_bgr)[2] > 2.0) {
        std::cerr << "NV12 limited-range black conversion mismatch\n";
        return 1;
    }
    bool invalid_nv12_rejected = false;
    try {
        (void)edge::nv12_to_bgr(black_nv12, width - 1, height);
    } catch (const std::invalid_argument&) {
        invalid_nv12_rejected = true;
    }
    if (!invalid_nv12_rejected) {
        std::cerr << "NV12 conversion accepted invalid plane dimensions\n";
        return 1;
    }

    edge::DecodedFrame color_frame{};
    color_frame.width = width;
    color_frame.height = height;
    color_frame.bgr.create(height, width, CV_8UC3);
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            color_frame.bgr.at<cv::Vec3b>(row, column) = cv::Vec3b(
                static_cast<std::uint8_t>((row + column) % 256),
                static_cast<std::uint8_t>((row * 3 + column) % 256),
                static_cast<std::uint8_t>((row + column * 5) % 256));
        }
    }
    constexpr int model_size = 64;
    const auto reference = edge::preprocess_frame(
        color_frame, model_size, model_size, false, edge::PreprocessBackend::Cpu);
    cv::Mat caller_owned_input(model_size, model_size, CV_8UC3);
    const auto into_caller_memory = edge::preprocess_frame_into(
        color_frame, model_size, model_size, false, edge::PreprocessBackend::Cpu,
        caller_owned_input);
    if (into_caller_memory.model_input.data != caller_owned_input.data ||
        into_caller_memory.model_input_bytes != model_size * model_size * 3ULL ||
        cv::norm(reference.model_input, into_caller_memory.model_input, cv::NORM_INF) != 0.0) {
        std::cerr << "caller-owned preprocessing output differs from the CPU reference\n";
        return 1;
    }
    bool invalid_destination_rejected = false;
    try {
        (void)edge::preprocess_frame_into(
            color_frame, model_size, model_size, false, edge::PreprocessBackend::Cpu,
            cv::Mat(model_size, model_size, CV_8UC1));
    } catch (const std::invalid_argument&) {
        invalid_destination_rejected = true;
    }
    if (!invalid_destination_rejected) {
        std::cerr << "caller-owned preprocessing accepted an invalid destination type\n";
        return 1;
    }
    edge::DecodedFrame unmapped_dma_frame{};
    unmapped_dma_frame.width = width;
    unmapped_dma_frame.height = height;
    unmapped_dma_frame.dma_buf_fd = 7;
    bool unmapped_dma_rejected = false;
    try {
        (void)edge::preprocess_frame_into(
            unmapped_dma_frame, model_size, model_size, false, edge::PreprocessBackend::Cpu,
            cv::Mat(model_size, model_size, CV_8UC3));
    } catch (const std::invalid_argument&) {
        unmapped_dma_rejected = true;
    }
    if (!unmapped_dma_rejected) {
        std::cerr << "CPU preprocessing accepted a DMA-BUF frame without a CPU mapping\n";
        return 1;
    }
    std::cout << "MP4 encode/decode passed: frames=" << decoded_frames
              << " fps=" << fps << " size=" << width << 'x' << height
              << "; NV12 conversion and caller-owned preprocessing checks passed\n";
    return 0;
}
