#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
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
    std::cout << "MP4 encode/decode passed: frames=" << decoded_frames
              << " fps=" << fps << " size=" << width << 'x' << height << '\n';
    return 0;
}
