#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "create_static_video INPUT_VIDEO FRAME_INDEX OUTPUT.avi FPS DURATION_SECONDS\n";
            return 0;
        }
        if (argc != 6) {
            throw std::invalid_argument(
                "usage: create_static_video INPUT_VIDEO FRAME_INDEX OUTPUT.avi FPS DURATION_SECONDS");
        }
        const fs::path input_path(argv[1]);
        const long long frame_index = std::stoll(argv[2]);
        const fs::path output_path(argv[3]);
        const double fps = std::stod(argv[4]);
        const double duration_seconds = std::stod(argv[5]);
        if (!fs::is_regular_file(input_path)) throw std::runtime_error("input video does not exist");
        if (frame_index < 0) throw std::invalid_argument("frame index must be non-negative");
        if (!std::isfinite(fps) || fps <= 0.0 || fps > 60.0) {
            throw std::invalid_argument("FPS must be in the range (0, 60]");
        }
        if (!std::isfinite(duration_seconds) || duration_seconds <= 0.0 || duration_seconds > 3600.0) {
            throw std::invalid_argument("duration must be in the range (0, 3600] seconds");
        }
        if (fs::exists(output_path)) throw std::runtime_error("refusing to overwrite existing output");
        const auto frame_count = static_cast<long long>(std::llround(fps * duration_seconds));
        if (frame_count < 1 || frame_count > 1000000) {
            throw std::invalid_argument("requested output frame count is out of range");
        }

        cv::VideoCapture source(input_path.string());
        if (!source.isOpened()) throw std::runtime_error("OpenCV could not open input video");
        cv::Mat frame;
        for (long long index = 0; index <= frame_index; ++index) {
            if (!source.read(frame) || frame.empty()) {
                throw std::runtime_error("input ended before the requested frame index");
            }
        }

        if (!output_path.parent_path().empty()) {
            fs::create_directories(output_path.parent_path());
        }
        cv::VideoWriter writer(
            output_path.string(), 0, fps, frame.size());
        if (!writer.isOpened()) throw std::runtime_error("OpenCV could not open MJPEG video writer");
        for (long long index = 0; index < frame_count; ++index) writer.write(frame);
        writer.release();

        cv::VideoCapture verify(output_path.string());
        if (!verify.isOpened()) throw std::runtime_error("generated output cannot be reopened");
        cv::Mat first;
        if (!verify.read(first) || first.empty()) throw std::runtime_error("generated output has no frames");
        long long verified_frames = 1;
        double max_frame_difference = 0.0;
        double max_mean_frame_difference = 0.0;
        cv::Mat decoded;
        while (verify.read(decoded)) {
            if (decoded.empty() || decoded.size() != first.size() || decoded.type() != first.type()) {
                throw std::runtime_error("generated output frame shape changed after decode");
            }
            cv::Mat difference;
            cv::absdiff(first, decoded, difference);
            max_frame_difference = std::max(max_frame_difference, cv::norm(difference, cv::NORM_INF));
            const cv::Scalar channel_means = cv::mean(difference);
            const double mean_difference = (channel_means[0] + channel_means[1] + channel_means[2]) / 3.0;
            max_mean_frame_difference = std::max(max_mean_frame_difference, mean_difference);
            ++verified_frames;
        }
        if (verified_frames != frame_count) throw std::runtime_error("output frame count mismatch");
        if (max_frame_difference > 50.0 || max_mean_frame_difference > 0.1) {
            throw std::runtime_error("repeated decoded frame difference exceeded tolerance (max=" +
                                     std::to_string(max_frame_difference) + ", mean=" +
                                     std::to_string(max_mean_frame_difference) + ")");
        }
        std::cout << "created=" << output_path << " source_frame=" << frame_index
                  << " codec=uncompressed fps=" << fps << " frames=" << verified_frames
                  << " duration_seconds=" << duration_seconds
                  << " max_decoded_pixel_difference=" << max_frame_difference
                  << " max_decoded_mean_difference=" << max_mean_frame_difference
                  << " tolerances=(50,0.1)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "create_static_video: " << error.what() << '\n';
        return 2;
    }
}
