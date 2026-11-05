// Exercise librga resize and color conversion on a CPU-decoded BGR frame.
// Build on the RK3576 with:
//   g++ -O2 -std=c++17 scripts/benchmark_rga.cpp -o rga-bench $(pkg-config --cflags --libs opencv4) -lrga

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <rga/im2d.hpp>
#include <rga/rga.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string video;
    int repetitions = 200;
    int warmup = 10;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "benchmark_rga --video FILE [--repetitions N] [--warmup N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--video") options.video = value;
        else if (key == "--repetitions") options.repetitions = std::stoi(value);
        else if (key == "--warmup") options.warmup = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.video.empty()) throw std::runtime_error("--video is required");
    if (options.repetitions < 1 || options.warmup < 0) {
        throw std::runtime_error("repetitions must be >= 1 and warmup >= 0");
    }
    return options;
}

void require_rga(IM_STATUS status, const std::string& operation) {
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        const char* error = imStrError_t(status);
        throw std::runtime_error(operation + " failed: " +
                                 (error == nullptr ? std::to_string(status) : error));
    }
}

template <typename Function>
double measure_ms_per_op(int repetitions, Function&& function) {
    const auto start = Clock::now();
    for (int i = 0; i < repetitions; ++i) function();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return elapsed_ms / static_cast<double>(repetitions);
}

cv::Mat max_abs_difference(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat difference;
    cv::absdiff(first, second, difference);
    return difference;
}

double mean_abs_difference(const cv::Mat& difference) {
    return cv::mean(difference)[0] + cv::mean(difference)[1] + cv::mean(difference)[2];
}

double max_abs_value(const cv::Mat& difference) {
    std::vector<cv::Mat> channels;
    cv::split(difference, channels);
    double maximum = 0.0;
    for (const auto& channel : channels) {
        double channel_max = 0.0;
        cv::minMaxLoc(channel, nullptr, &channel_max);
        maximum = std::max(maximum, channel_max);
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        cv::VideoCapture capture(options.video, cv::CAP_FFMPEG);
        if (!capture.isOpened()) throw std::runtime_error("OpenCV FFmpeg could not open input video");
        cv::Mat input;
        if (!capture.read(input) || input.empty() || input.type() != CV_8UC3) {
            throw std::runtime_error("could not decode a BGR input frame");
        }
        capture.release();

        const int output_width = 640;
        const int output_height = 480;
        cv::Mat cpu_resized(output_height, output_width, CV_8UC3);
        cv::Mat cpu_rgb(output_height, output_width, CV_8UC3);
        cv::Mat rga_resized(output_height, output_width, CV_8UC3);
        cv::Mat rga_rgb(output_height, output_width, CV_8UC3);
        cv::Mat rga_combined_rgb(output_height, output_width, CV_8UC3);

        const rga_buffer_t source = wrapbuffer_virtualaddr(
            input.data, input.cols, input.rows, RK_FORMAT_BGR_888, input.cols, input.rows);
        const rga_buffer_t resized = wrapbuffer_virtualaddr(
            rga_resized.data, output_width, output_height, RK_FORMAT_BGR_888,
            output_width, output_height);
        const rga_buffer_t rgb_source = wrapbuffer_virtualaddr(
            rga_resized.data, output_width, output_height, RK_FORMAT_BGR_888,
            output_width, output_height);
        const rga_buffer_t rgb_destination = wrapbuffer_virtualaddr(
            rga_rgb.data, output_width, output_height, RK_FORMAT_RGB_888,
            output_width, output_height);
        const rga_buffer_t combined_destination = wrapbuffer_virtualaddr(
            rga_combined_rgb.data, output_width, output_height, RK_FORMAT_RGB_888,
            output_width, output_height);
        const im_rect source_rect{0, 0, input.cols, input.rows};
        const im_rect destination_rect{0, 0, output_width, output_height};
        const im_rect empty_rect{};
        const rga_buffer_t empty_buffer{};
        im_opt_t combined_options{};
        combined_options.interp = INTER_LINEAR;

        require_rga(imresize(source, resized, 0, 0, INTER_LINEAR, 1), "RGA resize preflight");
        require_rga(imcvtcolor(rgb_source, rgb_destination, RK_FORMAT_BGR_888,
                               RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1),
                    "RGA color-conversion preflight");

        bool combined_supported = true;
        std::string combined_error;
        const auto combined_status = improcess(source, combined_destination, empty_buffer,
                                               source_rect, destination_rect, empty_rect,
                                               -1, nullptr, &combined_options, IM_SYNC);
        if (combined_status != IM_STATUS_SUCCESS && combined_status != IM_STATUS_NOERROR) {
            combined_supported = false;
            const char* error = imStrError_t(combined_status);
            combined_error = error == nullptr ? std::to_string(combined_status) : error;
        }

        for (int i = 0; i < options.warmup; ++i) {
            cv::resize(input, cpu_resized, cv::Size(output_width, output_height),
                       0.0, 0.0, cv::INTER_LINEAR);
            cv::cvtColor(cpu_resized, cpu_rgb, cv::COLOR_BGR2RGB);
            require_rga(imresize(source, resized, 0, 0, INTER_LINEAR, 1), "RGA resize warmup");
            require_rga(imcvtcolor(rgb_source, rgb_destination, RK_FORMAT_BGR_888,
                                   RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1),
                        "RGA color-conversion warmup");
            if (combined_supported) {
                require_rga(improcess(source, combined_destination, empty_buffer,
                                      source_rect, destination_rect, empty_rect,
                                      -1, nullptr, &combined_options, IM_SYNC),
                            "RGA combined-operation warmup");
            }
        }

        const double cpu_combined_ms = measure_ms_per_op(options.repetitions, [&] {
            cv::resize(input, cpu_resized, cv::Size(output_width, output_height),
                       0.0, 0.0, cv::INTER_LINEAR);
            cv::cvtColor(cpu_resized, cpu_rgb, cv::COLOR_BGR2RGB);
        });
        const double rga_resize_ms = measure_ms_per_op(options.repetitions, [&] {
            require_rga(imresize(source, resized, 0, 0, INTER_LINEAR, 1), "RGA resize");
        });
        const double rga_color_ms = measure_ms_per_op(options.repetitions, [&] {
            require_rga(imcvtcolor(rgb_source, rgb_destination, RK_FORMAT_BGR_888,
                                   RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1),
                        "RGA color conversion");
        });
        const double rga_combined_ms = measure_ms_per_op(options.repetitions, [&] {
            require_rga(imresize(source, resized, 0, 0, INTER_LINEAR, 1), "RGA combined resize");
            require_rga(imcvtcolor(rgb_source, rgb_destination, RK_FORMAT_BGR_888,
                                   RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1),
                        "RGA combined color conversion");
        });
        double rga_single_process_ms = -1.0;
        if (combined_supported) {
            rga_single_process_ms = measure_ms_per_op(options.repetitions, [&] {
                require_rga(improcess(source, combined_destination, empty_buffer,
                                      source_rect, destination_rect, empty_rect,
                                      -1, nullptr, &combined_options, IM_SYNC),
                            "RGA single combined operation");
            });
        }

        cv::resize(input, cpu_resized, cv::Size(output_width, output_height),
                   0.0, 0.0, cv::INTER_LINEAR);
        cv::cvtColor(cpu_resized, cpu_rgb, cv::COLOR_BGR2RGB);
        require_rga(imresize(source, resized, 0, 0, INTER_LINEAR, 1), "RGA resize comparison");
        require_rga(imcvtcolor(rgb_source, rgb_destination, RK_FORMAT_BGR_888,
                               RK_FORMAT_RGB_888, IM_COLOR_SPACE_DEFAULT, 1),
                    "RGA color-conversion comparison");
        const cv::Mat resize_difference = max_abs_difference(cpu_resized, rga_resized);
        const cv::Mat color_difference = max_abs_difference(cpu_rgb, rga_rgb);
        const cv::Mat combined_difference = max_abs_difference(cpu_rgb, rga_combined_rgb);

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"operation\":\"cpu_resize_plus_color\",\"ms_per_op\":"
                  << cpu_combined_ms << "}\n"
                  << "{\"operation\":\"rga_resize\",\"ms_per_op\":"
                  << rga_resize_ms << "}\n"
                  << "{\"operation\":\"rga_color_conversion\",\"ms_per_op\":"
                  << rga_color_ms << "}\n"
                  << "{\"operation\":\"rga_resize_plus_color\",\"ms_per_op\":"
                  << rga_combined_ms << "}\n"
                  << "{\"operation\":\"rga_single_combined_operation\",\"supported\":"
                  << (combined_supported ? "true" : "false")
                  << ",\"ms_per_op\":";
        if (rga_single_process_ms >= 0.0) std::cout << rga_single_process_ms;
        else std::cout << "null";
        std::cout << ",\"error\":\"" << combined_error << "\"}\n"
                  << "{\"type\":\"comparison\",\"input_width\":" << input.cols
                  << ",\"input_height\":" << input.rows
                  << ",\"output_width\":" << output_width
                  << ",\"output_height\":" << output_height
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"warmup\":" << options.warmup
                  << ",\"resize_mean_absolute_difference\":"
                  << mean_abs_difference(resize_difference) / 3.0
                  << ",\"resize_max_absolute_difference\":"
                  << max_abs_value(resize_difference)
                  << ",\"color_mean_absolute_difference\":"
                  << mean_abs_difference(color_difference) / 3.0
                  << ",\"color_max_absolute_difference\":"
                  << max_abs_value(color_difference)
                  << ",\"combined_rgb_supported\":"
                  << (combined_supported ? "true" : "false")
                  << ",\"combined_rgb_mean_absolute_difference\":";
        if (combined_supported) std::cout << mean_abs_difference(combined_difference) / 3.0;
        else std::cout << "null";
        std::cout << ",\"combined_rgb_max_absolute_difference\":";
        if (combined_supported) std::cout << max_abs_value(combined_difference);
        else std::cout << "null";
        std::cout
                  << ",\"input_memory\":\"CPU virtual address\","
                     "\"output_memory\":\"CPU virtual address\","
                     "\"note\":\"Operation-level RGA call check only; not zero-copy or full-pipeline proof.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RGA benchmark error: " << error.what() << '\n';
        return 2;
    }
}
