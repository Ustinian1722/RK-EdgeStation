// Compare the worker's OpenCV letterbox/RGB preprocessing with one RGA job.
// Build on RK3576: g++ -O2 -std=c++17 scripts/benchmark_rga_letterbox.cpp -o rga-letterbox-bench $(pkg-config --cflags --libs opencv4) -lrga

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <rga/im2d.hpp>
#include <rga/rga.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string video;
    std::string dump_inputs_dir;
    int width = 640;
    int height = 640;
    int frame_index = 0;
    int repetitions = 500;
    int warmup = 30;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "benchmark_rga_letterbox --video FILE [--width N] [--height N] "
                         "[--frame-index N] [--repetitions N] [--warmup N] [--dump-inputs DIR]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--video") options.video = value;
        else if (key == "--dump-inputs") options.dump_inputs_dir = value;
        else if (key == "--width") options.width = std::stoi(value);
        else if (key == "--height") options.height = std::stoi(value);
        else if (key == "--frame-index") options.frame_index = std::stoi(value);
        else if (key == "--repetitions") options.repetitions = std::stoi(value);
        else if (key == "--warmup") options.warmup = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.video.empty()) throw std::runtime_error("--video is required");
    if (options.width < 1 || options.height < 1 || options.frame_index < 0 ||
        options.repetitions < 1 || options.warmup < 0) {
        throw std::runtime_error("dimensions/repetitions must be positive and warmup nonnegative");
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
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() /
           static_cast<double>(repetitions);
}

double mean_abs_difference(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat difference;
    cv::absdiff(first, second, difference);
    const cv::Scalar mean = cv::mean(difference);
    return (mean[0] + mean[1] + mean[2]) / 3.0;
}

double max_abs_difference(const cv::Mat& first, const cv::Mat& second) {
    cv::Mat difference;
    cv::absdiff(first, second, difference);
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
        cv::Mat input;
        if (!capture.isOpened()) throw std::runtime_error("OpenCV FFmpeg could not open the input video");
        if (options.frame_index > 0 &&
            !capture.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(options.frame_index))) {
            throw std::runtime_error("could not seek to the requested frame index");
        }
        if (!capture.isOpened() || !capture.read(input) || input.empty() || input.type() != CV_8UC3) {
            throw std::runtime_error("OpenCV FFmpeg could not decode a BGR frame");
        }
        capture.release();

        const float scale = std::min(static_cast<float>(options.width) / input.cols,
                                     static_cast<float>(options.height) / input.rows);
        const int resized_width = std::max(1, static_cast<int>(std::round(input.cols * scale)));
        const int resized_height = std::max(1, static_cast<int>(std::round(input.rows * scale)));
        const int pad_x = (options.width - resized_width) / 2;
        const int pad_y = (options.height - resized_height) / 2;

        cv::Mat cpu_resized(resized_height, resized_width, CV_8UC3);
        cv::Mat cpu_padded(options.height, options.width, CV_8UC3);
        cv::Mat cpu_rgb(options.height, options.width, CV_8UC3);
        cv::Mat rga_rgb(options.height, options.width, CV_8UC3);

        const rga_buffer_t source = wrapbuffer_virtualaddr(
            input.data, input.cols, input.rows, RK_FORMAT_BGR_888, input.cols, input.rows);
        const rga_buffer_t destination = wrapbuffer_virtualaddr(
            rga_rgb.data, options.width, options.height, RK_FORMAT_RGB_888,
            options.width, options.height);
        const im_rect source_rect{0, 0, input.cols, input.rows};
        const im_rect destination_rect{pad_x, pad_y, resized_width, resized_height};
        const im_rect empty_rect{};
        const rga_buffer_t empty_buffer{};
        im_opt_t operation_options{};
        operation_options.interp = INTER_LINEAR;

        rga_rgb.setTo(cv::Scalar(114, 114, 114));
        require_rga(improcess(source, destination, empty_buffer, source_rect, destination_rect,
                              empty_rect, -1, nullptr, &operation_options, IM_SYNC),
                    "RGA letterbox preflight");

        auto run_cpu = [&] {
            cv::resize(input, cpu_resized, cv::Size(resized_width, resized_height),
                       0.0, 0.0, cv::INTER_LINEAR);
            cpu_padded.setTo(cv::Scalar(114, 114, 114));
            cpu_resized.copyTo(cpu_padded(cv::Rect(pad_x, pad_y, resized_width, resized_height)));
            cv::cvtColor(cpu_padded, cpu_rgb, cv::COLOR_BGR2RGB);
        };
        auto run_rga = [&] {
            rga_rgb.setTo(cv::Scalar(114, 114, 114));
            require_rga(improcess(source, destination, empty_buffer, source_rect, destination_rect,
                                  empty_rect, -1, nullptr, &operation_options, IM_SYNC),
                        "RGA letterbox");
        };

        for (int i = 0; i < options.warmup; ++i) {
            run_cpu();
            run_rga();
        }
        const double cpu_ms = measure_ms_per_op(options.repetitions, run_cpu);
        const double rga_ms = measure_ms_per_op(options.repetitions, run_rga);
        run_cpu();
        run_rga();

        if (!options.dump_inputs_dir.empty()) {
            std::filesystem::create_directories(options.dump_inputs_dir);
            for (const auto& entry : std::vector<std::pair<std::string, const cv::Mat*>>{
                     {"cpu-rgb-u8.bin", &cpu_rgb}, {"rga-rgb-u8.bin", &rga_rgb}}) {
                std::ofstream output(std::filesystem::path(options.dump_inputs_dir) / entry.first,
                                     std::ios::binary);
                const std::size_t bytes = entry.second->total() * entry.second->elemSize();
                output.write(reinterpret_cast<const char*>(entry.second->data),
                             static_cast<std::streamsize>(bytes));
                if (!output) throw std::runtime_error("failed to write preprocessed input: " + entry.first);
            }
        }

        std::cout << std::fixed << std::setprecision(6)
                  << "{\"type\":\"rga_letterbox_comparison\",\"input_width\":" << input.cols
                  << ",\"input_height\":" << input.rows
                  << ",\"model_width\":" << options.width
                  << ",\"model_height\":" << options.height
                  << ",\"frame_index\":" << options.frame_index
                  << ",\"resized_width\":" << resized_width
                  << ",\"resized_height\":" << resized_height
                  << ",\"pad_x\":" << pad_x << ",\"pad_y\":" << pad_y
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"warmup\":" << options.warmup
                  << ",\"cpu_worker_preprocess_ms\":" << cpu_ms
                  << ",\"rga_preprocess_with_cpu_canvas_fill_ms\":" << rga_ms
                  << ",\"rga_over_cpu_speed_ratio\":" << (cpu_ms / rga_ms)
                  << ",\"mean_abs_pixel_difference\":"
                  << mean_abs_difference(cpu_rgb, rga_rgb)
                  << ",\"max_abs_pixel_difference\":"
                  << max_abs_difference(cpu_rgb, rga_rgb)
                  << ",\"saved_inputs\":\""
                  << (options.dump_inputs_dir.empty() ? "" : options.dump_inputs_dir)
                  << "\""
                  << ",\"input_memory\":\"CPU virtual address\",\"output_memory\":\"CPU virtual address\""
                  << ",\"note\":\"Preprocessing microbenchmark only; includes CPU fill of 640x640 padding canvas, uses synchronous RGA, and does not establish zero-copy or end-to-end gain.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RGA letterbox benchmark error: " << error.what() << '\n';
        return 2;
    }
}
