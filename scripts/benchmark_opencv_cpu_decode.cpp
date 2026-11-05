// Decode a video through OpenCV's FFmpeg backend without inference or rendering.
// Build on the RK3576 with:
//   g++ -O2 -std=c++17 scripts/benchmark_opencv_cpu_decode.cpp -o decode-bench $(pkg-config --cflags --libs opencv4)

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string video;
    int expected_frames = 0;
    int warmup_frames = 30;
    int repetitions = 3;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "benchmark_opencv_cpu_decode --video FILE [--expected-frames N] "
                         "[--warmup-frames N] [--repetitions N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--video") options.video = value;
        else if (key == "--expected-frames") options.expected_frames = std::stoi(value);
        else if (key == "--warmup-frames") options.warmup_frames = std::stoi(value);
        else if (key == "--repetitions") options.repetitions = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.video.empty()) throw std::runtime_error("--video is required");
    if (options.expected_frames < 0 || options.warmup_frames < 0 || options.repetitions < 1) {
        throw std::runtime_error("expected-frames/warmup-frames must be nonnegative and repetitions >= 1");
    }
    return options;
}

struct CpuTimes {
    double user_seconds = 0.0;
    double system_seconds = 0.0;
};

CpuTimes process_cpu_times() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
    return {usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6,
            usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6};
}

long process_rss_kb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream input(line.substr(6));
            long value = 0;
            input >> value;
            return value;
        }
    }
    return -1;
}

void warmup(const std::string& path, int frame_count) {
    if (frame_count == 0) return;
    cv::VideoCapture capture(path, cv::CAP_FFMPEG);
    if (!capture.isOpened()) throw std::runtime_error("OpenCV FFmpeg could not open warmup video");
    cv::Mat frame;
    for (int i = 0; i < frame_count; ++i) {
        if (!capture.read(frame)) break;
    }
}

struct Result {
    int run = 0;
    int frames = 0;
    double elapsed_seconds = 0.0;
    double throughput_fps = 0.0;
    double process_cpu_seconds = 0.0;
    double process_cpu_core_percent = 0.0;
    double first_frame_latency_ms = 0.0;
    int width = 0;
    int height = 0;
    int channels = 0;
    double source_fps = 0.0;
    std::string backend;
    long rss_kb_after = -1;
};

Result run_once(const Options& options, int run_number) {
    warmup(options.video, options.warmup_frames);

    Result result;
    result.run = run_number;
    const auto cpu_before = process_cpu_times();
    const auto start = Clock::now();
    cv::VideoCapture capture(options.video, cv::CAP_FFMPEG);
    if (!capture.isOpened()) throw std::runtime_error("OpenCV FFmpeg could not open video");
    result.backend = capture.getBackendName();
    result.source_fps = capture.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    bool first_frame = true;
    while (capture.read(frame)) {
        if (first_frame) {
            result.first_frame_latency_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            result.width = frame.cols;
            result.height = frame.rows;
            result.channels = frame.channels();
            first_frame = false;
        }
        ++result.frames;
    }
    const auto end = Clock::now();
    capture.release();
    const auto cpu_after = process_cpu_times();

    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    result.throughput_fps = result.frames / result.elapsed_seconds;
    result.process_cpu_seconds = (cpu_after.user_seconds - cpu_before.user_seconds) +
                                 (cpu_after.system_seconds - cpu_before.system_seconds);
    result.process_cpu_core_percent = result.process_cpu_seconds / result.elapsed_seconds * 100.0;
    result.rss_kb_after = process_rss_kb();
    if (options.expected_frames > 0 && result.frames != options.expected_frames) {
        throw std::runtime_error("OpenCV decoded " + std::to_string(result.frames) +
                                 " frames; expected " + std::to_string(options.expected_frames));
    }
    return result;
}

void print_result(const Result& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\"run\":" << result.run
              << ",\"decoded_frames\":" << result.frames
              << ",\"elapsed_seconds\":" << result.elapsed_seconds
              << ",\"throughput_fps\":" << result.throughput_fps
              << ",\"process_cpu_seconds\":" << result.process_cpu_seconds
              << ",\"process_cpu_core_percent\":" << result.process_cpu_core_percent
              << ",\"first_frame_latency_ms\":" << result.first_frame_latency_ms
              << ",\"width\":" << result.width
              << ",\"height\":" << result.height
              << ",\"channels\":" << result.channels
              << ",\"source_fps\":" << result.source_fps
              << ",\"backend\":\"" << result.backend << "\""
              << ",\"process_rss_kb_after\":" << result.rss_kb_after << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::vector<double> throughputs;
        for (int run = 1; run <= options.repetitions; ++run) {
            const Result result = run_once(options, run);
            print_result(result);
            throughputs.push_back(result.throughput_fps);
        }
        std::sort(throughputs.begin(), throughputs.end());
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"type\":\"summary\",\"median_throughput_fps\":"
                  << throughputs[throughputs.size() / 2] << ",\"repetitions\":"
                  << options.repetitions << ",\"warmup_frames_each_run\":"
                  << options.warmup_frames << ",\"output_format\":\"BGR\","
                     "\"interpretation\":\"OpenCV FFmpeg software decode to BGR; no inference or rendering.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decode benchmark error: " << error.what() << '\n';
        return 2;
    }
}
