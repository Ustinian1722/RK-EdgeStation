#pragma once

#include "bounded_queue.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace edge {

using SteadyClock = std::chrono::steady_clock;

enum class VideoBackend { OpenCvFfmpeg, GStreamerMpp };
enum class FrameReadStatus { Frame, Timeout, EndOfStream, Error };

struct DecodedFrame {
    // OpenCV frames own a BGR image. GStreamer frames own a canonical packed NV12 copy;
    // conversion to BGR is deferred until CPU preprocessing or event evidence needs it.
    cv::Mat bgr;
    cv::Mat nv12;
    int width = 0;
    int height = 0;
    std::uint64_t sequence = 0;
    std::uint64_t stream_epoch = 0;
    double pts_ms = 0.0;
    double decode_ms = 0.0;
    std::uint64_t source_copy_bytes = 0;
    SteadyClock::time_point decoded_at{};
};

class VideoSource {
public:
    virtual ~VideoSource() = default;
    virtual bool open(const std::string& source, QueuePolicy policy, std::string& error) = 0;
    virtual FrameReadStatus read(DecodedFrame& frame, std::string& error) = 0;
    virtual void close() noexcept = 0;
    virtual bool is_live() const noexcept = 0;
    virtual double nominal_fps() const noexcept = 0;
    virtual std::uint64_t dropped_frames() const noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

std::unique_ptr<VideoSource> make_video_source(VideoBackend backend);
VideoBackend parse_video_backend(const std::string& value);
const char* video_backend_name(VideoBackend backend) noexcept;
cv::Mat nv12_to_bgr(const cv::Mat& nv12, int width, int height);

}  // namespace edge
