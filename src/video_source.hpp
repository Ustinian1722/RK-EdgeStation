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
    // OpenCV frames own BGR. GStreamer frames normally own a packed NV12 copy; the
    // experimental retained-DMA-BUF mode instead holds the GstSample through RGA/NPU use.
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
    // Optional ref-counted GstSample lifetime for the experimental DMA-BUF path.
    // fd is borrowed from that sample and must never be closed by DecodedFrame.
    int dma_buf_fd = -1;
    std::uint64_t dma_buf_size = 0;
    std::uint64_t dma_buf_offset = 0;
    int dma_buf_y_stride = 0;
    int dma_buf_uv_stride = 0;
    std::uint64_t dma_buf_uv_offset = 0;
    std::shared_ptr<void> dma_buf_lifetime;
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

std::unique_ptr<VideoSource> make_video_source(VideoBackend backend, bool retain_dmabuf = false);
VideoBackend parse_video_backend(const std::string& value);
const char* video_backend_name(VideoBackend backend) noexcept;
cv::Mat nv12_to_bgr(const cv::Mat& nv12, int width, int height);
cv::Mat frame_to_bgr(const DecodedFrame& frame);

}  // namespace edge
