#include "video_source.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cctype>

#include <cstdlib>

#ifdef EDGE_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#endif

namespace edge {
namespace {

bool is_rtsp_uri(const std::string& source) {
    std::string prefix = source.substr(0, std::min<std::size_t>(source.size(), 8));
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return prefix.rfind("rtsp://", 0) == 0 || prefix.rfind("rtsps://", 0) == 0;
}

class OpenCvVideoSource final : public VideoSource {
public:
    bool open(const std::string& source, QueuePolicy, std::string& error) override {
        close();
        source_ = source;
        live_ = is_rtsp_uri(source);
        try {
            if (live_) {
                ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
                const std::vector<int> params{
                    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                    cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000,
                };
                if (!capture_.open(source, cv::CAP_FFMPEG, params)) {
                    error = "OpenCV FFmpeg could not open RTSP source";
                    return false;
                }
            } else if (!capture_.open(source, cv::CAP_FFMPEG)) {
                error = "OpenCV could not open local video";
                return false;
            }
            fps_ = live_ ? 0.0 : capture_.get(cv::CAP_PROP_FPS);
            if (!std::isfinite(fps_) || fps_ <= 0.0 || fps_ > 240.0) fps_ = 0.0;
            return true;
        } catch (const std::exception& exception) {
            error = exception.what();
            close();
            return false;
        }
    }

    FrameReadStatus read(DecodedFrame& frame, std::string& error) override {
        const auto start = SteadyClock::now();
        cv::Mat decoded;
        bool read_ok = false;
        try {
            read_ok = capture_.read(decoded) && !decoded.empty();
        } catch (const cv::Exception& exception) {
            error = exception.what();
        }
        if (!read_ok) return live_ ? FrameReadStatus::Error : FrameReadStatus::EndOfStream;
        if (decoded.type() != CV_8UC3) {
            error = "OpenCV source returned a non-BGR frame";
            return FrameReadStatus::Error;
        }
        frame = {};
        frame.width = decoded.cols;
        frame.height = decoded.rows;
        frame.pts_ms = capture_.get(cv::CAP_PROP_POS_MSEC);
        frame.bgr = decoded.clone();
        frame.source_copy_bytes = frame.bgr.total() * frame.bgr.elemSize();
        frame.decoded_at = SteadyClock::now();
        frame.decode_ms = std::chrono::duration<double, std::milli>(frame.decoded_at - start).count();
        return FrameReadStatus::Frame;
    }

    void close() noexcept override {
        try { capture_.release(); } catch (...) {}
    }

    bool is_live() const noexcept override { return live_; }
    double nominal_fps() const noexcept override { return fps_; }
    std::uint64_t dropped_frames() const noexcept override { return 0; }
    const char* name() const noexcept override { return "opencv-ffmpeg"; }

private:
    cv::VideoCapture capture_;
    std::string source_;
    bool live_ = false;
    double fps_ = 0.0;
};

#ifdef EDGE_HAS_GSTREAMER

class GstMppVideoSource final : public VideoSource {
public:
    ~GstMppVideoSource() override { close(); }

    bool open(const std::string& source, QueuePolicy, std::string& error) override {
        close();
        static std::once_flag gst_initialized;
        std::call_once(gst_initialized, [] { gst_init(nullptr, nullptr); });
        live_ = is_rtsp_uri(source);
        last_sample_at_ = SteadyClock::now();
        callback_error_.clear();

        pipeline_ = gst_pipeline_new("rk-edge-mpp-pipeline");
        if (pipeline_ == nullptr) return fail(error, "could not create GStreamer pipeline");

        auto make = [](const char* factory, const char* name) {
            return gst_element_factory_make(factory, name);
        };
        GstElement* input_caps_filter = make("capsfilter", "parsed-h264-caps");
        GstElement* decoder = make("mppvideodec", "mpp-decoder");
        GstElement* caps_filter = make("capsfilter", "nv12-caps");
        appsink_element_ = make("appsink", "frame-sink");
        if (input_caps_filter == nullptr || decoder == nullptr || caps_filter == nullptr ||
            appsink_element_ == nullptr) {
            if (input_caps_filter) gst_object_unref(input_caps_filter);
            if (decoder) gst_object_unref(decoder);
            if (caps_filter) gst_object_unref(caps_filter);
            if (appsink_element_) {
                gst_object_unref(appsink_element_);
                appsink_element_ = nullptr;
            }
            return fail(error, "required GStreamer element is unavailable (capsfilter, MPP decoder, appsink)");
        }

        const char* input_caps_text = live_
            ? "video/x-h264,parsed=true,alignment=au"
            : "video/x-h264,parsed=true,stream-format=avc,alignment=au";
        GstCaps* input_caps = gst_caps_from_string(input_caps_text);
        GstCaps* output_caps = gst_caps_from_string("video/x-raw,format=NV12");
        if (input_caps == nullptr || output_caps == nullptr) {
            if (input_caps != nullptr) gst_caps_unref(input_caps);
            if (output_caps != nullptr) gst_caps_unref(output_caps);
            gst_object_unref(input_caps_filter);
            gst_object_unref(decoder);
            gst_object_unref(caps_filter);
            gst_object_unref(appsink_element_);
            appsink_element_ = nullptr;
            return fail(error, "could not create H.264/NV12 caps");
        }
        g_object_set(input_caps_filter, "caps", input_caps, nullptr);
        g_object_set(caps_filter, "caps", output_caps, nullptr);
        g_object_set(appsink_element_,
                     "max-buffers", static_cast<guint>(1),
                     // Keep appsink bounded and let the observable Worker queue
                     // apply Latest/FIFO. Hidden appsink drops break offline
                     // comparisons and under-report the frame drop counter.
                     "drop", FALSE,
                     "sync", FALSE,
                     "wait-on-eos", FALSE,
                     "enable-last-sample", FALSE,
                     nullptr);
        gst_caps_unref(input_caps);
        gst_caps_unref(output_caps);
        gst_bin_add_many(GST_BIN(pipeline_), input_caps_filter, decoder, caps_filter,
                         appsink_element_, nullptr);
        input_caps_element_ = input_caps_filter;
        if (!gst_element_link_many(input_caps_filter, decoder, caps_filter, appsink_element_, nullptr)) {
            return fail(error, "could not link parsed H.264 caps, MPP decoder, NV12 caps, and appsink");
        }
        appsink_ = GST_APP_SINK(appsink_element_);

        if (live_) {
            source_element_ = make("rtspsrc", "rtsp-source");
            depayloader_ = make("rtph264depay", "h264-depayloader");
            if (source_element_ == nullptr || depayloader_ == nullptr) {
                if (source_element_) gst_object_unref(source_element_);
                if (depayloader_) gst_object_unref(depayloader_);
                source_element_ = nullptr;
                depayloader_ = nullptr;
                return fail(error, "rtspsrc or H.264 RTP depayloader is unavailable");
            }
            g_object_set(source_element_, "location", source.c_str(),
                         "latency", static_cast<guint>(150),
                         "drop-on-latency", TRUE, nullptr);
            gst_bin_add_many(GST_BIN(pipeline_), source_element_, depayloader_, nullptr);
            if (!gst_element_link(depayloader_, input_caps_filter)) {
                return fail(error, "could not link RTP depayloader to parsed H.264 caps");
            }
            g_signal_connect(source_element_, "pad-added",
                             G_CALLBACK(on_rtsp_pad_added), this);
        } else {
            source_element_ = make("filesrc", "file-source");
            demuxer_ = make("qtdemux", "mp4-demuxer");
            if (source_element_ == nullptr || demuxer_ == nullptr) {
                if (source_element_) gst_object_unref(source_element_);
                if (demuxer_) gst_object_unref(demuxer_);
                source_element_ = nullptr;
                demuxer_ = nullptr;
                return fail(error, "filesrc or MP4 qtdemux is unavailable");
            }
            g_object_set(source_element_, "location", source.c_str(), nullptr);
            gst_bin_add_many(GST_BIN(pipeline_), source_element_, demuxer_, nullptr);
            if (!gst_element_link(source_element_, demuxer_)) {
                return fail(error, "could not link filesrc to MP4 qtdemux");
            }
            if (!gst_element_link(demuxer_, input_caps_filter)) {
                // qtdemux exposes a dynamic H.264 source pad; the pad-added callback links it.
                GstPad* caps_sink = gst_element_get_static_pad(input_caps_filter, "sink");
                if (caps_sink == nullptr) return fail(error, "H.264 caps filter has no sink pad");
                gst_object_unref(caps_sink);
            }
            g_signal_connect(demuxer_, "pad-added", G_CALLBACK(on_demux_pad_added), this);
        }

        bus_ = gst_element_get_bus(pipeline_);
        if (bus_ == nullptr) return fail(error, "GStreamer pipeline has no bus");
        const auto state_result = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (state_result == GST_STATE_CHANGE_FAILURE) {
            error = pop_pipeline_error("GStreamer pipeline failed to enter PLAYING");
            close();
            return false;
        }
        return true;
    }

    FrameReadStatus read(DecodedFrame& frame, std::string& error) override {
        if (appsink_ == nullptr) {
            error = "GStreamer appsink is not open";
            return FrameReadStatus::Error;
        }
        const auto start = SteadyClock::now();
        GstSample* sample = gst_app_sink_try_pull_sample(appsink_, 200 * GST_MSECOND);
        const auto finish = SteadyClock::now();
        if (sample == nullptr) {
            {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (!callback_error_.empty()) {
                    error = callback_error_;
                    return FrameReadStatus::Error;
                }
            }
            GstMessage* message = gst_bus_pop_filtered(
                bus_, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            if (message != nullptr) {
                FrameReadStatus status = FrameReadStatus::Error;
                if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
                    error = "GStreamer source reached EOS";
                    status = live_ ? FrameReadStatus::Error : FrameReadStatus::EndOfStream;
                } else {
                    GError* gst_error = nullptr;
                    gchar* debug = nullptr;
                    gst_message_parse_error(message, &gst_error, &debug);
                    error = gst_error != nullptr ? gst_error->message : "GStreamer pipeline error";
                    if (debug != nullptr && *debug != '\0') error += "; " + std::string(debug);
                    if (gst_error != nullptr) g_error_free(gst_error);
                    g_free(debug);
                }
                gst_message_unref(message);
                return status;
            }
            if (gst_app_sink_is_eos(appsink_)) {
                error = "GStreamer source reached EOS";
                return live_ ? FrameReadStatus::Error : FrameReadStatus::EndOfStream;
            }
            if (live_ && finish - last_sample_at_ > std::chrono::seconds(5)) {
                error = "GStreamer RTSP source produced no frame for five seconds";
                return FrameReadStatus::Error;
            }
            return FrameReadStatus::Timeout;
        }

        last_sample_at_ = finish;
        GstCaps* caps = gst_sample_get_caps(sample);
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstVideoInfo info{};
        if (caps == nullptr || buffer == nullptr || !gst_video_info_from_caps(&info, caps) ||
            GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
            gst_sample_unref(sample);
            error = "GStreamer MPP appsink did not produce a valid NV12 frame";
            return FrameReadStatus::Error;
        }
        const int width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
        const int height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) {
            gst_sample_unref(sample);
            error = "NV12 frame has invalid or odd dimensions";
            return FrameReadStatus::Error;
        }
        GstVideoFrame mapped{};
        if (!gst_video_frame_map(&mapped, &info, buffer, GST_MAP_READ)) {
            gst_sample_unref(sample);
            error = "could not map MPP output as a GStreamer video frame";
            return FrameReadStatus::Error;
        }

        frame = {};
        frame.width = width;
        frame.height = height;
        frame.nv12.create(height * 3 / 2, width, CV_8UC1);
        const auto* source_y = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 0));
        const auto* source_uv = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 1));
        const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 0);
        const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 1);
        if (source_y == nullptr || source_uv == nullptr || y_stride < width || uv_stride < width) {
            gst_video_frame_unmap(&mapped);
            gst_sample_unref(sample);
            error = "GStreamer NV12 strides do not cover the visible frame width";
            return FrameReadStatus::Error;
        }
        for (int row = 0; row < height; ++row) {
            std::memcpy(frame.nv12.ptr(row), source_y + static_cast<std::ptrdiff_t>(row) * y_stride,
                        static_cast<std::size_t>(width));
        }
        for (int row = 0; row < height / 2; ++row) {
            std::memcpy(frame.nv12.ptr(height + row),
                        source_uv + static_cast<std::ptrdiff_t>(row) * uv_stride,
                        static_cast<std::size_t>(width));
        }
        gst_video_frame_unmap(&mapped);
        frame.source_copy_bytes = frame.nv12.total() * frame.nv12.elemSize();
        const GstClockTime pts = GST_BUFFER_PTS(buffer);
        if (GST_CLOCK_TIME_IS_VALID(pts)) {
            const GstSegment* segment = gst_sample_get_segment(sample);
            const GstClockTime running_time = segment != nullptr && segment->format == GST_FORMAT_TIME
                ? gst_segment_to_running_time(segment, GST_FORMAT_TIME, pts) : pts;
            if (GST_CLOCK_TIME_IS_VALID(running_time)) {
                frame.pts_ms = static_cast<double>(running_time) / static_cast<double>(GST_MSECOND);
            }
        }
        const int fps_n = GST_VIDEO_INFO_FPS_N(&info);
        const int fps_d = GST_VIDEO_INFO_FPS_D(&info);
        fps_ = fps_d > 0 ? static_cast<double>(fps_n) / fps_d : 0.0;
        frame.decoded_at = SteadyClock::now();
        frame.decode_ms = std::chrono::duration<double, std::milli>(frame.decoded_at - start).count();
        gst_sample_unref(sample);
        return FrameReadStatus::Frame;
    }

    void close() noexcept override {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
        }
        if (bus_ != nullptr) gst_object_unref(bus_);
        pipeline_ = nullptr;
        bus_ = nullptr;
        source_element_ = nullptr;
        demuxer_ = nullptr;
        input_caps_element_ = nullptr;
        depayloader_ = nullptr;
        appsink_element_ = nullptr;
        appsink_ = nullptr;
    }

    bool is_live() const noexcept override { return live_; }
    double nominal_fps() const noexcept override { return fps_; }
    std::uint64_t dropped_frames() const noexcept override {
        if (appsink_element_ == nullptr) return 0;
        GParamSpec* property = g_object_class_find_property(
            G_OBJECT_GET_CLASS(appsink_element_), "dropped");
        if (property == nullptr) return 0;
        guint64 dropped = 0;
        g_object_get(appsink_element_, "dropped", &dropped, nullptr);
        return dropped;
    }
    const char* name() const noexcept override { return "gstreamer-mpp"; }

private:
    bool fail(std::string& error, const std::string& message) {
        error = message;
        close();
        return false;
    }

    static void on_demux_pad_added(GstElement*, GstPad* pad, gpointer user_data) {
        auto* self = static_cast<GstMppVideoSource*>(user_data);
        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps == nullptr) caps = gst_pad_query_caps(pad, nullptr);
        const bool is_h264 = caps != nullptr && gst_caps_get_size(caps) > 0 &&
            g_strcmp0(gst_structure_get_name(gst_caps_get_structure(caps, 0)), "video/x-h264") == 0;
        if (caps != nullptr) gst_caps_unref(caps);
        if (is_h264) self->link_dynamic_pad(pad, self->input_caps_element_);
    }

    static void on_rtsp_pad_added(GstElement*, GstPad* pad, gpointer user_data) {
        auto* self = static_cast<GstMppVideoSource*>(user_data);
        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps == nullptr) caps = gst_pad_query_caps(pad, nullptr);
        bool is_h264_video = false;
        if (caps != nullptr && gst_caps_get_size(caps) > 0) {
            const GstStructure* structure = gst_caps_get_structure(caps, 0);
            const gchar* media = gst_structure_get_string(structure, "media");
            const gchar* encoding = gst_structure_get_string(structure, "encoding-name");
            is_h264_video = g_strcmp0(gst_structure_get_name(structure), "application/x-rtp") == 0 &&
                media != nullptr && g_ascii_strcasecmp(media, "video") == 0 &&
                encoding != nullptr && g_ascii_strcasecmp(encoding, "H264") == 0;
        }
        if (caps != nullptr) gst_caps_unref(caps);
        if (is_h264_video) self->link_dynamic_pad(pad, self->depayloader_);
    }

    void link_dynamic_pad(GstPad* source_pad, GstElement* destination) {
        if (destination == nullptr) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            callback_error_ = "GStreamer dynamic pad destination is missing";
            return;
        }
        GstPad* sink_pad = gst_element_get_static_pad(destination, "sink");
        if (sink_pad == nullptr) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            callback_error_ = "GStreamer dynamic pad target has no sink pad";
            return;
        }
        if (!gst_pad_is_linked(sink_pad)) {
            const GstPadLinkReturn status = gst_pad_link(source_pad, sink_pad);
            if (status != GST_PAD_LINK_OK) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                callback_error_ = "GStreamer could not link H.264 video pad: " +
                                  std::to_string(static_cast<int>(status));
            }
        }
        gst_object_unref(sink_pad);
    }

    std::string pop_pipeline_error(const std::string& fallback) {
        GstMessage* message = bus_ == nullptr ? nullptr :
            gst_bus_pop_filtered(bus_, GST_MESSAGE_ERROR);
        if (message == nullptr) return fallback;
        GError* gst_error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &gst_error, &debug);
        std::string result = gst_error != nullptr ? gst_error->message : fallback;
        if (debug != nullptr && *debug != '\0') result += "; " + std::string(debug);
        if (gst_error != nullptr) g_error_free(gst_error);
        g_free(debug);
        gst_message_unref(message);
        return result;
    }

    GstElement* pipeline_ = nullptr;
    GstElement* source_element_ = nullptr;
    GstElement* demuxer_ = nullptr;
    GstElement* input_caps_element_ = nullptr;
    GstElement* depayloader_ = nullptr;
    GstElement* appsink_element_ = nullptr;
    GstAppSink* appsink_ = nullptr;
    GstBus* bus_ = nullptr;
    std::string callback_error_;
    std::mutex callback_mutex_;
    bool live_ = false;
    double fps_ = 0.0;
    SteadyClock::time_point last_sample_at_{};
};

#endif

}  // namespace

std::unique_ptr<VideoSource> make_video_source(VideoBackend backend) {
    switch (backend) {
        case VideoBackend::OpenCvFfmpeg:
            return std::make_unique<OpenCvVideoSource>();
        case VideoBackend::GStreamerMpp:
#ifdef EDGE_HAS_GSTREAMER
            return std::make_unique<GstMppVideoSource>();
#else
            throw std::runtime_error("GStreamer support was not built; configure with EDGE_ENABLE_GSTREAMER=ON");
#endif
    }
    throw std::invalid_argument("unknown video backend");
}

VideoBackend parse_video_backend(const std::string& value) {
    if (value == "opencv") return VideoBackend::OpenCvFfmpeg;
    if (value == "gstreamer-mpp") return VideoBackend::GStreamerMpp;
    throw std::invalid_argument("video backend must be opencv or gstreamer-mpp");
}

const char* video_backend_name(VideoBackend backend) noexcept {
    switch (backend) {
        case VideoBackend::OpenCvFfmpeg: return "opencv-ffmpeg";
        case VideoBackend::GStreamerMpp: return "gstreamer-mpp";
    }
    return "unknown";
}

cv::Mat nv12_to_bgr(const cv::Mat& nv12, int width, int height) {
    if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0 ||
        nv12.type() != CV_8UC1 || nv12.cols != width || nv12.rows != height * 3 / 2 ||
        !nv12.isContinuous()) {
        throw std::invalid_argument("expected contiguous even-sized NV12 with shape (height*3/2,width)");
    }
    const cv::Mat y(height, width, CV_8UC1, nv12.data, static_cast<std::size_t>(width));
    const cv::Mat uv(height / 2, width / 2, CV_8UC2,
                     nv12.data + static_cast<std::size_t>(height) * width,
                     static_cast<std::size_t>(width));
    cv::Mat bgr;
    cv::cvtColorTwoPlane(y, uv, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
}

}  // namespace edge
