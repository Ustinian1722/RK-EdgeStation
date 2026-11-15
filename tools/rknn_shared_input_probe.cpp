#include "preprocess.hpp"
#include "video_source.hpp"
#include "yolox_ppe_postprocess.hpp"

#include <rknn_api.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <rga/im2d.hpp>
#include <rga/rga.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Model {
    rknn_context context = 0;
    rknn_input_output_num io{};
    rknn_tensor_attr input{};
    rknn_tensor_attr native_input{};
    std::vector<rknn_tensor_attr> outputs;
    rknn_tensor_mem* input_mem = nullptr;
    rknn_app_context_t postprocess{};

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept
        : context(other.context), io(other.io), input(other.input), native_input(other.native_input),
          outputs(std::move(other.outputs)), input_mem(other.input_mem), postprocess(other.postprocess) {
        other.context = 0;
        other.input_mem = nullptr;
        postprocess.output_attrs = outputs.data();
    }
    Model& operator=(Model&&) = delete;

    ~Model() {
        if (input_mem != nullptr && context != 0) rknn_destroy_mem(context, input_mem);
        if (context != 0) rknn_destroy(context);
    }
};

struct Timing {
    double preprocess_ms = 0.0;
    double sync_ms = 0.0;
    double input_ms = 0.0;
    double run_ms = 0.0;
    double fetch_ms = 0.0;
    double cpu_user_ms = 0.0;
    double cpu_system_ms = 0.0;
    long max_rss_kb = 0;
    std::vector<std::vector<std::uint8_t>> outputs;
    std::vector<object_detect_result> detections;
    std::uint64_t input_hash = 0;
    bool rga_dma_import_ok = false;
    int rga_dma_status = 0;
    std::string rga_dma_error;
};

struct LinkContext {
    GstElement* h264_filter = nullptr;
    std::string error;
};

struct MppInput {
    GstElement* pipeline = nullptr;
    GstElement* sink_element = nullptr;
    GstBus* bus = nullptr;
    GstSample* sample = nullptr;
    bool eos = false;

    MppInput() = default;
    MppInput(const MppInput&) = delete;
    MppInput& operator=(const MppInput&) = delete;
    MppInput(MppInput&& other) noexcept
        : pipeline(other.pipeline), sink_element(other.sink_element), bus(other.bus),
          sample(other.sample), eos(other.eos) {
        other.pipeline = nullptr;
        other.sink_element = nullptr;
        other.bus = nullptr;
        other.sample = nullptr;
    }
    ~MppInput() {
        if (sample != nullptr) gst_sample_unref(sample);
        if (pipeline != nullptr) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
        }
        if (bus != nullptr) gst_object_unref(bus);
    }
};

void on_demux_pad_added(GstElement*, GstPad* source_pad, gpointer user_data) {
    auto* context = static_cast<LinkContext*>(user_data);
    GstCaps* caps = gst_pad_get_current_caps(source_pad);
    if (caps == nullptr) caps = gst_pad_query_caps(source_pad, nullptr);
    const bool is_h264 = caps != nullptr && gst_caps_get_size(caps) > 0 &&
        g_strcmp0(gst_structure_get_name(gst_caps_get_structure(caps, 0)), "video/x-h264") == 0;
    if (caps != nullptr) gst_caps_unref(caps);
    if (!is_h264) return;
    GstPad* sink_pad = gst_element_get_static_pad(context->h264_filter, "sink");
    if (sink_pad == nullptr) {
        context->error = "H.264 caps filter has no sink pad";
        return;
    }
    if (!gst_pad_is_linked(sink_pad)) {
        const GstPadLinkReturn status = gst_pad_link(source_pad, sink_pad);
        if (status != GST_PAD_LINK_OK) context->error = "qtdemux H.264 pad link failed: " +
            std::to_string(static_cast<int>(status));
    }
    gst_object_unref(sink_pad);
}

MppInput open_mpp_input(const std::string& path) {
    static bool initialized = false;
    if (!initialized) {
        gst_init(nullptr, nullptr);
        initialized = true;
    }
    MppInput input;
    input.pipeline = gst_pipeline_new("rknn-shared-input-probe");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* demuxer = gst_element_factory_make("qtdemux", "mp4-demuxer");
    GstElement* h264_filter = gst_element_factory_make("capsfilter", "h264-filter");
    GstElement* decoder = gst_element_factory_make("mppvideodec", "mpp-decoder");
    GstElement* nv12_filter = gst_element_factory_make("capsfilter", "nv12-filter");
    input.sink_element = gst_element_factory_make("appsink", "frame-sink");
    if (input.pipeline == nullptr || source == nullptr || demuxer == nullptr || h264_filter == nullptr ||
        decoder == nullptr || nv12_filter == nullptr || input.sink_element == nullptr) {
        throw std::runtime_error("GStreamer/MPP probe element is unavailable");
    }
    g_object_set(source, "location", path.c_str(), nullptr);
    g_object_set(decoder, "dma-feature", FALSE, nullptr);
    g_object_set(input.sink_element, "max-buffers", static_cast<guint>(1), "drop", FALSE,
                 "sync", FALSE, "wait-on-eos", FALSE, "enable-last-sample", FALSE, nullptr);
    GstCaps* h264_caps = gst_caps_from_string(
        "video/x-h264,parsed=true,stream-format=avc,alignment=au");
    GstCaps* nv12_caps = gst_caps_from_string("video/x-raw,format=NV12");
    g_object_set(h264_filter, "caps", h264_caps, nullptr);
    g_object_set(nv12_filter, "caps", nv12_caps, nullptr);
    gst_caps_unref(h264_caps);
    gst_caps_unref(nv12_caps);
    gst_bin_add_many(GST_BIN(input.pipeline), source, demuxer, h264_filter, decoder,
                     nv12_filter, input.sink_element, nullptr);
    if (!gst_element_link(source, demuxer) ||
        !gst_element_link_many(h264_filter, decoder, nv12_filter, input.sink_element, nullptr)) {
        throw std::runtime_error("could not link GStreamer MPP source pipeline");
    }
    LinkContext link_context{h264_filter, {}};
    g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_demux_pad_added), &link_context);
    input.bus = gst_element_get_bus(input.pipeline);
    if (gst_element_set_state(input.pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        throw std::runtime_error("GStreamer MPP pipeline failed to start");
    }
    input.sample = gst_app_sink_try_pull_sample(GST_APP_SINK(input.sink_element), 10 * GST_SECOND);
    if (input.sample == nullptr) {
        GstMessage* message = gst_bus_pop_filtered(input.bus, GST_MESSAGE_ERROR);
        std::string detail = link_context.error;
        if (message != nullptr) {
            GError* error = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_error(message, &error, &debug);
            if (error != nullptr) detail = error->message;
            if (debug != nullptr && *debug != '\0') detail += " | " + std::string(debug);
            if (error != nullptr) g_error_free(error);
            g_free(debug);
            gst_message_unref(message);
        }
        throw std::runtime_error("could not pull MPP NV12 sample: " + detail);
    }
    return input;
}

edge::DecodedFrame copy_visible_nv12(GstSample* sample) {
    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (caps == nullptr || buffer == nullptr) throw std::runtime_error("MPP sample lacks caps or buffer");
    GstVideoInfo info{};
    if (!gst_video_info_from_caps(&info, caps) || GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
        throw std::runtime_error("MPP sample did not negotiate NV12");
    }
    GstVideoFrame mapped{};
    if (!gst_video_frame_map(&mapped, &info, buffer, GST_MAP_READ)) {
        throw std::runtime_error("could not map MPP sample for the baseline comparison");
    }
    edge::DecodedFrame frame;
    frame.width = GST_VIDEO_INFO_WIDTH(&info);
    frame.height = GST_VIDEO_INFO_HEIGHT(&info);
    frame.nv12.create(frame.height * 3 / 2, frame.width, CV_8UC1);
    const auto* y_plane = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 0));
    const auto* uv_plane = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 1));
    const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 0);
    const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 1);
    if (y_plane == nullptr || uv_plane == nullptr || y_stride < frame.width || uv_stride < frame.width) {
        gst_video_frame_unmap(&mapped);
        throw std::runtime_error("MPP sample has invalid plane addresses/strides");
    }
    for (int row = 0; row < frame.height; ++row) {
        std::memcpy(frame.nv12.ptr(row), y_plane + static_cast<std::size_t>(row) * y_stride,
                    static_cast<std::size_t>(frame.width));
    }
    for (int row = 0; row < frame.height / 2; ++row) {
        std::memcpy(frame.nv12.ptr(frame.height + row),
                    uv_plane + static_cast<std::size_t>(row) * uv_stride,
                    static_cast<std::size_t>(frame.width));
    }
    const GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) frame.pts_ms = static_cast<double>(pts) / GST_MSECOND;
    frame.source_copy_bytes = frame.nv12.total() * frame.nv12.elemSize();
    gst_video_frame_unmap(&mapped);
    return frame;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open model: " + path);
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("model is empty: " + path);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!input) throw std::runtime_error("failed reading model: " + path);
    return data;
}

std::uint64_t hash_bytes(const std::uint8_t* bytes, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string dims_string(const rknn_tensor_attr& attr) {
    std::string result = "[";
    for (std::uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i != 0) result += ',';
        result += std::to_string(attr.dims[i]);
    }
    return result + ']';
}

Model open_model(const std::string& path, bool shared_input) {
    Model model;
    auto bytes = read_file(path);
    int ret = rknn_init(&model.context, bytes.data(), static_cast<std::uint32_t>(bytes.size()), 0, nullptr);
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
    ret = rknn_query(model.context, RKNN_QUERY_IN_OUT_NUM, &model.io, sizeof(model.io));
    if (ret != RKNN_SUCC || model.io.n_input != 1 || model.io.n_output != 3) {
        throw std::runtime_error("PPE model must expose one input and three outputs; query ret=" +
                                 std::to_string(ret));
    }
    model.input.index = 0;
    model.native_input.index = 0;
    ret = rknn_query(model.context, RKNN_QUERY_INPUT_ATTR, &model.input, sizeof(model.input));
    if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_INPUT_ATTR failed: " + std::to_string(ret));
    ret = rknn_query(model.context, RKNN_QUERY_NATIVE_INPUT_ATTR, &model.native_input,
                     sizeof(model.native_input));
    if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_NATIVE_INPUT_ATTR failed: " + std::to_string(ret));

    model.outputs.resize(model.io.n_output);
    for (std::uint32_t i = 0; i < model.io.n_output; ++i) {
        model.outputs[i] = {};
        model.outputs[i].index = i;
        ret = rknn_query(model.context, RKNN_QUERY_OUTPUT_ATTR, &model.outputs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_OUTPUT_ATTR failed: " + std::to_string(ret));
    }
    model.postprocess.rknn_ctx = model.context;
    model.postprocess.io_num = model.io;
    model.postprocess.output_attrs = model.outputs.data();
    model.postprocess.model_width = static_cast<int>(model.input.dims[2]);
    model.postprocess.model_height = static_cast<int>(model.input.dims[1]);
    model.postprocess.model_channel = static_cast<int>(model.input.dims[3]);
    model.postprocess.is_quant = model.outputs[0].type == RKNN_TENSOR_INT8 ||
                                 model.outputs[0].type == RKNN_TENSOR_UINT8;
    if (model.postprocess.model_width != 640 || model.postprocess.model_height != 640 ||
        model.postprocess.model_channel != 3 || model.input.fmt != RKNN_TENSOR_NHWC) {
        throw std::runtime_error("unexpected PPE input tensor contract");
    }

    std::cout << "{\"kind\":\"model\",\"path\":\"" << path
              << "\",\"standard_input_dims\":\"" << dims_string(model.input)
              << "\",\"standard_input_fmt\":" << model.input.fmt
              << ",\"standard_input_type\":" << model.input.type
              << ",\"standard_input_w_stride\":" << model.input.w_stride
              << ",\"standard_input_size\":" << model.input.size
              << ",\"standard_input_size_with_stride\":" << model.input.size_with_stride
              << ",\"native_input_dims\":\"" << dims_string(model.native_input)
              << "\",\"native_input_fmt\":" << model.native_input.fmt
              << ",\"native_input_type\":" << model.native_input.type
              << ",\"native_input_w_stride\":" << model.native_input.w_stride
              << ",\"native_input_h_stride\":" << model.native_input.h_stride
              << ",\"native_input_size\":" << model.native_input.size
              << ",\"native_input_size_with_stride\":" << model.native_input.size_with_stride
              << ",\"native_input_pass_through\":" << static_cast<int>(model.native_input.pass_through)
              << ",\"output0_type\":" << model.outputs[0].type
              << ",\"output0_qnt_type\":" << model.outputs[0].qnt_type
              << "}\n";

    if (shared_input) {
        if (model.native_input.fmt != RKNN_TENSOR_NHWC) {
            throw std::runtime_error("native input is not NHWC; refusing an unsafe direct RGA write");
        }
        // Rockchip's zero-copy API example uses UINT8 for raw RGB image input,
        // allowing the runtime to apply the model input quantization internally.
        model.native_input.type = RKNN_TENSOR_UINT8;
        model.input_mem = rknn_create_mem(model.context, model.native_input.size_with_stride);
        if (model.input_mem == nullptr || model.input_mem->virt_addr == nullptr || model.input_mem->fd < 0) {
            throw std::runtime_error("rknn_create_mem did not return CPU address and valid fd");
        }
        ret = rknn_set_io_mem(model.context, model.input_mem, &model.native_input);
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_set_io_mem failed: " + std::to_string(ret));
        std::cout << "{\"kind\":\"shared_memory\",\"fd\":" << model.input_mem->fd
                  << ",\"size\":" << model.input_mem->size
                  << ",\"writable_virtual_address\":"
                  << (model.input_mem->virt_addr != nullptr ? "true" : "false")
                  << ",\"binding_ret\":" << ret << "}\n";
    }
    return model;
}

letterbox_t letterbox_for(const edge::DecodedFrame& frame, int width, int height,
                          int& resized_width, int& resized_height) {
    const float scale = std::min(static_cast<float>(width) / frame.width,
                                 static_cast<float>(height) / frame.height);
    resized_width = std::max(1, static_cast<int>(std::round(frame.width * scale)));
    resized_height = std::max(1, static_cast<int>(std::round(frame.height * scale)));
    return {(width - resized_width) / 2, (height - resized_height) / 2, scale};
}

double direct_rga_preprocess(const edge::DecodedFrame& frame, int width, int height,
                             rknn_tensor_mem* memory, std::uint32_t w_stride) {
    if (frame.nv12.empty() || frame.nv12.type() != CV_8UC1 || !frame.nv12.isContinuous()) {
        throw std::runtime_error("direct RGA probe requires canonical contiguous NV12");
    }
    if (w_stride < static_cast<std::uint32_t>(width) || memory->size < w_stride * height * 3) {
        throw std::runtime_error("RKNN shared input allocation cannot hold an NHWC stride image");
    }
    const auto started = Clock::now();
    int resized_width = 0;
    int resized_height = 0;
    const letterbox_t letterbox = letterbox_for(frame, width, height, resized_width, resized_height);
    const int row_stride_bytes = static_cast<int>(w_stride * 3);
    cv::Mat output(height, width, CV_8UC3, memory->virt_addr,
                   static_cast<std::size_t>(row_stride_bytes));
    const rga_buffer_t source = wrapbuffer_virtualaddr(
        frame.nv12.data, frame.width, frame.height, RK_FORMAT_YCbCr_420_SP,
        frame.width, frame.height);
    const rga_buffer_t destination = wrapbuffer_virtualaddr(
        memory->virt_addr, width, height, RK_FORMAT_BGR_888,
        static_cast<int>(w_stride), height);
    // Match the production CPU-RGA path: padding fill is a CPU write, then RGA
    // overwrites the letterboxed image region. The destination is RKNN-owned.
    output.setTo(cv::Scalar(114, 114, 114));
    const im_rect src_rect{0, 0, frame.width, frame.height};
    const im_rect dst_rect{letterbox.x_pad, letterbox.y_pad, resized_width, resized_height};
    const rga_buffer_t empty_buffer{};
    const im_rect empty_rect{};
    im_opt_t options{};
    options.interp = INTER_LINEAR;
    const IM_STATUS status = improcess(source, destination, empty_buffer, src_rect, dst_rect, empty_rect,
                                       -1, nullptr, &options, IM_SYNC);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        throw std::runtime_error(std::string("RGA direct preprocess failed: ") + imStrError_t(status));
    }
    (void)output;
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double direct_rga_preprocess_dmabuf(const edge::DecodedFrame& frame, GstSample* sample,
                                    rknn_tensor_mem* destination_memory,
                                    int model_width, int model_height,
                                    std::uint32_t destination_w_stride) {
    GstBuffer* gst_buffer = sample == nullptr ? nullptr : gst_sample_get_buffer(sample);
    GstCaps* caps = sample == nullptr ? nullptr : gst_sample_get_caps(sample);
    if (gst_buffer == nullptr || caps == nullptr || gst_buffer_n_memory(gst_buffer) != 1) {
        throw std::runtime_error("MPP sample must contain exactly one GstMemory");
    }
    GstMemory* source_memory = gst_buffer_peek_memory(gst_buffer, 0);
    if (!gst_is_dmabuf_memory(source_memory)) {
        throw std::runtime_error("MPP output GstMemory is not DMA-BUF backed");
    }
    const int source_fd = gst_dmabuf_memory_get_fd(source_memory);
    gsize source_offset = 0;
    gsize source_maxsize = 0;
    gst_memory_get_sizes(source_memory, &source_offset, &source_maxsize);
    if (source_fd < 0 || fcntl(source_fd, F_GETFD) < 0 || source_offset != 0 ||
        source_maxsize == 0 || source_maxsize > static_cast<gsize>(0x7fffffff)) {
        throw std::runtime_error("MPP DMA-BUF fd/offset/size is invalid for RGA import");
    }
    GstVideoInfo info{};
    if (!gst_video_info_from_caps(&info, caps) || GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
        throw std::runtime_error("MPP DMA-BUF sample is not negotiated NV12");
    }
    const GstVideoMeta* video_meta = gst_buffer_get_video_meta(gst_buffer);
    if (video_meta == nullptr || video_meta->n_planes < 2 || video_meta->stride[0] != video_meta->stride[1] ||
        video_meta->offset[0] != 0 ||
        video_meta->offset[1] != static_cast<gsize>(video_meta->stride[0]) * frame.height) {
        throw std::runtime_error("MPP NV12 plane layout is not a contiguous linear Y+UV layout");
    }
    if (destination_w_stride < static_cast<std::uint32_t>(model_width))
        throw std::runtime_error("RKNN destination width stride is smaller than model width");
    if (destination_memory == nullptr || destination_memory->fd < 0 ||
        destination_memory->size < static_cast<std::uint64_t>(destination_w_stride) * model_height * 3) {
        throw std::runtime_error("RKNN destination DMA-BUF is invalid or too small");
    }

    const auto started = Clock::now();
    const rga_buffer_handle_t source_handle = importbuffer_fd(
        source_fd, static_cast<int>(source_maxsize));
    if (source_handle == 0) throw std::runtime_error("RGA importbuffer_fd(MPP source) returned 0");
    const rga_buffer_handle_t destination_handle = importbuffer_fd(
        destination_memory->fd, static_cast<int>(destination_memory->size));
    if (destination_handle == 0) {
        releasebuffer_handle(source_handle);
        throw std::runtime_error("RGA importbuffer_fd(RKNN destination) returned 0");
    }

    int resized_width = 0;
    int resized_height = 0;
    const letterbox_t letterbox = letterbox_for(frame, model_width, model_height,
                                                 resized_width, resized_height);
    if (destination_memory->virt_addr == nullptr)
        throw std::runtime_error("RKNN destination has no CPU-visible mapping for validation");
    const rga_buffer_t source = wrapbuffer_handle(
        source_handle, frame.width, frame.height, RK_FORMAT_YCbCr_420_SP,
        video_meta->stride[0], frame.height);
    const rga_buffer_t destination = wrapbuffer_handle(
        destination_handle, model_width, model_height, RK_FORMAT_BGR_888,
        static_cast<int>(destination_w_stride), model_height);
    IM_STATUS status = imfill(destination, im_rect{0, 0, model_width, model_height},
                              0x727272, IM_SYNC);
    if (status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR) {
        im_opt_t options{};
        options.interp = INTER_LINEAR;
        status = improcess(source, destination, rga_buffer_t{},
        im_rect{0, 0, frame.width, frame.height},
        im_rect{letterbox.x_pad, letterbox.y_pad, resized_width, resized_height},
        im_rect{}, -1, nullptr, &options, IM_SYNC);
    }
    const int release_destination = releasebuffer_handle(destination_handle);
    const int release_source = releasebuffer_handle(source_handle);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        throw std::runtime_error(std::string("RGA MPP-FD to RKNN-FD improcess failed: ") +
                                 imStrError_t(status) + " (status " + std::to_string(status) + ")");
    }
    if (release_destination != IM_STATUS_SUCCESS && release_destination != IM_STATUS_NOERROR) {
        throw std::runtime_error("RGA RKNN destination handle release failed: " +
                                 std::to_string(release_destination));
    }
    if (release_source != IM_STATUS_SUCCESS && release_source != IM_STATUS_NOERROR) {
        throw std::runtime_error("RGA MPP source handle release failed: " +
                                 std::to_string(release_source));
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

Timing infer(Model& model, const edge::DecodedFrame& frame, bool shared,
             const cv::Mat* baseline_input, GstSample* source_sample = nullptr,
             bool source_is_dmabuf = false) {
    Timing timing;
    struct rusage usage_before{};
    struct rusage usage_after{};
    getrusage(RUSAGE_SELF, &usage_before);
    letterbox_t letterbox{};
    cv::Mat baseline_prepared_input;
    if (shared) {
        int resized_width = 0;
        int resized_height = 0;
        letterbox = letterbox_for(frame, model.postprocess.model_width,
                                  model.postprocess.model_height, resized_width, resized_height);
        const std::uint32_t w_stride = model.native_input.w_stride == 0
            ? static_cast<std::uint32_t>(model.postprocess.model_width)
            : model.native_input.w_stride;
        if (source_is_dmabuf) {
            timing.preprocess_ms = direct_rga_preprocess_dmabuf(
                frame, source_sample, model.input_mem,
                model.postprocess.model_width, model.postprocess.model_height, w_stride);
            timing.rga_dma_import_ok = true;
        } else {
            timing.preprocess_ms = direct_rga_preprocess(frame, model.postprocess.model_width,
                                                         model.postprocess.model_height,
                                                         model.input_mem, w_stride);
        }
        const auto sync_start = Clock::now();
        // The RGA virtual-address path is preceded by CPU padding writes, so it
        // uses TO_DEVICE. The MPP-FD path is device-written through imported
        // handles. Use bidirectional cache synchronization before the NPU
        // consumes the externally written buffer and before CPU parity reads.
        const auto sync_mode = source_is_dmabuf ? RKNN_MEMORY_SYNC_BIDIRECTIONAL
                                                 : RKNN_MEMORY_SYNC_TO_DEVICE;
        const int sync_ret = rknn_mem_sync(model.context, model.input_mem, sync_mode);
        if (sync_ret != RKNN_SUCC) throw std::runtime_error("rknn_mem_sync(shared input) failed: " +
                                                            std::to_string(sync_ret));
        timing.sync_ms = std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count();
        const std::size_t row_bytes = static_cast<std::size_t>(model.postprocess.model_width) * 3;
        const auto* input_data = static_cast<const std::uint8_t*>(model.input_mem->virt_addr);
        timing.input_hash = hash_bytes(input_data, row_bytes * model.postprocess.model_height);
        if (baseline_input != nullptr) {
            bool equal = true;
            for (int row = 0; row < model.postprocess.model_height; ++row) {
                const auto* actual = input_data + static_cast<std::size_t>(row) *
                    (model.native_input.w_stride == 0 ? model.postprocess.model_width : model.native_input.w_stride) * 3;
                if (std::memcmp(actual, baseline_input->ptr(row), row_bytes) != 0) equal = false;
            }
            if (!equal) throw std::runtime_error("direct RGA tensor bytes differ from production RGA preprocessing");
        }
    } else {
        auto prepared = edge::preprocess_frame(frame, model.postprocess.model_width,
                                                model.postprocess.model_height, false,
                                                edge::PreprocessBackend::Rga);
        timing.preprocess_ms = prepared.elapsed_ms;
        letterbox = prepared.letterbox;
        timing.input_hash = hash_bytes(prepared.model_input.data,
            prepared.model_input.total() * prepared.model_input.elemSize());
        rknn_input input{};
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = static_cast<std::uint32_t>(prepared.model_input.total() * prepared.model_input.elemSize());
        input.buf = prepared.model_input.data;
        const auto input_start = Clock::now();
        const int ret = rknn_inputs_set(model.context, model.io.n_input, &input);
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));
        timing.input_ms = std::chrono::duration<double, std::milli>(Clock::now() - input_start).count();
        baseline_prepared_input = std::move(prepared.model_input);
    }

    const auto run_start = Clock::now();
    int ret = rknn_run(model.context, nullptr);
    timing.run_ms = std::chrono::duration<double, std::milli>(Clock::now() - run_start).count();
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(ret));

    std::vector<rknn_output> outputs(model.io.n_output);
    for (std::uint32_t i = 0; i < model.io.n_output; ++i) {
        outputs[i] = {};
        outputs[i].index = i;
        outputs[i].want_float = model.postprocess.is_quant ? 0 : 1;
    }
    const auto fetch_start = Clock::now();
    ret = rknn_outputs_get(model.context, model.io.n_output, outputs.data(), nullptr);
    timing.fetch_ms = std::chrono::duration<double, std::milli>(Clock::now() - fetch_start).count();
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));
    timing.outputs.resize(outputs.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto* begin = static_cast<const std::uint8_t*>(outputs[i].buf);
        timing.outputs[i].assign(begin, begin + outputs[i].size);
    }
    object_detect_result_list detections{};
    if (!shared) {
        // Letterbox metadata is needed for postprocess; recompute from the same frame/model shape.
        int rw = 0, rh = 0;
        letterbox = letterbox_for(frame, model.postprocess.model_width,
                                  model.postprocess.model_height, rw, rh);
    }
    if (post_process_yolox_ppe(&model.postprocess, outputs.data(), &letterbox, 0.25F, 0.45F,
                               frame.width, frame.height, &detections) != 0) {
        rknn_outputs_release(model.context, model.io.n_output, outputs.data());
        throw std::runtime_error("PPE postprocess failed");
    }
    timing.detections.assign(detections.results, detections.results + detections.count);
    rknn_outputs_release(model.context, model.io.n_output, outputs.data());
    getrusage(RUSAGE_SELF, &usage_after);
    const auto cpu_ms = [](const timeval& start, const timeval& end) {
        return (static_cast<double>(end.tv_sec - start.tv_sec) * 1000.0) +
               (static_cast<double>(end.tv_usec - start.tv_usec) / 1000.0);
    };
    timing.cpu_user_ms = cpu_ms(usage_before.ru_utime, usage_after.ru_utime);
    timing.cpu_system_ms = cpu_ms(usage_before.ru_stime, usage_after.ru_stime);
    timing.max_rss_kb = usage_after.ru_maxrss;
    return timing;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(p * static_cast<double>(values.size() - 1))];
}

void report_run(const char* name, const std::vector<Timing>& runs) {
    std::vector<double> preprocess, input, run, fetch;
    for (const auto& result : runs) {
        preprocess.push_back(result.preprocess_ms);
        input.push_back(result.input_ms);
        run.push_back(result.run_ms);
        fetch.push_back(result.fetch_ms);
    }
    std::vector<double> total, cpu_user, cpu_system;
    for (const auto& result : runs) {
        total.push_back(result.preprocess_ms + result.sync_ms + result.input_ms +
                        result.run_ms + result.fetch_ms);
        cpu_user.push_back(result.cpu_user_ms);
        cpu_system.push_back(result.cpu_system_ms);
    }
    const auto& last = runs.back();
    std::cout << "{\"kind\":\"result\",\"mode\":\"" << name
              << "\",\"iterations\":" << runs.size()
              << ",\"preprocess_p50_ms\":" << percentile(preprocess, 0.50)
              << ",\"preprocess_p95_ms\":" << percentile(preprocess, 0.95)
              << ",\"sync_p50_ms\":" << percentile([&] {
                    std::vector<double> values;
                    for (const auto& result : runs) values.push_back(result.sync_ms);
                    return values;
                }(), 0.50)
              << ",\"input_submit_p50_ms\":" << percentile(input, 0.50)
              << ",\"rknn_run_p50_ms\":" << percentile(run, 0.50)
              << ",\"output_fetch_p50_ms\":" << percentile(fetch, 0.50)
              << ",\"measured_total_p50_ms\":" << percentile(total, 0.50)
              << ",\"measured_total_p95_ms\":" << percentile(total, 0.95)
              << ",\"cpu_user_p50_ms\":" << percentile(cpu_user, 0.50)
              << ",\"cpu_system_p50_ms\":" << percentile(cpu_system, 0.50)
              << ",\"process_max_rss_kb\":" << last.max_rss_kb
              << ",\"rga_dma_import_ok\":" << (last.rga_dma_import_ok ? "true" : "false")
              << ",\"input_hash_fnv1a64\":" << last.input_hash
              << ",\"output_hashes\":[";
    for (std::size_t i = 0; i < last.outputs.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << hash_bytes(last.outputs[i].data(), last.outputs[i].size());
    }
    std::cout << "],\"detection_count\":" << last.detections.size() << ",\"detections\":[";
    for (std::size_t i = 0; i < last.detections.size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto& detection = last.detections[i];
        std::cout << "{\"class_id\":" << detection.cls_id << ",\"confidence\":"
                  << std::setprecision(7) << detection.prop << ",\"x\":" << detection.box.left
                  << ",\"y\":" << detection.box.top << ",\"width\":" << detection.box.right - detection.box.left
                  << ",\"height\":" << detection.box.bottom - detection.box.top << '}';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: rknn_shared_input_probe MODEL.rknn INPUT.mp4 [iterations=12]\n";
        return 2;
    }
    const int iterations = argc == 4 ? std::max(1, std::atoi(argv[3])) : 12;
    try {
        MppInput input = open_mpp_input(argv[2]);
        edge::DecodedFrame frame = copy_visible_nv12(input.sample);
        GstBuffer* gst_buffer = gst_sample_get_buffer(input.sample);
        GstMemory* gst_memory = gst_buffer_peek_memory(gst_buffer, 0);
        GstVideoMeta* video_meta = gst_buffer_get_video_meta(gst_buffer);
        const int mpp_fd = gst_is_dmabuf_memory(gst_memory)
            ? gst_dmabuf_memory_get_fd(gst_memory) : -1;
        gsize mpp_offset = 0;
        gsize mpp_maxsize = 0;
        const gsize mpp_memory_size = gst_memory_get_sizes(gst_memory, &mpp_offset, &mpp_maxsize);
        std::cout << "{\"kind\":\"source_frame\",\"width\":" << frame.width
                  << ",\"height\":" << frame.height << ",\"format\":\"NV12\""
                  << ",\"source_copy_bytes\":" << frame.source_copy_bytes
                  << ",\"pts_ms\":" << frame.pts_ms
                  << ",\"gst_memory_type\":\""
                  << (gst_memory->allocator == nullptr ? "unknown" : gst_memory->allocator->mem_type)
                  << "\",\"mpp_dma_fd_valid\":"
                  << (mpp_fd >= 0 && fcntl(mpp_fd, F_GETFD) >= 0 ? "true" : "false")
                  << ",\"mpp_fd\":" << mpp_fd << ",\"mpp_offset\":" << mpp_offset
                  << ",\"mpp_memory_size\":" << mpp_memory_size
                  << ",\"mpp_max_size\":" << mpp_maxsize
                  << ",\"plane_y_stride\":" << (video_meta == nullptr ? 0 : video_meta->stride[0])
                  << ",\"plane_uv_stride\":" << (video_meta == nullptr ? 0 : video_meta->stride[1])
                  << ",\"plane_uv_offset\":" << (video_meta == nullptr ? 0 : video_meta->offset[1])
                  << "}\n";

        Model baseline_model = open_model(argv[1], false);
        const auto baseline_prepared = edge::preprocess_frame(
            frame, baseline_model.postprocess.model_width, baseline_model.postprocess.model_height,
            false, edge::PreprocessBackend::Rga);
        const cv::Mat expected_input = baseline_prepared.model_input.clone();
        std::vector<Timing> baseline_runs;
        for (int i = 0; i < iterations + 2; ++i) {
            auto result = infer(baseline_model, frame, false, nullptr);
            if (i >= 2) baseline_runs.push_back(std::move(result));
        }
        report_run("B0_rknn_inputs_set", baseline_runs);

        Model shared_model = open_model(argv[1], true);
        std::vector<Timing> shared_runs;
        for (int i = 0; i < iterations + 2; ++i) {
            auto result = infer(shared_model, frame, true, &expected_input);
            if (i >= 2) shared_runs.push_back(std::move(result));
        }
        report_run("B1_rknn_set_io_mem_rga_virtualaddr", shared_runs);

        Model dmabuf_model = open_model(argv[1], true);
        std::vector<Timing> dmabuf_runs;
        for (int i = 0; i < iterations + 2; ++i) {
            auto result = infer(dmabuf_model, frame, true, &expected_input,
                                input.sample, true);
            if (i >= 2) dmabuf_runs.push_back(std::move(result));
        }
        report_run("B2_mpp_dmabuf_to_rga_to_rknn_fd", dmabuf_runs);
        const auto& baseline = baseline_runs.back();
        const auto& shared = shared_runs.back();
        const auto& dmabuf = dmabuf_runs.back();
        bool outputs_equal = baseline.outputs == shared.outputs;
        const bool dmabuf_outputs_equal = baseline.outputs == dmabuf.outputs;
        bool detections_equal = baseline.detections.size() == shared.detections.size();
        if (detections_equal) {
            for (std::size_t i = 0; i < baseline.detections.size(); ++i) {
                const auto& a = baseline.detections[i];
                const auto& b = shared.detections[i];
                detections_equal = detections_equal && a.cls_id == b.cls_id &&
                    a.box.left == b.box.left && a.box.top == b.box.top &&
                    a.box.right == b.box.right && a.box.bottom == b.box.bottom &&
                    a.prop == b.prop;
            }
        }
        bool dmabuf_detections_equal = baseline.detections.size() == dmabuf.detections.size();
        if (dmabuf_detections_equal) {
            for (std::size_t i = 0; i < baseline.detections.size(); ++i) {
                const auto& a = baseline.detections[i];
                const auto& b = dmabuf.detections[i];
                dmabuf_detections_equal = dmabuf_detections_equal && a.cls_id == b.cls_id &&
                    a.box.left == b.box.left && a.box.top == b.box.top &&
                    a.box.right == b.box.right && a.box.bottom == b.box.bottom && a.prop == b.prop;
            }
        }
        std::cout << "{\"kind\":\"comparison\",\"outputs_byte_equal\":"
                  << (outputs_equal ? "true" : "false") << ",\"detections_exact_equal\":"
                  << (detections_equal ? "true" : "false") << ",\"repeated_frames\":"
                  << iterations << ",\"dmabuf_outputs_byte_equal\":"
                  << (dmabuf_outputs_equal ? "true" : "false")
                  << ",\"dmabuf_detections_exact_equal\":"
                  << (dmabuf_detections_equal ? "true" : "false") << "}\n";
        return outputs_equal && detections_equal && dmabuf_outputs_equal &&
               dmabuf_detections_equal ? 0 : 7;
    } catch (const std::exception& exception) {
        std::cerr << "probe failed: " << exception.what() << '\n';
        return 1;
    }
}
