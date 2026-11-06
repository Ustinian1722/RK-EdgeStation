#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <rga/im2d.hpp>
#include <rga/rga.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct LinkContext {
    GstElement* h264_filter = nullptr;
    std::string error;
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
        if (status != GST_PAD_LINK_OK) {
            context->error = "qtdemux H.264 pad link failed: " +
                             std::to_string(static_cast<int>(status));
        }
    }
    gst_object_unref(sink_pad);
}

std::size_t open_fd_count() {
    DIR* directory = opendir("/proc/self/fd");
    if (directory == nullptr) return 0;
    std::size_t count = 0;
    while (readdir(directory) != nullptr) ++count;
    closedir(directory);
    return count >= 2 ? count - 2 : count;
}

std::string json_escape(const gchar* value);

std::string open_fd_targets_json() {
    DIR* directory = opendir("/proc/self/fd");
    if (directory == nullptr) return "[]";
    const int directory_fd = dirfd(directory);
    std::string result = "[";
    bool first = true;
    while (const dirent* entry = readdir(directory)) {
        char* end = nullptr;
        const long fd = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || fd == directory_fd) continue;
        const std::string link = "/proc/self/fd/" + std::to_string(fd);
        char target[512]{};
        const ssize_t size = readlink(link.c_str(), target, sizeof(target) - 1);
        if (size < 0) continue;
        target[size] = '\0';
        if (!first) result += ',';
        first = false;
        result += "{\"fd\":" + std::to_string(fd) + ",\"target\":\"" +
                  json_escape(target) + "\"}";
    }
    closedir(directory);
    result += ']';
    return result;
}

std::string json_escape(const gchar* value);

std::string json_escape(const gchar* value) {
    std::string result;
    if (value == nullptr) return result;
    for (const unsigned char ch : std::string(value)) {
        switch (ch) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += static_cast<char>(ch); break;
        }
    }
    return result;
}

struct ProbeTotals {
    std::uint64_t samples = 0;
    std::uint64_t dmabuf_samples = 0;
    std::uint64_t dmabuf_memories = 0;
    std::uint64_t valid_fds = 0;
    bool first_sample_details_printed = false;
    bool rga_attempted = false;
    bool rga_imported = false;
    bool rga_byte_match = false;
    int rga_handle = 0;
    int rga_status = 0;
    std::string rga_error;
    std::uint64_t visible_bytes = 0;
    std::uint64_t mapped_allocation_bytes = 0;
    std::uint64_t imported_fd_bytes = 0;
};

void try_rga_import_and_compare(GstSample* sample, GstVideoInfo& info,
                                const GstVideoMeta* video_meta,
                                GstMemory* memory, int fd, gsize maxsize,
                                ProbeTotals& totals) {
    totals.rga_attempted = true;
    totals.imported_fd_bytes = maxsize;
    const int width = GST_VIDEO_INFO_WIDTH(&info);
    const int height = GST_VIDEO_INFO_HEIGHT(&info);
    const int stride = video_meta != nullptr ? video_meta->stride[0]
                                             : GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0 ||
        stride < width || gst_buffer_n_memory(gst_sample_get_buffer(sample)) != 1) {
        totals.rga_error = "unsupported dimensions, stride, or multi-memory frame";
        return;
    }

    const rga_buffer_handle_t handle = importbuffer_fd(fd, static_cast<int>(maxsize));
    totals.rga_handle = handle;
    if (handle == 0) {
        totals.rga_error = "importbuffer_fd returned handle 0";
        return;
    }
    totals.rga_imported = true;

    const auto visible_size = static_cast<std::size_t>(width) * height * 3 / 2;
    std::vector<std::uint8_t> rga_output(visible_size, 0);
    const rga_buffer_t rga_source = wrapbuffer_handle(
        handle, width, height, RK_FORMAT_YCbCr_420_SP, stride, height);
    const rga_buffer_t rga_destination = wrapbuffer_virtualaddr(
        rga_output.data(), width, height, RK_FORMAT_YCbCr_420_SP, width, height);
    const IM_STATUS operation_status = imcopy(rga_source, rga_destination, IM_SYNC);
    totals.rga_status = static_cast<int>(operation_status);
    if (operation_status != IM_STATUS_SUCCESS && operation_status != IM_STATUS_NOERROR) {
        const char* message = imStrError_t(operation_status);
        totals.rga_error = message == nullptr ? "imcopy failed" : message;
        releasebuffer_handle(handle);
        totals.rga_handle = 0;
        return;
    }

    // This CPU map/copy exists only in the isolated correctness probe. It is not
    // part of the production Worker or a claimed zero-copy benchmark path.
    GstVideoFrame mapped{};
    if (!gst_video_frame_map(&mapped, &info, gst_sample_get_buffer(sample), GST_MAP_READ)) {
        totals.rga_error = "gst_video_frame_map failed during byte comparison";
        releasebuffer_handle(handle);
        totals.rga_handle = 0;
        return;
    }
    std::vector<std::uint8_t> reference(visible_size);
    const auto* y_plane = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 0));
    const auto* uv_plane = static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 1));
    const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 0);
    const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 1);
    bool layout_valid = y_plane != nullptr && uv_plane != nullptr &&
                        y_stride >= width && uv_stride >= width;
    if (layout_valid) {
        for (int y = 0; y < height; ++y) {
            std::memcpy(reference.data() + static_cast<std::size_t>(y) * width,
                        y_plane + static_cast<std::size_t>(y) * y_stride,
                        static_cast<std::size_t>(width));
        }
        auto* reference_uv = reference.data() + static_cast<std::size_t>(width) * height;
        for (int y = 0; y < height / 2; ++y) {
            std::memcpy(reference_uv + static_cast<std::size_t>(y) * width,
                        uv_plane + static_cast<std::size_t>(y) * uv_stride,
                        static_cast<std::size_t>(width));
        }
        totals.rga_byte_match = reference == rga_output;
    }
    gst_video_frame_unmap(&mapped);
    totals.visible_bytes = visible_size;
    totals.mapped_allocation_bytes = gst_memory_get_sizes(memory, nullptr, nullptr);
    if (!layout_valid) totals.rga_error = "mapped plane pointers or strides are invalid";
    else if (!totals.rga_byte_match) totals.rga_error = "RGA output differs from visible NV12 bytes";
    releasebuffer_handle(handle);
    totals.rga_handle = 0;
}

void inspect_first_sample(GstSample* sample, ProbeTotals& totals) {
    GstCaps* caps = gst_sample_get_caps(sample);
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (caps == nullptr || buffer == nullptr) return;

    GstVideoInfo info{};
    const bool parsed = gst_video_info_from_caps(&info, caps);
    const GstVideoMeta* video_meta = gst_buffer_get_video_meta(buffer);
    gchar* caps_text = gst_caps_to_string(caps);
    const GstCapsFeatures* features = gst_caps_get_features(caps, 0);
    const bool caps_dmabuf = features != nullptr &&
        gst_caps_features_contains(features, GST_CAPS_FEATURE_MEMORY_DMABUF);
    const GstStructure* structure = gst_caps_get_structure(caps, 0);
    const gchar* drm_format = structure == nullptr ? nullptr :
        gst_structure_get_string(structure, "drm-format");

    std::cout << "{\"kind\":\"first_sample\",\"caps\":\""
              << json_escape(caps_text) << "\",\"caps_dmabuf\":"
              << (caps_dmabuf ? "true" : "false") << ",\"drm_format\":\""
              << json_escape(drm_format) << "\",\"video_info_valid\":"
              << (parsed ? "true" : "false") << ",\"width\":"
              << (parsed ? GST_VIDEO_INFO_WIDTH(&info) : 0) << ",\"height\":"
              << (parsed ? GST_VIDEO_INFO_HEIGHT(&info) : 0)
              << ",\"video_meta_present\":" << (video_meta != nullptr ? "true" : "false")
              << ",\"buffer_bytes\":" << gst_buffer_get_size(buffer)
              << ",\"memory_count\":" << gst_buffer_n_memory(buffer)
              << ",\"memories\":[";
    for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
        GstMemory* memory = gst_buffer_peek_memory(buffer, i);
        gsize offset = 0;
        gsize maxsize = 0;
        const gsize memory_size = gst_memory_get_sizes(memory, &offset, &maxsize);
        const bool is_dmabuf = gst_is_dmabuf_memory(memory);
        const int fd = is_dmabuf ? gst_dmabuf_memory_get_fd(memory) : -1;
        const bool fd_valid = fd >= 0 && fcntl(fd, F_GETFD) >= 0;
        struct stat stat_info{};
        const bool stat_valid = fd_valid && fstat(fd, &stat_info) == 0;
        if (i != 0) std::cout << ',';
        std::cout << "{\"index\":" << i << ",\"allocator_type\":\""
                  << json_escape(memory->allocator == nullptr ? nullptr : memory->allocator->mem_type)
                  << "\",\"is_dmabuf\":" << (is_dmabuf ? "true" : "false")
                  << ",\"fd\":" << fd << ",\"fd_valid\":" << (fd_valid ? "true" : "false")
                  << ",\"fstat_size\":" << (stat_valid ? stat_info.st_size : -1)
                  << ",\"offset\":" << offset << ",\"memory_size\":" << memory_size
                  << ",\"max_size\":" << maxsize << '}';
        if (parsed && is_dmabuf && fd_valid && i == 0) {
            try_rga_import_and_compare(sample, info, video_meta, memory, fd, maxsize, totals);
        }
    }
    std::cout << "],\"planes\":[";
    if (video_meta != nullptr) {
        for (guint i = 0; i < video_meta->n_planes; ++i) {
            if (i != 0) std::cout << ',';
            std::cout << "{\"offset\":" << video_meta->offset[i]
                      << ",\"stride\":" << video_meta->stride[i] << '}';
        }
    } else if (parsed) {
        for (guint i = 0; i < GST_VIDEO_INFO_N_PLANES(&info); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << "{\"offset\":" << GST_VIDEO_INFO_PLANE_OFFSET(&info, i)
                      << ",\"stride\":" << GST_VIDEO_INFO_PLANE_STRIDE(&info, i) << '}';
        }
    }
    std::cout << "],\"rga_attempted\":" << (totals.rga_attempted ? "true" : "false")
              << ",\"rga_imported\":" << (totals.rga_imported ? "true" : "false")
              << ",\"rga_status\":" << totals.rga_status
              << ",\"rga_byte_match\":" << (totals.rga_byte_match ? "true" : "false")
              << ",\"rga_error\":\"" << json_escape(totals.rga_error.c_str())
              << "\",\"visible_bytes\":" << totals.visible_bytes
              << ",\"mapped_allocation_bytes\":" << totals.mapped_allocation_bytes
              << ",\"imported_fd_bytes\":" << totals.imported_fd_bytes << "}\n";
    g_free(caps_text);
    totals.first_sample_details_printed = true;
}

void print_bus_error(GstBus* bus, const char* fallback, std::string& error_text) {
    GstMessage* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
    if (message == nullptr) {
        error_text = fallback;
        return;
    }
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);
    error_text = error == nullptr ? fallback : error->message;
    if (debug != nullptr && *debug != '\0') error_text += " | " + std::string(debug);
    if (error != nullptr) g_error_free(error);
    g_free(debug);
    gst_message_unref(message);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "usage: gst_dmabuf_probe INPUT.mp4 [dma-feature:0|1] "
                     "[request-caps-dmabuf:0|1] [frames:0=drain-EOS]\n";
        return 2;
    }
    const std::string dma_feature_arg = argc >= 3 ? argv[2] : "0";
    const std::string caps_feature_arg = argc >= 4 ? argv[3] : "0";
    const std::uint64_t frame_limit = argc >= 5 ? std::strtoull(argv[4], nullptr, 10) : 0;
    if ((dma_feature_arg != "0" && dma_feature_arg != "1") ||
        (caps_feature_arg != "0" && caps_feature_arg != "1")) {
        std::cerr << "dma-feature and request-caps-dmabuf must each be 0 or 1\n";
        return 2;
    }
    const bool enable_dma_feature = dma_feature_arg == "1";
    const bool request_caps_dmabuf = caps_feature_arg == "1";

    gst_init(nullptr, nullptr);
    const std::size_t fds_before = open_fd_count();
    const std::string fd_targets_before = open_fd_targets_json();
    GstElement* pipeline = gst_pipeline_new("gst-dmabuf-probe");
    GstElement* source = gst_element_factory_make("filesrc", "file-source");
    GstElement* demuxer = gst_element_factory_make("qtdemux", "mp4-demuxer");
    GstElement* h264_filter = gst_element_factory_make("capsfilter", "h264-filter");
    GstElement* decoder = gst_element_factory_make("mppvideodec", "mpp-decoder");
    GstElement* output_filter = gst_element_factory_make("capsfilter", "output-filter");
    GstElement* sink_element = gst_element_factory_make("appsink", "sample-sink");
    if (pipeline == nullptr || source == nullptr || demuxer == nullptr || h264_filter == nullptr ||
        decoder == nullptr || output_filter == nullptr || sink_element == nullptr) {
        std::cerr << "required GStreamer element unavailable\n";
        if (pipeline != nullptr) gst_object_unref(pipeline);
        return 3;
    }
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(decoder), "dma-feature") == nullptr) {
        std::cerr << "installed mppvideodec has no dma-feature property\n";
        gst_object_unref(pipeline);
        return 3;
    }

    g_object_set(source, "location", argv[1], nullptr);
    g_object_set(decoder, "dma-feature", enable_dma_feature, nullptr);
    g_object_set(sink_element, "max-buffers", static_cast<guint>(1), "drop", FALSE,
                 "sync", FALSE, "wait-on-eos", FALSE, "enable-last-sample", FALSE, nullptr);
    GstCaps* h264_caps = gst_caps_from_string(
        "video/x-h264,parsed=true,stream-format=avc,alignment=au");
    GstCaps* nv12_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
                                             "NV12", nullptr);
    if (request_caps_dmabuf) {
        gst_caps_set_features(nv12_caps, 0, gst_caps_features_new(
            GST_CAPS_FEATURE_MEMORY_DMABUF, nullptr));
    }
    g_object_set(h264_filter, "caps", h264_caps, nullptr);
    g_object_set(output_filter, "caps", nv12_caps, nullptr);
    gst_caps_unref(h264_caps);
    gst_caps_unref(nv12_caps);
    gst_bin_add_many(GST_BIN(pipeline), source, demuxer, h264_filter, decoder,
                     output_filter, sink_element, nullptr);
    if (!gst_element_link(source, demuxer) ||
        !gst_element_link_many(h264_filter, decoder, output_filter, sink_element, nullptr)) {
        std::cerr << "could not link GStreamer probe elements\n";
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        return 4;
    }
    LinkContext link_context{h264_filter, {}};
    g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_demux_pad_added), &link_context);
    GstBus* bus = gst_element_get_bus(pipeline);
    GstAppSink* sink = GST_APP_SINK(sink_element);
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::string error;
        print_bus_error(bus, "GStreamer pipeline failed to enter PLAYING", error);
        std::cerr << error << '\n';
        gst_object_unref(bus);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        return 5;
    }

    ProbeTotals totals;
    std::string pipeline_error;
    bool saw_eos = false;
    const auto start = std::chrono::steady_clock::now();
    while (frame_limit == 0 || totals.samples < frame_limit) {
        GstSample* sample = gst_app_sink_try_pull_sample(sink, 5 * GST_SECOND);
        if (sample == nullptr) {
            if (!link_context.error.empty()) pipeline_error = link_context.error;
            if (gst_app_sink_is_eos(sink)) {
                saw_eos = true;
                break;
            }
            print_bus_error(bus, "timed out waiting for an appsink sample", pipeline_error);
            break;
        }
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        bool sample_has_dmabuf = false;
        for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
            GstMemory* memory = gst_buffer_peek_memory(buffer, i);
            if (gst_is_dmabuf_memory(memory)) {
                sample_has_dmabuf = true;
                ++totals.dmabuf_memories;
                const int fd = gst_dmabuf_memory_get_fd(memory);
                if (fd >= 0 && fcntl(fd, F_GETFD) >= 0) ++totals.valid_fds;
            }
        }
        if (sample_has_dmabuf) ++totals.dmabuf_samples;
        if (!totals.first_sample_details_printed) inspect_first_sample(sample, totals);
        ++totals.samples;
        gst_sample_unref(sample);
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    const std::size_t fds_after_pipeline_teardown = open_fd_count();
    const std::string fd_targets_after_pipeline_teardown = open_fd_targets_json();
    gst_deinit();
    const std::size_t fds_after_gst_deinit = open_fd_count();
    const std::string fd_targets_after_gst_deinit = open_fd_targets_json();

    std::cout << "{\"kind\":\"summary\",\"dma_feature\":"
              << (enable_dma_feature ? "true" : "false")
              << ",\"requested_caps_dmabuf\":" << (request_caps_dmabuf ? "true" : "false")
              << ",\"frame_limit\":" << frame_limit << ",\"samples\":" << totals.samples
              << ",\"dmabuf_samples\":" << totals.dmabuf_samples
              << ",\"dmabuf_memories\":" << totals.dmabuf_memories
              << ",\"valid_fds\":" << totals.valid_fds
              << ",\"rga_attempted\":" << (totals.rga_attempted ? "true" : "false")
              << ",\"rga_imported\":" << (totals.rga_imported ? "true" : "false")
              << ",\"rga_byte_match\":" << (totals.rga_byte_match ? "true" : "false")
              << ",\"rga_error\":\"" << json_escape(totals.rga_error.c_str())
              << "\",\"saw_eos\":" << (saw_eos ? "true" : "false")
              << ",\"pipeline_error\":\"" << json_escape(pipeline_error.c_str())
              << "\",\"elapsed_ms\":" << elapsed_ms
              << ",\"open_fds_before\":" << fds_before
              << ",\"open_fd_targets_before\":" << fd_targets_before
              << ",\"open_fds_after_pipeline_teardown\":" << fds_after_pipeline_teardown
              << ",\"open_fd_targets_after_pipeline_teardown\":" << fd_targets_after_pipeline_teardown
              << ",\"open_fds_after_gst_deinit\":" << fds_after_gst_deinit
              << ",\"open_fd_targets_after_gst_deinit\":" << fd_targets_after_gst_deinit
              << "}\n";
    return totals.samples > 0 && pipeline_error.empty() ? 0 : 6;
}
