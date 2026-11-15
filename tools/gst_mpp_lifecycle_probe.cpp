#include "video_source.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::size_t open_fd_count() {
    DIR* directory = opendir("/proc/self/fd");
    if (directory == nullptr) return 0;
    const int directory_fd = dirfd(directory);
    std::size_t count = 0;
    while (const dirent* entry = readdir(directory)) {
        char* end = nullptr;
        const long fd = std::strtol(entry->d_name, &end, 10);
        if (end != entry->d_name && *end == '\0' && fd != directory_fd) ++count;
    }
    closedir(directory);
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: gst_mpp_lifecycle_probe INPUT.mp4 [cycles=10]\n";
        return 2;
    }
    const int cycles = argc == 3 ? std::stoi(argv[2]) : 10;
    if (cycles < 1 || cycles > 100) {
        std::cerr << "cycles must be between 1 and 100\n";
        return 2;
    }
    const auto initial_fds = open_fd_count();
    std::cout << "{\"kind\":\"lifecycle_start\",\"open_fds\":" << initial_fds << "}\n";
    for (int cycle = 1; cycle <= cycles; ++cycle) {
        auto source = edge::make_video_source(edge::VideoBackend::GStreamerMpp, true);
        std::string error;
        if (!source->open(argv[1], edge::QueuePolicy::Latest, error)) {
            std::cerr << "open failed at cycle " << cycle << ": " << error << '\n';
            return 3;
        }
        edge::DecodedFrame frame;
        const auto status = source->read(frame, error);
        if (status != edge::FrameReadStatus::Frame || frame.dma_buf_fd < 0 ||
            !frame.dma_buf_lifetime) {
            std::cerr << "read failed at cycle " << cycle << ": " << error << '\n';
            return 4;
        }
        const int borrowed_fd = frame.dma_buf_fd;
        struct stat before_close{};
        const bool valid_before_close = fcntl(borrowed_fd, F_GETFD) >= 0 &&
                                        fstat(borrowed_fd, &before_close) == 0;
        source->close();
        struct stat after_close{};
        const bool valid_while_frame_retained = fcntl(borrowed_fd, F_GETFD) >= 0 &&
                                                fstat(borrowed_fd, &after_close) == 0 &&
                                                before_close.st_dev == after_close.st_dev &&
                                                before_close.st_ino == after_close.st_ino;
        frame.dma_buf_lifetime.reset();
        const bool fd_released_with_frame = fcntl(borrowed_fd, F_GETFD) < 0;
        source.reset();
        const auto fds_after_cycle = open_fd_count();
        std::cout << "{\"kind\":\"lifecycle_cycle\",\"cycle\":" << cycle
                  << ",\"fd\":" << borrowed_fd
                  << ",\"source_copy_bytes\":" << frame.source_copy_bytes
                  << ",\"frame_memory_size\":" << frame.dma_buf_size
                  << ",\"frame_stride_y\":" << frame.dma_buf_y_stride
                  << ",\"valid_before_close\":" << (valid_before_close ? "true" : "false")
                  << ",\"valid_while_frame_retained\":"
                  << (valid_while_frame_retained ? "true" : "false")
                  << ",\"fd_released_with_frame\":" << (fd_released_with_frame ? "true" : "false")
                  << ",\"open_fds_after_cycle\":" << fds_after_cycle << "}\n";
        if (!valid_before_close || !valid_while_frame_retained || !fd_released_with_frame) return 5;
    }
    std::cout << "{\"kind\":\"lifecycle_summary\",\"cycles\":" << cycles
              << ",\"initial_open_fds\":" << initial_fds
              << ",\"final_open_fds\":" << open_fd_count() << "}\n";
    return 0;
}
