#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace edge {

struct MultiCameraConfig {
    std::string id;
    std::string source;
    bool enabled = true;
    double max_fps = 0.0;
    unsigned int weight = 1;
    std::size_t queue_capacity = 1;
    std::vector<std::string> tasks{"ppe", "roi"};
    double roi_x = 0.0;
    double roi_y = 0.0;
    double roi_width = 1.0;
    double roi_height = 1.0;
};

struct MultiStreamConfig {
    std::vector<MultiCameraConfig> cameras;
};

// Reads a deliberately small, strict JSON config so the RK3576 build has no
// extra parser dependency. Unknown keys are ignored for forward compatibility.
MultiStreamConfig load_multistream_config(const std::string& path);

// Smooth weighted round robin over currently ready cameras. Cameras with no
// queued frame accrue no credit, so a temporarily quiet stream cannot hold up
// the inference executor or starve active streams.
class WeightedFairScheduler {
public:
    explicit WeightedFairScheduler(std::vector<unsigned int> weights);
    int select(const std::vector<bool>& ready);
    void reset();

private:
    std::vector<unsigned int> weights_;
    std::vector<long long> current_;
};

std::string redact_source(const std::string& source);

}  // namespace edge
