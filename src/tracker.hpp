#pragma once

#include <cstdint>
#include <set>
#include <vector>

namespace edge {

struct Detection {
    int class_id = -1;
    float confidence = 0.0F;
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
    const char* label = "unknown";
};

struct TrackedDetection {
    std::uint64_t track_id = 0;
    Detection detection;
    bool created = false;
};

class IoUTracker {
public:
    explicit IoUTracker(float match_threshold = 0.30F, int max_missed = 5);
    std::vector<TrackedDetection> update(const std::vector<Detection>& detections);
    std::set<std::uint64_t> active_ids() const;

private:
    struct TrackState {
        std::uint64_t id;
        Detection detection;
        float velocity_x = 0.0F;
        float velocity_y = 0.0F;
        int missed = 0;
        std::uint64_t age = 0;
    };

    static float iou(const Detection& a, const Detection& b);

    float match_threshold_;
    int max_missed_;
    std::uint64_t next_id_ = 1;
    std::vector<TrackState> tracks_;
};

}  // namespace edge
