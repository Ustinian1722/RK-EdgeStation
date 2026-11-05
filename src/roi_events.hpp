#pragma once

#include "tracker.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace edge {

enum class RoiEventType { Enter, Leave, Dwell };

struct RoiObservation {
    std::uint64_t track_id = 0;
    Detection detection;
    bool inside = false;
};

struct RoiEventTransition {
    std::uint64_t track_id = 0;
    RoiEventType type = RoiEventType::Enter;
    Detection detection;
    std::size_t occupancy_count = 0;
    std::uint64_t cumulative_entries = 0;
};

const char* roi_event_type_name(RoiEventType type) noexcept;
const char* roi_event_state_name(RoiEventType type) noexcept;
const char* roi_event_reason(RoiEventType type) noexcept;

class RoiEventMachine {
public:
    using Clock = std::chrono::steady_clock;

    explicit RoiEventMachine(std::chrono::milliseconds dwell_threshold = std::chrono::seconds(5),
                             std::size_t debounce_frames = 2);

    std::vector<RoiEventTransition> update(const std::vector<RoiObservation>& observations,
                                           const std::set<std::uint64_t>& active_track_ids,
                                           Clock::time_point now);

    std::size_t occupancy_count() const noexcept;
    std::uint64_t cumulative_entries() const noexcept { return cumulative_entries_; }

private:
    struct TrackState {
        bool inside = false;
        bool dwell_emitted = false;
        std::size_t consecutive_inside = 0;
        std::size_t consecutive_outside = 0;
        Clock::time_point entered_at{};
    };

    RoiEventTransition transition(std::uint64_t track_id, RoiEventType type,
                                  const Detection& detection) const;

    std::chrono::milliseconds dwell_threshold_;
    std::size_t debounce_frames_;
    std::uint64_t cumulative_entries_ = 0;
    std::map<std::uint64_t, TrackState> tracks_;
};

}  // namespace edge
