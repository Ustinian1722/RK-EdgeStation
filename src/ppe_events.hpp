#pragma once

#include "tracker.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace edge {

struct NormalizedZone {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;

    bool enabled() const noexcept { return width > 0.0 && height > 0.0; }
    bool contains(const Detection& detection, int frame_width, int frame_height) const noexcept;
};

struct PPEEventConfig {
    NormalizedZone danger_zone;
    NormalizedZone conflict_zone;
    std::size_t confirmation_frames = 3;
    std::chrono::milliseconds cooldown{30000};
    std::chrono::milliseconds conflict_dwell{1000};
};

struct PPEEventTransition {
    std::string event_type;
    std::string severity;
    std::string reason;
    std::string person_association;
    std::string ppe_status;
    std::uint64_t track_id = 0;
    std::optional<std::uint64_t> related_track_id;
    Detection detection;
    std::size_t confirmation_frames = 0;
};

struct PPEStatusCounts {
    std::size_t compliant_people = 0;
    std::size_t violation_people = 0;
    std::size_t unknown_people = 0;
};

class PPEEventMachine {
public:
    using Clock = std::chrono::steady_clock;

    explicit PPEEventMachine(PPEEventConfig config = {});

    std::vector<PPEEventTransition> update(const std::vector<TrackedDetection>& detections,
                                           const std::set<std::uint64_t>& active_track_ids,
                                           int frame_width, int frame_height,
                                           Clock::time_point now);
    void reset_stream() noexcept;

    std::uint64_t total_events() const noexcept { return total_events_; }
    std::uint64_t ppe_violations() const noexcept { return ppe_violations_; }
    std::uint64_t danger_zone_intrusions() const noexcept { return danger_zone_intrusions_; }
    std::uint64_t forklift_person_cooccurrences() const noexcept { return forklift_person_cooccurrences_; }
    const PPEStatusCounts& person_status_counts() const noexcept { return person_status_counts_; }

private:
    struct RuleState {
        std::uint64_t track_id = 0;
        std::optional<std::uint64_t> related_track_id;
        std::size_t consecutive_frames = 0;
        std::optional<Clock::time_point> first_seen;
        std::optional<Clock::time_point> last_emitted;
        std::size_t last_seen_update = 0;
    };

    struct PersonStatusState {
        int status = -1;
        std::size_t consecutive_frames = 0;
        std::size_t last_seen_update = 0;
    };

    void observe(const std::string& key, std::uint64_t identity_track_id,
                 const PPEEventTransition& candidate,
                 std::chrono::milliseconds dwell, Clock::time_point now,
                 std::vector<PPEEventTransition>& events);

    PPEEventConfig config_;
    std::size_t update_index_ = 0;
    std::map<std::string, RuleState> states_;
    std::map<std::uint64_t, PersonStatusState> person_status_states_;
    PPEStatusCounts person_status_counts_;
    std::uint64_t total_events_ = 0;
    std::uint64_t ppe_violations_ = 0;
    std::uint64_t danger_zone_intrusions_ = 0;
    std::uint64_t forklift_person_cooccurrences_ = 0;
};

}  // namespace edge
