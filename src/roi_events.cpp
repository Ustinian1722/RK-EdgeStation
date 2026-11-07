#include "roi_events.hpp"

#include <algorithm>
#include <stdexcept>

namespace edge {

const char* roi_event_type_name(RoiEventType type) noexcept {
    switch (type) {
        case RoiEventType::Enter: return "enter";
        case RoiEventType::Leave: return "leave";
        case RoiEventType::Dwell: return "dwell";
    }
    return "unknown";
}

const char* roi_event_state_name(RoiEventType type) noexcept {
    return type == RoiEventType::Leave ? "closed" : "active";
}

const char* roi_event_reason(RoiEventType type) noexcept {
    switch (type) {
        case RoiEventType::Enter: return "roi_enter";
        case RoiEventType::Leave: return "roi_exit";
        case RoiEventType::Dwell: return "dwell_threshold";
    }
    return "unknown";
}

RoiEventMachine::RoiEventMachine(std::chrono::milliseconds dwell_threshold,
                                 std::size_t debounce_frames, int person_class_id)
    : dwell_threshold_(dwell_threshold), debounce_frames_(debounce_frames),
      person_class_id_(person_class_id) {
    if (dwell_threshold_.count() <= 0 || debounce_frames_ == 0 || debounce_frames_ > 10000 ||
        person_class_id_ < 0) {
        throw std::invalid_argument("dwell threshold and ROI debounce frames must be positive and bounded");
    }
}

RoiEventTransition RoiEventMachine::transition(std::uint64_t track_id, RoiEventType type,
                                               const Detection& detection) const {
    return {track_id, type, detection, occupancy_count(), cumulative_entries_};
}

std::vector<RoiEventTransition> RoiEventMachine::update(
    const std::vector<RoiObservation>& observations,
    const std::set<std::uint64_t>& active_track_ids,
    Clock::time_point now) {
    std::vector<RoiEventTransition> events;
    events.reserve(observations.size());

    for (const auto& observation : observations) {
        if (observation.detection.class_id != person_class_id_) continue;
        auto& state = tracks_[observation.track_id];
        if (observation.inside) {
            state.consecutive_outside = 0;
            if (!state.inside) {
                state.consecutive_inside = std::min(debounce_frames_, state.consecutive_inside + 1);
                if (state.consecutive_inside >= debounce_frames_) {
                    state.inside = true;
                    state.entered_at = now;
                    state.dwell_emitted = false;
                    ++cumulative_entries_;
                    events.push_back(transition(observation.track_id, RoiEventType::Enter,
                                                 observation.detection));
                }
            } else if (!state.dwell_emitted && now - state.entered_at >= dwell_threshold_) {
                state.dwell_emitted = true;
                events.push_back(transition(observation.track_id, RoiEventType::Dwell,
                                             observation.detection));
            }
        } else {
            state.consecutive_inside = 0;
            if (state.inside) {
                state.consecutive_outside = std::min(debounce_frames_, state.consecutive_outside + 1);
                if (state.consecutive_outside >= debounce_frames_) {
                    state.inside = false;
                    state.dwell_emitted = false;
                    events.push_back(transition(observation.track_id, RoiEventType::Leave,
                                                 observation.detection));
                }
            } else {
                state.consecutive_outside = std::min(debounce_frames_, state.consecutive_outside + 1);
            }
        }
    }

    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (active_track_ids.count(it->first) == 0) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
    return events;
}

std::size_t RoiEventMachine::occupancy_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        tracks_.begin(), tracks_.end(), [](const auto& entry) { return entry.second.inside; }));
}

}  // namespace edge
