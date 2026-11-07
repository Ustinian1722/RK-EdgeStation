#include "ppe_events.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace edge {
namespace {

constexpr int kPersonClass = 5;
constexpr int kForkliftClass = 10;
constexpr int kHelmetClass = 0;
constexpr int kVestClass = 2;
constexpr int kNoHelmetClass = 6;
constexpr int kNoVestClass = 11;

bool is_target_ppe_violation(int class_id) {
    return class_id == kNoHelmetClass || class_id == kNoVestClass;
}

std::string pair_key(std::uint64_t forklift_id, std::uint64_t person_id) {
    return "forklift_person:" + std::to_string(forklift_id) + ":" + std::to_string(person_id);
}

std::optional<std::uint64_t> associated_person_track(
    const Detection& violation, const std::vector<TrackedDetection>& detections) {
    const float center_x = (violation.left + violation.right) * 0.5F;
    const float center_y = (violation.top + violation.bottom) * 0.5F;
    const TrackedDetection* best = nullptr;
    long long best_area = 0;
    for (const auto& candidate : detections) {
        const auto& person = candidate.detection;
        if (person.class_id != kPersonClass || center_x < person.left || center_x >= person.right ||
            center_y < person.top || center_y >= person.bottom) {
            continue;
        }
        const long long area = static_cast<long long>(person.right - person.left) *
                               static_cast<long long>(person.bottom - person.top);
        if (area > 0 && (best == nullptr || area < best_area)) {
            best = &candidate;
            best_area = area;
        }
    }
    if (best == nullptr) return std::nullopt;
    return best->track_id;
}

}  // namespace

bool NormalizedZone::contains(const Detection& detection, int frame_width,
                              int frame_height) const noexcept {
    if (!enabled() || frame_width <= 0 || frame_height <= 0) return false;
    const double center_x = (static_cast<double>(detection.left) + detection.right) *
                            0.5 / frame_width;
    const double center_y = (static_cast<double>(detection.top) + detection.bottom) *
                            0.5 / frame_height;
    return center_x >= x && center_x <= x + width &&
           center_y >= y && center_y <= y + height;
}

PPEEventMachine::PPEEventMachine(PPEEventConfig config) : config_(std::move(config)) {
    if (config_.confirmation_frames == 0 || config_.confirmation_frames > 10000 ||
        config_.cooldown.count() < 0 || config_.conflict_dwell.count() < 0) {
        throw std::invalid_argument("PPE confirmation and cooldown settings must be bounded and non-negative");
    }
    const auto valid_zone = [](const NormalizedZone& zone) {
        return zone.x >= 0.0 && zone.y >= 0.0 && zone.width >= 0.0 && zone.height >= 0.0 &&
               zone.x + zone.width <= 1.0 && zone.y + zone.height <= 1.0 &&
               ((zone.width == 0.0 && zone.height == 0.0) || zone.enabled());
    };
    if (!valid_zone(config_.danger_zone) || !valid_zone(config_.conflict_zone)) {
        throw std::invalid_argument("safety zones must be disabled or lie within normalized frame coordinates");
    }
}

void PPEEventMachine::observe(const std::string& key, std::uint64_t identity_track_id,
                              const PPEEventTransition& candidate,
                              std::chrono::milliseconds dwell, Clock::time_point now,
                              std::vector<PPEEventTransition>& events) {
    auto& state = states_[key];
    if (state.consecutive_frames == 0) state.first_seen = now;
    state.track_id = identity_track_id;
    state.related_track_id = candidate.related_track_id;
    state.last_seen_update = update_index_;
    ++state.consecutive_frames;

    const bool confirmed = state.consecutive_frames >= config_.confirmation_frames &&
                           state.first_seen.has_value() && now - *state.first_seen >= dwell;
    const bool cooldown_elapsed = !state.last_emitted.has_value() ||
                                  now - *state.last_emitted >= config_.cooldown;
    if (confirmed && cooldown_elapsed) {
        auto event = candidate;
        event.confirmation_frames = state.consecutive_frames;
        events.push_back(std::move(event));
        state.last_emitted = now;
        ++total_events_;
        if (candidate.event_type == "ppe_violation") ++ppe_violations_;
        else if (candidate.event_type == "danger_zone_intrusion") ++danger_zone_intrusions_;
        else if (candidate.event_type == "forklift_person_cooccurrence") ++forklift_person_cooccurrences_;
    }
}

std::vector<PPEEventTransition> PPEEventMachine::update(
    const std::vector<TrackedDetection>& detections,
    const std::set<std::uint64_t>& active_track_ids,
    int frame_width, int frame_height, Clock::time_point now) {
    if (frame_width <= 0 || frame_height <= 0) {
        throw std::invalid_argument("frame dimensions must be positive for PPE event rules");
    }
    ++update_index_;
    std::vector<PPEEventTransition> events;
    events.reserve(detections.size());

    struct PersonEquipment {
        bool helmet = false;
        bool vest = false;
        bool violation = false;
    };
    std::map<std::uint64_t, PersonEquipment> equipment_by_person;
    for (const auto& item : detections) {
        if (item.detection.class_id == kPersonClass) {
            equipment_by_person.try_emplace(item.track_id);
        }
    }
    for (const auto& item : detections) {
        const int class_id = item.detection.class_id;
        if (class_id != kHelmetClass && class_id != kVestClass &&
            !is_target_ppe_violation(class_id)) continue;
        const auto person_id = associated_person_track(item.detection, detections);
        if (!person_id.has_value()) continue;
        auto& equipment = equipment_by_person[*person_id];
        if (class_id == kHelmetClass) equipment.helmet = true;
        else if (class_id == kVestClass) equipment.vest = true;
        else equipment.violation = true;
    }
    person_status_counts_ = {};
    for (const auto& item : detections) {
        if (item.detection.class_id != kPersonClass) continue;
        const auto evidence = equipment_by_person[item.track_id];
        const int status = evidence.violation ? 2 :
                           (evidence.helmet && evidence.vest ? 1 : 0);
        auto& state = person_status_states_[item.track_id];
        if (state.status == status) {
            ++state.consecutive_frames;
        } else {
            state.status = status;
            state.consecutive_frames = 1;
        }
        state.last_seen_update = update_index_;
        if (state.consecutive_frames < config_.confirmation_frames) continue;
        if (status == 2) ++person_status_counts_.violation_people;
        else if (status == 1) ++person_status_counts_.compliant_people;
        else ++person_status_counts_.unknown_people;
    }

    for (const auto& item : detections) {
        const auto& detection = item.detection;
        if (is_target_ppe_violation(detection.class_id)) {
            const auto person_id = associated_person_track(detection, detections);
            PPEEventTransition event;
            event.event_type = "ppe_violation";
            event.severity = "high";
            event.reason = "ppe_violation_persisted";
            event.person_association = person_id.has_value() ? "matched" : "unknown";
            event.ppe_status = "violation";
            event.track_id = item.track_id;
            event.related_track_id = person_id;
            event.detection = detection;
            const auto identity_track_id = person_id.value_or(item.track_id);
            const std::string event_key = "ppe:" + std::to_string(identity_track_id) + ":" +
                                          std::to_string(detection.class_id);
            observe(event_key, identity_track_id, event,
                    std::chrono::milliseconds::zero(), now, events);
        }

        if (config_.danger_zone.enabled() &&
            (detection.class_id == kPersonClass || detection.class_id == kForkliftClass) &&
            config_.danger_zone.contains(detection, frame_width, frame_height)) {
            PPEEventTransition event;
            event.event_type = "danger_zone_intrusion";
            event.severity = detection.class_id == kForkliftClass ? "high" : "medium";
            event.reason = "configured_zone_presence_persisted";
            event.person_association = "not_applicable";
            event.track_id = item.track_id;
            event.detection = detection;
            observe("zone:" + std::to_string(item.track_id), item.track_id, event,
                    std::chrono::milliseconds::zero(), now, events);
        }
    }

    if (config_.conflict_zone.enabled()) {
        for (const auto& forklift : detections) {
            const auto& forklift_detection = forklift.detection;
            if (forklift_detection.class_id != kForkliftClass ||
                !config_.conflict_zone.contains(forklift_detection, frame_width, frame_height)) {
                continue;
            }
            for (const auto& person : detections) {
                const auto& person_detection = person.detection;
                if (person_detection.class_id != kPersonClass ||
                    !config_.conflict_zone.contains(person_detection, frame_width, frame_height)) {
                    continue;
                }
                PPEEventTransition event;
                event.event_type = "forklift_person_cooccurrence";
                event.severity = "high";
                event.reason = "person_and_forklift_share_configured_conflict_zone";
                event.person_association = "not_applicable";
                event.track_id = forklift.track_id;
                event.related_track_id = person.track_id;
                event.detection = forklift_detection;
                observe(pair_key(forklift.track_id, person.track_id), forklift.track_id, event,
                        config_.conflict_dwell, now, events);
            }
        }
    }

    for (auto it = states_.begin(); it != states_.end();) {
        const auto& state = it->second;
        const bool primary_active = active_track_ids.count(state.track_id) != 0;
        const bool related_active = !state.related_track_id.has_value() ||
                                    active_track_ids.count(*state.related_track_id) != 0;
        const bool stale = update_index_ > state.last_seen_update + 10;
        const bool ppe_cooldown_active = it->first.rfind("ppe:", 0) == 0 &&
                                         state.last_emitted.has_value() &&
                                         now - *state.last_emitted < config_.cooldown;
        if (!primary_active || !related_active || (stale && !ppe_cooldown_active)) {
            it = states_.erase(it);
        } else if (state.last_seen_update != update_index_) {
            it->second.consecutive_frames = 0;
            it->second.first_seen.reset();
            ++it;
        } else {
            ++it;
        }
    }
    for (auto it = person_status_states_.begin(); it != person_status_states_.end();) {
        if (!active_track_ids.count(it->first) || update_index_ > it->second.last_seen_update + 10) {
            it = person_status_states_.erase(it);
        } else if (it->second.last_seen_update != update_index_) {
            it->second.consecutive_frames = 0;
            ++it;
        } else {
            ++it;
        }
    }
    return events;
}

void PPEEventMachine::reset_stream() noexcept {
    states_.clear();
    person_status_states_.clear();
    person_status_counts_ = {};
}

}  // namespace edge
