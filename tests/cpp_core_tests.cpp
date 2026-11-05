#include "bounded_queue.hpp"
#include "ppe_events.hpp"
#include "roi_events.hpp"
#include "tracker.hpp"

#include <chrono>
#include <iostream>
#include <string>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << "check failed at line " << __LINE__ << ": "          \
                      << #condition << '\n';                                    \
            return 1;                                                           \
        }                                                                       \
    } while (false)

int main() {
    {
        edge::BoundedQueue<int> queue(1, edge::QueuePolicy::Latest);
        CHECK(queue.push(10) == edge::QueuePushResult::Queued);
        CHECK(queue.push(20) == edge::QueuePushResult::ReplacedOldest);
        CHECK(queue.dropped_count() == 1);
        CHECK(queue.size() == 1);
        CHECK(edge::queue_policy_name(queue.policy()) == std::string("latest"));
        int value = 0;
        CHECK(queue.pop(value) && value == 20);
        queue.close();
        CHECK(!queue.pop(value));
        CHECK(queue.push(30) == edge::QueuePushResult::Closed);
    }
    {
        edge::BoundedQueue<int> queue(2, edge::QueuePolicy::Fifo);
        CHECK(queue.push(10) == edge::QueuePushResult::Queued);
        CHECK(queue.push(20) == edge::QueuePushResult::Queued);
        CHECK(queue.push(30) == edge::QueuePushResult::DroppedNewest);
        CHECK(queue.dropped_count() == 1);
        CHECK(queue.size() == 2);
        CHECK(edge::queue_policy_name(queue.policy()) == std::string("fifo"));
        int value = 0;
        CHECK(queue.pop(value) && value == 10);
        CHECK(queue.push(30) == edge::QueuePushResult::Queued);
        CHECK(queue.pop(value) && value == 20);
        CHECK(queue.pop(value) && value == 30);
        queue.close();
        CHECK(!queue.pop(value));
    }
    {
        edge::IoUTracker tracker(0.3F, 2);
        edge::Detection first{0, 0.9F, 10, 10, 80, 90, "person"};
        auto result = tracker.update({first});
        CHECK(result.size() == 1 && result[0].created && result[0].track_id == 1);
        edge::Detection shifted{0, 0.8F, 12, 12, 82, 92, "person"};
        result = tracker.update({shifted});
        CHECK(result.size() == 1 && !result[0].created && result[0].track_id == 1);
        tracker.update({});
        tracker.update({});
        tracker.update({});
        CHECK(tracker.active_ids().empty());
        result = tracker.update({first});
        CHECK(result.size() == 1 && result[0].created && result[0].track_id == 2);
        tracker.reset();
        CHECK(tracker.active_ids().empty());
        result = tracker.update({first});
        CHECK(result.size() == 1 && result[0].created && result[0].track_id == 3);
    }
    {
        edge::IoUTracker tracker(0.3F, 2);
        edge::Detection first{0, 0.9F, 0, 0, 20, 80, "person"};
        edge::Detection next{0, 0.9F, 8, 0, 28, 80, "person"};
        edge::Detection after_gap{0, 0.9F, 24, 0, 44, 80, "person"};

        auto result = tracker.update({first});
        CHECK(result.size() == 1 && result[0].created && result[0].track_id == 1);
        result = tracker.update({next});
        CHECK(result.size() == 1 && !result[0].created && result[0].track_id == 1);
        CHECK(tracker.update({}).empty());
        result = tracker.update({after_gap});
        CHECK(result.size() == 1 && !result[0].created && result[0].track_id == 1);
    }
    {
        edge::IoUTracker tracker(0.3F, 2);
        edge::Detection left{0, 0.9F, 0, 0, 30, 80, "person"};
        edge::Detection right{0, 0.9F, 100, 0, 130, 80, "person"};
        auto result = tracker.update({left, right});
        CHECK(result.size() == 2 && result[0].track_id == 1 && result[1].track_id == 2);

        edge::Detection right_next{0, 0.9F, 90, 0, 120, 80, "person"};
        edge::Detection left_next{0, 0.9F, 10, 0, 40, 80, "person"};
        result = tracker.update({right_next, left_next});
        CHECK(result.size() == 2 && result[0].track_id == 2 && result[1].track_id == 1);
        CHECK(!result[0].created && !result[1].created);
    }
    {
        edge::IoUTracker tracker(0.3F, 2);
        edge::Detection person{0, 0.9F, 0, 0, 30, 80, "person"};
        edge::Detection car{2, 0.9F, 0, 0, 30, 80, "car"};
        tracker.update({person});
        const auto result = tracker.update({car});
        CHECK(result.size() == 1 && result[0].created && result[0].track_id == 2);
    }
    {
        using namespace std::chrono_literals;
        edge::RoiEventMachine events(5000ms, 2);
        const auto start = edge::RoiEventMachine::Clock::time_point{};
        const std::set<std::uint64_t> active{7};
        const edge::Detection detection{0, 0.9F, 10, 20, 50, 100, "person"};
        const edge::RoiObservation inside{7, detection, true};
        const edge::RoiObservation outside{7, detection, false};

        CHECK(events.update({inside}, active, start).empty());
        CHECK(events.occupancy_count() == 0);
        auto transitions = events.update({inside}, active, start + 100ms);
        CHECK(transitions.size() == 1 && transitions[0].type == edge::RoiEventType::Enter);
        CHECK(transitions[0].occupancy_count == 1 && transitions[0].cumulative_entries == 1);

        CHECK(events.update({outside}, active, start + 200ms).empty());
        CHECK(events.update({inside}, active, start + 300ms).empty());
        CHECK(events.occupancy_count() == 1);
        CHECK(events.update({inside}, active, start + 4s).empty());
        transitions = events.update({inside}, active, start + 5100ms);
        CHECK(transitions.size() == 1 && transitions[0].type == edge::RoiEventType::Dwell);
        CHECK(transitions[0].cumulative_entries == 1 && transitions[0].occupancy_count == 1);

        CHECK(events.update({outside}, active, start + 5200ms).empty());
        transitions = events.update({outside}, active, start + 5300ms);
        CHECK(transitions.size() == 1 && transitions[0].type == edge::RoiEventType::Leave);
        CHECK(transitions[0].occupancy_count == 0 && transitions[0].cumulative_entries == 1);
        CHECK(events.cumulative_entries() == 1);
        events.update({inside}, active, start + 6s);
        transitions = events.update({inside}, active, start + 6100ms);
        CHECK(transitions.size() == 1 && transitions[0].type == edge::RoiEventType::Enter);
        CHECK(transitions[0].cumulative_entries == 2 && transitions[0].occupancy_count == 1);
        events.reset_stream();
        CHECK(events.occupancy_count() == 0);
        CHECK(events.cumulative_entries() == 2);
    }
    {
        using namespace std::chrono_literals;
        edge::RoiEventMachine events(1000ms, 2);
        const auto start = edge::RoiEventMachine::Clock::time_point{};
        const std::set<std::uint64_t> active{1};
        const edge::Detection detection{0, 0.9F, 0, 0, 20, 80, "person"};
        const edge::RoiObservation inside{1, detection, true};
        CHECK(events.update({inside}, active, start).empty());
        CHECK(events.update({inside}, active, start + 10ms).size() == 1);
        CHECK(events.update({}, active, start + 20ms).empty());
        CHECK(events.occupancy_count() == 1);
        CHECK(events.update({}, {}, start + 30ms).empty());
        CHECK(events.occupancy_count() == 0);
    }
    {
        using namespace std::chrono_literals;
        edge::PPEEventConfig config;
        config.danger_zone = {0.4, 0.0, 0.6, 1.0};
        config.conflict_zone = {0.4, 0.0, 0.6, 1.0};
        config.confirmation_frames = 2;
        config.cooldown = 1000ms;
        config.conflict_dwell = 500ms;
        edge::PPEEventMachine events(config);
        const auto start = edge::PPEEventMachine::Clock::time_point{};
        const edge::Detection person_detection{5, 0.95F, 40, 10, 75, 95, "Person"};
        const edge::Detection forklift_detection{10, 0.91F, 65, 25, 99, 90, "forklift"};
        const edge::Detection violation_detection{6, 0.87F, 50, 13, 60, 27, "no_helmet"};
        const std::vector<edge::TrackedDetection> detections{
            {11, person_detection, true},
            {22, forklift_detection, true},
            {33, violation_detection, true},
        };
        const std::set<std::uint64_t> active{11, 22, 33};

        CHECK(events.update(detections, active, 100, 100, start).empty());
        auto transitions = events.update(detections, active, 100, 100, start + 100ms);
        CHECK(transitions.size() == 3);
        bool found_violation = false;
        for (const auto& event : transitions) {
            if (event.event_type == "ppe_violation") {
                found_violation = true;
                CHECK(event.track_id == 33);
                CHECK(event.related_track_id == std::optional<std::uint64_t>(11));
                CHECK(event.person_association == "matched");
                CHECK(event.ppe_status == "violation");
                CHECK(event.detection.label == std::string("no_helmet"));
            }
        }
        CHECK(found_violation);
        CHECK(events.ppe_violations() == 1);
        CHECK(events.danger_zone_intrusions() == 2);
        CHECK(events.forklift_person_cooccurrences() == 0);

        transitions = events.update(detections, active, 100, 100, start + 600ms);
        CHECK(transitions.size() == 1);
        CHECK(transitions[0].event_type == "forklift_person_cooccurrence");
        CHECK(transitions[0].track_id == 22);
        CHECK(transitions[0].related_track_id == std::optional<std::uint64_t>(11));
        CHECK(transitions[0].reason.find("share_configured_conflict_zone") != std::string::npos);
        CHECK(events.forklift_person_cooccurrences() == 1);

        events.reset_stream();
        CHECK(events.total_events() == 4);
    }
    {
        using namespace std::chrono_literals;
        edge::PPEEventConfig config;
        config.confirmation_frames = 2;
        edge::PPEEventMachine events(config);
        const auto start = edge::PPEEventMachine::Clock::time_point{};
        const edge::Detection person{5, 0.95F, 0, 0, 100, 120, "Person"};
        const edge::Detection helmet{0, 0.91F, 35, 8, 65, 35, "helmet"};
        const edge::Detection vest{2, 0.92F, 25, 40, 75, 95, "vest"};
        const edge::Detection out_of_scope_no_goggle{7, 0.91F, 38, 36, 62, 50, "no_goggle"};
        const std::set<std::uint64_t> safe_active{1, 2, 3, 4};
        const std::vector<edge::TrackedDetection> safe{
            {1, person, true}, {2, helmet, true}, {3, vest, true},
            {4, out_of_scope_no_goggle, true},
        };
        CHECK(events.update(safe, safe_active, 100, 120, start).empty());
        CHECK(events.person_status_counts().compliant_people == 0);
        CHECK(events.update(safe, safe_active, 100, 120, start + 100ms).empty());
        CHECK(events.person_status_counts().compliant_people == 1);
        CHECK(events.person_status_counts().unknown_people == 0);

        const std::vector<edge::TrackedDetection> partial{{1, person, true}, {2, helmet, true}};
        const std::set<std::uint64_t> partial_active{1, 2};
        events.update(partial, partial_active, 100, 120, start + 200ms);
        CHECK(events.person_status_counts().unknown_people == 0);
        events.update(partial, partial_active, 100, 120, start + 300ms);
        CHECK(events.person_status_counts().unknown_people == 1);
        CHECK(events.person_status_counts().compliant_people == 0);

        const edge::Detection no_helmet{6, 0.90F, 35, 8, 65, 35, "no_helmet"};
        const std::vector<edge::TrackedDetection> violation{{1, person, true}, {2, no_helmet, true}};
        const std::set<std::uint64_t> violation_active{1, 2};
        events.update(violation, violation_active, 100, 120, start + 400ms);
        CHECK(events.person_status_counts().violation_people == 0);
        events.update(violation, violation_active, 100, 120, start + 500ms);
        CHECK(events.person_status_counts().violation_people == 1);
        CHECK(events.person_status_counts().unknown_people == 0);
    }
    std::cout << "core tests passed\n";
    return 0;
}
