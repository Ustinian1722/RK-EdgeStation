#include "bounded_queue.hpp"
#include "tracker.hpp"

#include <iostream>

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
        edge::BoundedLatestQueue<int> queue(1);
        CHECK(queue.push(10));
        CHECK(queue.push(20));
        CHECK(queue.dropped_count() == 1);
        CHECK(queue.size() == 1);
        int value = 0;
        CHECK(queue.pop(value) && value == 20);
        queue.close();
        CHECK(!queue.pop(value));
        CHECK(!queue.push(30));
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
    std::cout << "core tests passed\n";
    return 0;
}
