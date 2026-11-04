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
    std::cout << "core tests passed\n";
    return 0;
}
