#include "multistream.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include "bounded_queue.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TempJson {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("rk-edge-multistream-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    ~TempJson() { std::error_code error; std::filesystem::remove(path, error); }
    void write(const std::string& text) {
        std::ofstream output(path, std::ios::binary);
        output << text;
        if (!output) throw std::runtime_error("failed to write test config");
    }
};

void config_round_trip() {
    TempJson file;
    file.write(R"({"cameras":[
      {"id":"north-01","source":"rtsp://user:pass@10.0.0.1/live","weight":2,"max_fps":5,"roi":[0.1,0.2,0.5,0.6],"tasks":["ppe","roi"]},
      {"id":"local_02","source":"samples/motion.mp4","enabled":false,"queue_capacity":2}
    ]})");
    const auto config = edge::load_multistream_config(file.path.string());
    require(config.cameras.size() == 2, "camera count mismatch");
    require(config.cameras[0].id == "north-01", "camera id mismatch");
    require(config.cameras[0].weight == 2 && config.cameras[0].max_fps == 5,
            "camera scheduling options mismatch");
    require(config.cameras[0].tasks.size() == 2, "camera tasks mismatch");
    require(config.cameras[0].roi_x == 0.1 && config.cameras[0].roi_height == 0.6,
            "camera ROI mismatch");
    require(!config.cameras[1].enabled && config.cameras[1].queue_capacity == 2,
            "disabled camera config mismatch");
    require(edge::redact_source(config.cameras[0].source) == "rtsp://10.0.0.1",
            "RTSP source credentials were not redacted");
}

void reject_bad_configs() {
    TempJson file;
    file.write(R"({"cameras":[{"id":"x","source":"a"},{"id":"x","source":"b"}]})");
    bool threw = false;
    try { (void)edge::load_multistream_config(file.path.string()); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "duplicate camera ids were accepted");
    file.write(R"({"cameras":[{"id":"x","source":"a","tasks":["fire"]}]})");
    threw = false;
    try { (void)edge::load_multistream_config(file.path.string()); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "unready algorithm task was accepted");
    file.write(R"({"cameras":[{"id":"x","source":"a","roi":[0,0,2,1]}]})");
    threw = false;
    try { (void)edge::load_multistream_config(file.path.string()); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "out-of-frame ROI was accepted");
    file.write(R"({"cameras":[{"id":"x","source":"a","max_fps":1e}]})");
    threw = false;
    try { (void)edge::load_multistream_config(file.path.string()); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "malformed JSON number was accepted");
}

void weighted_fairness() {
    edge::WeightedFairScheduler scheduler({1, 2, 1});
    const std::vector<bool> ready{true, true, true};
    int counts[3] = {0, 0, 0};
    for (int frame = 0; frame < 400; ++frame) {
        const int camera = scheduler.select(ready);
        require(camera >= 0, "scheduler returned no camera");
        ++counts[camera];
    }
    require(counts[0] == 100 && counts[1] == 200 && counts[2] == 100,
            "scheduler did not preserve configured weights");
    require(scheduler.select({false, false, false}) == -1,
            "scheduler selected an empty camera");
    require(scheduler.select({false, true, false}) == 1,
            "ready camera was blocked by empty queues");
}

void bounded_queue_nonblocking_pop() {
    edge::BoundedQueue<int> queue(2);
    int value = 0;
    require(!queue.try_pop(value), "empty queue unexpectedly produced an item");
    queue.push(7);
    require(queue.try_pop(value) && value == 7, "try_pop failed to consume queued item");
    queue.close();
    require(queue.closed(), "closed state not observable");
}

}  // namespace

int main() {
    try {
        config_round_trip();
        reject_bad_configs();
        weighted_fairness();
        bounded_queue_nonblocking_pop();
        std::cout << "multistream tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multistream test failed: " << error.what() << '\n';
        return 1;
    }
}
