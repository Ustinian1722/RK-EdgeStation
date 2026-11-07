#include "ppe_events.hpp"
#include "tracker.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FrameInput {
    std::uint64_t index = 0;
    double pts_ms = 0.0;
    int width = 0;
    int height = 0;
    std::vector<edge::Detection> detections;
};

const char* label_for(int class_id) {
    static const char* labels[] = {
        "helmet", "gloves", "vest", "boots", "goggles", "Person",
        "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest"};
    return class_id >= 0 && class_id < static_cast<int>(sizeof(labels) / sizeof(labels[0]))
        ? labels[class_id] : "unknown";
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    for (std::string field; std::getline(stream, field, ',');) fields.push_back(field);
    return fields;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            std::cerr << "usage: replay_ppe_prediction_events INPUT.csv "
                         "[confirmation_frames=3] [cooldown_ms=30000]\n";
            return 2;
        }
        const auto confirmation_frames = argc >= 3 ? std::stoul(argv[2]) : 3UL;
        const auto cooldown_ms = argc >= 4 ? std::stoll(argv[3]) : 30000LL;
        edge::PPEEventConfig config;
        config.confirmation_frames = confirmation_frames;
        config.cooldown = std::chrono::milliseconds(cooldown_ms);
        edge::PPEEventMachine event_machine(config);
        edge::IoUTracker tracker;

        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("cannot open event replay CSV");
        std::map<std::uint64_t, FrameInput> frames;
        std::size_t source_detection_count = 0;
        for (std::string line; std::getline(input, line);) {
            if (line.empty()) continue;
            const auto fields = split_csv_row(line);
            if (fields.size() != 10) throw std::runtime_error("event replay CSV row must have 10 columns");
            const auto frame_index = static_cast<std::uint64_t>(std::stoull(fields[0]));
            auto& frame = frames[frame_index];
            frame.index = frame_index;
            frame.pts_ms = std::stod(fields[1]);
            frame.width = std::stoi(fields[2]);
            frame.height = std::stoi(fields[3]);
            const int class_id = std::stoi(fields[4]);
            if (class_id < 0) continue;
            edge::Detection detection;
            detection.class_id = class_id;
            detection.confidence = std::stof(fields[5]);
            detection.left = static_cast<int>(std::lround(std::stod(fields[6])));
            detection.top = static_cast<int>(std::lround(std::stod(fields[7])));
            detection.right = static_cast<int>(std::lround(std::stod(fields[8])));
            detection.bottom = static_cast<int>(std::lround(std::stod(fields[9])));
            detection.label = label_for(class_id);
            frame.detections.push_back(detection);
            ++source_detection_count;
        }
        if (frames.empty()) throw std::runtime_error("event replay CSV has no frames");

        const auto time_origin = edge::PPEEventMachine::Clock::time_point{};
        std::uint64_t event_count = 0;
        for (const auto& entry : frames) {
            const auto& frame = entry.second;
            if (frame.width <= 0 || frame.height <= 0) {
                throw std::runtime_error("event replay frame dimensions must be positive");
            }
            const auto tracked = tracker.update(frame.detections);
            const auto events = event_machine.update(
                tracked, tracker.active_ids(), frame.width, frame.height,
                time_origin + std::chrono::milliseconds(static_cast<std::int64_t>(std::llround(frame.pts_ms))));
            for (const auto& event : events) {
                ++event_count;
                std::cout << "{\"type\":\"event\",\"frame_index\":" << frame.index
                          << ",\"source_pts_ms\":" << frame.pts_ms
                          << ",\"event_type\":\"" << event.event_type
                          << "\",\"class_id\":" << event.detection.class_id
                          << ",\"label\":\"" << event.detection.label
                          << "\",\"confidence\":" << event.detection.confidence
                          << ",\"track_id\":" << event.track_id
                          << ",\"person_association\":\"" << event.person_association << "\"";
                if (event.related_track_id.has_value()) {
                    std::cout << ",\"related_track_id\":" << *event.related_track_id;
                }
                std::cout << ",\"confirmation_frames\":" << event.confirmation_frames << "}\n";
            }
        }
        std::cout << "{\"type\":\"summary\",\"frames\":" << frames.size()
                  << ",\"source_detections\":" << source_detection_count
                  << ",\"event_count\":" << event_count
                  << ",\"ppe_violation_count\":" << event_machine.ppe_violations()
                  << ",\"confirmation_frames\":" << confirmation_frames
                  << ",\"cooldown_ms\":" << cooldown_ms << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "replay_ppe_prediction_events: " << error.what() << '\n';
        return 1;
    }
}
