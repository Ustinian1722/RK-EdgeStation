#include "tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace edge {

IoUTracker::IoUTracker(float match_threshold, int max_missed)
    : match_threshold_(match_threshold), max_missed_(max_missed) {
    if (match_threshold_ < 0.0F || match_threshold_ > 1.0F || max_missed_ < 0) {
        throw std::invalid_argument("invalid tracker parameters");
    }
}

float IoUTracker::iou(const Detection& a, const Detection& b) {
    const int left = std::max(a.left, b.left);
    const int top = std::max(a.top, b.top);
    const int right = std::min(a.right, b.right);
    const int bottom = std::min(a.bottom, b.bottom);
    const float intersection = static_cast<float>(std::max(0, right - left)) *
                              static_cast<float>(std::max(0, bottom - top));
    const float area_a = static_cast<float>(std::max(0, a.right - a.left)) *
                         static_cast<float>(std::max(0, a.bottom - a.top));
    const float area_b = static_cast<float>(std::max(0, b.right - b.left)) *
                         static_cast<float>(std::max(0, b.bottom - b.top));
    const float union_area = area_a + area_b - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

std::vector<TrackedDetection> IoUTracker::update(const std::vector<Detection>& detections) {
    constexpr float kCenterGateFactor = 0.40F;
    constexpr float kMaximumAreaRatio = 4.0F;
    constexpr float kIouWeight = 0.45F;
    constexpr float kCenterDistanceWeight = 0.45F;
    constexpr float kAreaChangeWeight = 0.10F;

    struct Candidate {
        std::size_t track_index;
        std::size_t detection_index;
        float cost;
        std::uint64_t track_id;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(tracks_.size() * detections.size());

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        const auto& track = tracks_[track_index];
        const int elapsed = track.missed + 1;
        const int shift_x = static_cast<int>(std::lround(track.velocity_x * elapsed));
        const int shift_y = static_cast<int>(std::lround(track.velocity_y * elapsed));
        Detection predicted = track.detection;
        predicted.left += shift_x;
        predicted.right += shift_x;
        predicted.top += shift_y;
        predicted.bottom += shift_y;

        const float predicted_width = static_cast<float>(std::max(1, predicted.right - predicted.left));
        const float predicted_height = static_cast<float>(std::max(1, predicted.bottom - predicted.top));
        const float predicted_area = predicted_width * predicted_height;
        const float gate = std::max(10.0F, kCenterGateFactor * std::hypot(predicted_width, predicted_height));

        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            const auto& detection = detections[detection_index];
            if (track.detection.class_id != detection.class_id || detection.right <= detection.left ||
                detection.bottom <= detection.top) {
                continue;
            }

            const float detection_width = static_cast<float>(detection.right - detection.left);
            const float detection_height = static_cast<float>(detection.bottom - detection.top);
            const float detection_area = detection_width * detection_height;
            const float area_ratio = std::max(predicted_area, detection_area) /
                                     std::min(predicted_area, detection_area);
            if (area_ratio > kMaximumAreaRatio) {
                continue;
            }

            const float overlap = iou(predicted, detection);
            const float predicted_center_x = (predicted.left + predicted.right) * 0.5F;
            const float predicted_center_y = (predicted.top + predicted.bottom) * 0.5F;
            const float detection_center_x = (detection.left + detection.right) * 0.5F;
            const float detection_center_y = (detection.top + detection.bottom) * 0.5F;
            const float center_distance = std::hypot(predicted_center_x - detection_center_x,
                                                     predicted_center_y - detection_center_y);
            if (overlap < match_threshold_ && center_distance > gate) {
                continue;
            }

            const float normalized_distance = std::min(1.0F, center_distance / gate);
            const float normalized_area_change = std::min(1.0F, std::abs(std::log(area_ratio)) /
                                                                      std::log(kMaximumAreaRatio));
            const float cost = kIouWeight * (1.0F - overlap) + kCenterDistanceWeight * normalized_distance +
                               kAreaChangeWeight * normalized_area_change;
            candidates.push_back({track_index, detection_index, cost, track.id});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return std::tie(lhs.cost, lhs.track_id, lhs.detection_index) <
               std::tie(rhs.cost, rhs.track_id, rhs.detection_index);
    });

    std::vector<int> detection_for_track(tracks_.size(), -1);
    std::vector<int> track_for_detection(detections.size(), -1);
    for (const auto& candidate : candidates) {
        if (detection_for_track[candidate.track_index] >= 0 ||
            track_for_detection[candidate.detection_index] >= 0) {
            continue;
        }
        detection_for_track[candidate.track_index] = static_cast<int>(candidate.detection_index);
        track_for_detection[candidate.detection_index] = static_cast<int>(candidate.track_index);
    }

    std::vector<TrackState> next_tracks;
    std::vector<TrackedDetection> current;
    next_tracks.reserve(tracks_.size() + detections.size());
    current.reserve(detections.size());

    for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
        const auto& detection = detections[detection_index];
        const int track_index = track_for_detection[detection_index];
        if (track_index >= 0) {
            auto track = tracks_[static_cast<std::size_t>(track_index)];
            const int elapsed = track.missed + 1;
            const float old_center_x = (track.detection.left + track.detection.right) * 0.5F;
            const float old_center_y = (track.detection.top + track.detection.bottom) * 0.5F;
            const float new_center_x = (detection.left + detection.right) * 0.5F;
            const float new_center_y = (detection.top + detection.bottom) * 0.5F;
            track.velocity_x = (new_center_x - old_center_x) / elapsed;
            track.velocity_y = (new_center_y - old_center_y) / elapsed;
            track.detection = detection;
            track.missed = 0;
            ++track.age;
            current.push_back({track.id, detection, false});
            next_tracks.push_back(track);
        } else {
            TrackState track{next_id_++, detection, 0.0F, 0.0F, 0, 1};
            current.push_back({track.id, detection, true});
            next_tracks.push_back(track);
        }
    }

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        if (detection_for_track[track_index] >= 0) {
            continue;
        }
        auto track = tracks_[track_index];
        ++track.missed;
        if (track.missed <= max_missed_) {
            next_tracks.push_back(track);
        }
    }
    tracks_ = std::move(next_tracks);
    return current;
}

std::set<std::uint64_t> IoUTracker::active_ids() const {
    std::set<std::uint64_t> ids;
    for (const auto& track : tracks_) {
        ids.insert(track.id);
    }
    return ids;
}

}  // namespace edge
