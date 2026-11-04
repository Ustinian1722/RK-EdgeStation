#include "tracker.hpp"

#include <algorithm>
#include <stdexcept>

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
    std::vector<bool> matched(tracks_.size(), false);
    std::vector<TrackState> next_tracks;
    std::vector<TrackedDetection> current;
    next_tracks.reserve(tracks_.size() + detections.size());
    current.reserve(detections.size());

    for (const auto& detection : detections) {
        std::size_t best_index = tracks_.size();
        float best_iou = match_threshold_;
        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            if (matched[i] || tracks_[i].detection.class_id != detection.class_id) {
                continue;
            }
            const float overlap = iou(tracks_[i].detection, detection);
            if (overlap >= best_iou) {
                best_iou = overlap;
                best_index = i;
            }
        }

        if (best_index != tracks_.size()) {
            auto track = tracks_[best_index];
            matched[best_index] = true;
            track.detection = detection;
            track.missed = 0;
            ++track.age;
            current.push_back({track.id, detection, false});
            next_tracks.push_back(track);
        } else {
            TrackState track{next_id_++, detection, 0, 1};
            current.push_back({track.id, detection, true});
            next_tracks.push_back(track);
        }
    }

    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        if (matched[i]) {
            continue;
        }
        auto track = tracks_[i];
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
