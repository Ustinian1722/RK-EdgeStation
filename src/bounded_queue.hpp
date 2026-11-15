#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace edge {

enum class QueuePolicy { Latest, Fifo };
enum class QueuePushResult { Queued, ReplacedOldest, DroppedNewest, Closed };

inline const char* queue_policy_name(QueuePolicy policy) noexcept {
    switch (policy) {
        case QueuePolicy::Latest: return "latest";
        case QueuePolicy::Fifo: return "fifo";
    }
    return "unknown";
}

// Latest evicts the oldest pending item; FIFO preserves pending order and drops
// each new item while full. Both policies remain bounded and non-blocking.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity, QueuePolicy policy = QueuePolicy::Latest)
        : capacity_(capacity), policy_(policy) {
        if (capacity_ == 0) {
            throw std::invalid_argument("queue capacity must be positive");
        }
    }

    QueuePushResult push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return QueuePushResult::Closed;
        }
        if (items_.size() == capacity_) {
            ++dropped_;
            if (policy_ == QueuePolicy::Fifo) {
                return QueuePushResult::DroppedNewest;
            }
            items_.pop_front();
            items_.push_back(std::move(value));
            ready_.notify_one();
            return QueuePushResult::ReplacedOldest;
        }
        items_.push_back(std::move(value));
        ready_.notify_one();
        return QueuePushResult::Queued;
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return false;
        }
        value = std::move(items_.front());
        items_.pop_front();
        return true;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (items_.empty()) return false;
        value = std::move(items_.front());
        items_.pop_front();
        return true;
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        ready_.notify_all();
    }

    std::size_t dropped_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    std::size_t capacity() const noexcept { return capacity_; }
    QueuePolicy policy() const noexcept { return policy_; }

private:
    const std::size_t capacity_;
    const QueuePolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> items_;
    std::size_t dropped_ = 0;
    bool closed_ = false;
};

}  // namespace edge
