#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// FIFO queue of line indices backed by a vector + head cursor. Each index is
// enqueued at most once between drains (dirty-flag dedup) and a queue only ever
// grows or only drains at any moment (a row drain enqueues columns, never its
// own rows), so the buffer resets to empty on full drain. Same FIFO order as
// std::deque without per-op block (de)allocation.
class FifoQueue {
public:
    void push_back(int v) { buf_.push_back(v); }
    bool empty() const { return head_ >= buf_.size(); }
    int front() const { return buf_[head_]; }
    void pop_front() {
        ++head_;
        if (head_ >= buf_.size()) { buf_.clear(); head_ = 0; }
    }

private:
    std::vector<int> buf_;
    std::size_t head_ = 0;
};

class Picture {
public:
    Picture(int height, int width);
    ~Picture();

    void mark_row_dirty(int row);
    void mark_col_dirty(int col);
    bool has_dirty() const;

    std::int8_t get_pixel(int row, int col) const;
    void set_pixel(int row, int col, std::int8_t value);

    std::vector<std::int8_t> get_row(int row) const;
    std::vector<std::int8_t> get_col(int col) const;

    bool is_solved() const;

    Picture copy() const;

    int height() const { return height_; }
    int width() const { return width_; }

    std::vector<std::int8_t> pixels;
    std::vector<std::uint8_t> row_dirty;
    std::vector<std::uint8_t> col_dirty;
    FifoQueue row_queue;
    FifoQueue col_queue;
    int unknown_count;

private:
    int height_;
    int width_;
};
