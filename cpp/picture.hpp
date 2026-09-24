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
    // capacity = number of lines: the dirty-flag dedup bounds the entries
    // between two resets by that, so the buffer never grows. One spare slot
    // absorbs push_back_if's unconditional write when the queue is full.
    explicit FifoQueue(std::size_t capacity = 0) : buf_(capacity + 1), tail_(0), head_(0) {}
    void push_back(int v) { buf_[tail_++] = v; }
    // Branch-free conditional enqueue: the value is always written to the
    // tail slot and the tail advances by `cond`. Whether a line is already
    // dirty is data-dependent noise to the branch predictor.
    void push_back_if(int v, bool cond) { buf_[tail_] = v; tail_ += cond; }
    bool empty() const { return head_ >= tail_; }
    int front() const { return buf_[head_]; }
    void pop_front() { ++head_; }
    // A drain that keeps its own cursor: read the buffer and bounds once,
    // walk locals, and write the head back only on an early exit. The
    // queue never grows during its own drain (a row drain enqueues columns
    // only), so the tail read at the start stays the bound.
    const int* data() const { return buf_.data(); }
    std::size_t head() const { return head_; }
    std::size_t tail() const { return tail_; }
    void set_head(std::size_t h) { head_ = h; }
    // A bulk enqueue that keeps its own cursor (write_intersection): the
    // buffer and tail read once, push_back_if's write-then-advance done on
    // locals per cell, and the tail written back once at the end.
    int* data() { return buf_.data(); }
    void set_tail(std::size_t t) { tail_ = t; }
    // Call once the queue has been drained (empty() is true) to reclaim the
    // buffer; kept out of pop_front so the drain loop pays no per-pop check.
    void reset() { tail_ = 0; head_ = 0; }

private:
    std::vector<int> buf_;
    std::size_t tail_;
    std::size_t head_;
};

class Picture {
public:
    Picture(int height, int width);
    ~Picture();

    void mark_row_dirty(int row) {
        row_queue.push_back_if(row, !row_dirty[row]);
        row_dirty[row] = 1;
    }
    void mark_col_dirty(int col) {
        col_queue.push_back_if(col, !col_dirty[col]);
        col_dirty[col] = 1;
    }
    bool has_dirty() const;

    std::int8_t get_pixel(int row, int col) const;
    void set_pixel(int row, int col, std::int8_t value);

    // Transition (row, col) from UNKNOWN to val: pixel, packed line keys and
    // unknown_count all updated. Precondition: the cell is UNKNOWN.
    void set_known(int row, int col, std::int8_t val);
    // Transition (row, col) from a known value back to UNKNOWN: pixel and
    // packed line keys updated. unknown_count is deliberately NOT adjusted --
    // every caller restores it wholesale from a snapshot.
    void unset(int row, int col);

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

    // Packed line keys: every row and every column as 2 bits per cell (digit =
    // cell value: EMPTY 0, FULL 1, UNKNOWN 2), cells beyond the line length 0.
    // row_keys[r * key_words + w] holds cells 32w..32w+31 of row r; col_keys
    // likewise per column. Maintained incrementally by set_known / unset /
    // set_pixel, so the line cache can key a lookup on key_words machine words
    // instead of hashing and comparing the line's bytes.
    int key_words;
    std::vector<std::uint64_t> row_keys;
    std::vector<std::uint64_t> col_keys;

private:
    // XOR (old_digit ^ new_digit) into both keys covering (row, col).
    void toggle_key(int row, int col, int x) {
        row_keys[static_cast<std::size_t>(row) * key_words + (col >> 5)] ^=
            static_cast<std::uint64_t>(x) << (2 * (col & 31));
        col_keys[static_cast<std::size_t>(col) * key_words + (row >> 5)] ^=
            static_cast<std::uint64_t>(x) << (2 * (row & 31));
    }

    int height_;
    int width_;
};
