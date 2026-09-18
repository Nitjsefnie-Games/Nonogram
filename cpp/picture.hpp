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
