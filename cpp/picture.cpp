#include "picture.hpp"
#include "types.hpp"

#include <algorithm>

Picture::Picture(int height, int width)
    : pixels(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), UNKNOWN),
      row_dirty(static_cast<std::size_t>(height), 1),
      col_dirty(static_cast<std::size_t>(width), 1),
      row_queue(static_cast<std::size_t>(height)),
      col_queue(static_cast<std::size_t>(width)),
      unknown_count(height * width),
      key_words((std::max(height, width) + 31) / 32),
      row_keys(static_cast<std::size_t>(height) * key_words, 0),
      col_keys(static_cast<std::size_t>(width) * key_words, 0),
      height_(height),
      width_(width) {
    for (int i = 0; i < height; ++i) {
        row_queue.push_back(i);
    }
    for (int j = 0; j < width; ++j) {
        col_queue.push_back(j);
    }
    // Every cell starts UNKNOWN (digit 2).
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            row_keys[static_cast<std::size_t>(r) * key_words + (c >> 5)] |=
                static_cast<std::uint64_t>(UNKNOWN) << (2 * (c & 31));
            col_keys[static_cast<std::size_t>(c) * key_words + (r >> 5)] |=
                static_cast<std::uint64_t>(UNKNOWN) << (2 * (r & 31));
        }
    }
}

Picture::~Picture() = default;

bool Picture::has_dirty() const {
    return !row_queue.empty() || !col_queue.empty();
}

std::int8_t Picture::get_pixel(int row, int col) const {
    return pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col)];
}

void Picture::set_pixel(int row, int col, std::int8_t value) {
    std::size_t idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col);
    std::int8_t old = pixels[idx];
    pixels[idx] = value;
    toggle_key(row, col, static_cast<int>(old) ^ static_cast<int>(value));
    if (old == UNKNOWN && value != UNKNOWN) {
        --unknown_count;
    }
}

void Picture::set_known(int row, int col, std::int8_t val) {
    pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col)] = val;
    --unknown_count;
    toggle_key(row, col, static_cast<int>(UNKNOWN) ^ static_cast<int>(val));
}

void Picture::unset(int row, int col) {
    std::int8_t& cell = pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col)];
    toggle_key(row, col, static_cast<int>(UNKNOWN) ^ static_cast<int>(cell));
    cell = UNKNOWN;
}

std::vector<std::int8_t> Picture::get_row(int row) const {
    std::size_t start = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
    return std::vector<std::int8_t>(pixels.begin() + start, pixels.begin() + start + width_);
}

std::vector<std::int8_t> Picture::get_col(int col) const {
    std::vector<std::int8_t> result(static_cast<std::size_t>(height_));
    for (int r = 0; r < height_; ++r) {
        result[r] = pixels[static_cast<std::size_t>(r) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col)];
    }
    return result;
}

bool Picture::is_solved() const {
    return unknown_count == 0;
}

Picture Picture::copy() const {
    Picture new_pic(height_, width_);
    new_pic.pixels = pixels;
    new_pic.row_dirty = row_dirty;
    new_pic.col_dirty = col_dirty;
    new_pic.row_queue = row_queue;
    new_pic.col_queue = col_queue;
    new_pic.unknown_count = unknown_count;
    new_pic.key_words = key_words;
    new_pic.row_keys = row_keys;
    new_pic.col_keys = col_keys;
    return new_pic;
}
