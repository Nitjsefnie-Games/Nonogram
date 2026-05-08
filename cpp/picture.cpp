#include "picture.hpp"
#include "types.hpp"

Picture::Picture(int height, int width)
    : pixels(static_cast<std::size_t>(height) * static_cast<std::size_t>(width), UNKNOWN),
      row_dirty(static_cast<std::size_t>(height), 1),
      col_dirty(static_cast<std::size_t>(width), 1),
      unknown_count(height * width),
      height_(height),
      width_(width) {
    for (int i = 0; i < height; ++i) {
        row_queue.push_back(i);
    }
    for (int j = 0; j < width; ++j) {
        col_queue.push_back(j);
    }
}

Picture::~Picture() = default;

void Picture::mark_row_dirty(int row) {
    if (!row_dirty[row]) {
        row_dirty[row] = 1;
        row_queue.push_back(row);
    }
}

void Picture::mark_col_dirty(int col) {
    if (!col_dirty[col]) {
        col_dirty[col] = 1;
        col_queue.push_back(col);
    }
}

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
    if (old == UNKNOWN && value != UNKNOWN) {
        --unknown_count;
    }
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
    new_pic.solved_rows = solved_rows;
    new_pic.solved_cols = solved_cols;
    new_pic.unknown_count = unknown_count;
    return new_pic;
}
