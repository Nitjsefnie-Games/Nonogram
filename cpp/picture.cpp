#include "picture.hpp"
#include "types.hpp"

Picture::Picture(int height, int width)
    : height_(height), width_(width) {}

Picture::~Picture() = default;

void Picture::mark_row_dirty(int row) {
    (void)row;
}

void Picture::mark_col_dirty(int col) {
    (void)col;
}

bool Picture::has_dirty() const {
    return false;
}

std::int8_t Picture::get_pixel(int row, int col) const {
    (void)row;
    (void)col;
    return UNKNOWN;
}

void Picture::set_pixel(int row, int col, std::int8_t value) {
    (void)row;
    (void)col;
    (void)value;
}

std::vector<std::int8_t> Picture::get_row(int row) const {
    (void)row;
    return std::vector<std::int8_t>(width_, UNKNOWN);
}

std::vector<std::int8_t> Picture::get_col(int col) const {
    (void)col;
    return std::vector<std::int8_t>(height_, UNKNOWN);
}

bool Picture::is_solved() const {
    return false;
}

Picture Picture::copy() const {
    return Picture(height_, width_);
}
