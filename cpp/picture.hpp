#pragma once
#include <cstdint>
#include <vector>

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

private:
    int height_;
    int width_;
};
