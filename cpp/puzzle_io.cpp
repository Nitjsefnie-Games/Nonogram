#include "puzzle_io.hpp"

#include <cctype>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Strip leading and trailing ASCII whitespace (mirrors Python str.strip()).
std::string strip(const std::string& s) {
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

// Parse a whitespace-separated list of ints. Throws on malformed tokens.
std::vector<int> parse_ints(const std::string& line) {
    std::vector<int> out;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        try {
            std::size_t consumed = 0;
            int v = std::stoi(tok, &consumed);
            if (consumed != tok.size()) {
                throw std::invalid_argument("trailing garbage");
            }
            out.push_back(v);
        } catch (const std::exception&) {
            throw std::runtime_error("puzzle_io: invalid integer token '" + tok + "'");
        }
    }
    return out;
}

// "Fits" check matching lines.check_line in Python:
//   1-element clue: clue[0] <= size
//   multi-element:  sum(clue) + len(clue) - 1 <= size
//   empty clue: trivially fits (we never enter the multi-element branch
//   negatively: -1 vs unsigned size_t needs care).
bool clue_fits(const std::vector<int>& clue, std::size_t size) {
    if (clue.empty()) {
        return true;
    }
    if (clue.size() == 1) {
        return clue[0] >= 0 && static_cast<std::size_t>(clue[0]) <= size;
    }
    long long total = 0;
    for (int v : clue) total += v;
    total += static_cast<long long>(clue.size()) - 1;
    return total >= 0 && static_cast<unsigned long long>(total) <= size;
}

}  // namespace

PuzzleClues load_clues(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("puzzle_io: cannot open '" + path + "'");
    }

    // Mirror the Python parser exactly: accumulate into a local "cols"
    // vector, swap with "rows" on each '---'. Net result for a file with
    // a single '---': first section ends up in rows, second in cols.
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> cols;

    std::string raw;
    while (std::getline(f, raw)) {
        std::string line = strip(raw);
        if (line.empty()) {
            cols.emplace_back();  // empty clue
        } else if (line[0] == '#') {
            continue;
        } else if (line == "---") {
            std::swap(rows, cols);
        } else {
            cols.push_back(parse_ints(line));
        }
    }

    PuzzleClues clues;
    clues.rows = std::move(rows);
    clues.cols = std::move(cols);
    return clues;
}

bool clues_valid(const PuzzleClues& clues) {
    long long row_sum = 0;
    for (const auto& r : clues.rows) {
        row_sum += std::accumulate(r.begin(), r.end(), 0LL);
    }
    long long col_sum = 0;
    for (const auto& c : clues.cols) {
        col_sum += std::accumulate(c.begin(), c.end(), 0LL);
    }
    if (row_sum != col_sum) {
        return false;
    }
    const std::size_t n_cols = clues.cols.size();
    for (const auto& r : clues.rows) {
        if (!clue_fits(r, n_cols)) return false;
    }
    const std::size_t n_rows = clues.rows.size();
    for (const auto& c : clues.cols) {
        if (!clue_fits(c, n_rows)) return false;
    }
    return true;
}
