#pragma once
#include <string>
#include <vector>

struct PuzzleClues {
    std::vector<std::vector<int>> rows;
    std::vector<std::vector<int>> cols;
};

PuzzleClues load_clues(const std::string& path);
bool clues_valid(const PuzzleClues& clues);
