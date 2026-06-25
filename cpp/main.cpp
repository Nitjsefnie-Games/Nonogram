#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "picture.hpp"
#include "puzzle_io.hpp"
#include "search.hpp"
#include "types.hpp"

namespace {

void print_usage(std::FILE* stream) {
    std::fprintf(stream, "Usage: solver <puzzle_file> [--print] [--max N] [--print-every N]\n");
    std::fprintf(stream, "\n");
    std::fprintf(stream, "Puzzle file format:\n");
    std::fprintf(stream, "  Row clues (one per line, space-separated numbers)\n");
    std::fprintf(stream, "  ---\n");
    std::fprintf(stream, "  Column clues (one per line, space-separated numbers)\n");
    std::fprintf(stream, "\nOptions:\n");
    std::fprintf(stream, "  --print          Print each solution as a grid\n");
    std::fprintf(stream, "  --anytime        Keep lookahead probing on for the whole search\n");
    std::fprintf(stream, "                   (far more solutions/sec on deep enumerations that\n");
    std::fprintf(stream, "                    otherwise stall; slower on easy puzzles)\n");
    std::fprintf(stream, "  --max N          Stop after finding N solutions\n");
    std::fprintf(stream, "  --print-every N  Log count + rate + elapsed every N solutions\n");
    std::fprintf(stream, "                   (default: progressive batches starting at 10, x1.1)\n");
}

// Format an integer with comma thousands separators (e.g. 30000 -> "30,000").
std::string fmt_int_commas(long long n) {
    bool neg = n < 0;
    unsigned long long v = neg ? static_cast<unsigned long long>(-(n + 1)) + 1ULL
                               : static_cast<unsigned long long>(n);
    std::string digits;
    if (v == 0) {
        digits = "0";
    } else {
        while (v > 0) {
            digits.push_back(static_cast<char>('0' + (v % 10)));
            v /= 10;
        }
    }
    std::string out;
    int count = 0;
    for (char c : digits) {
        if (count > 0 && count % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(c);
        ++count;
    }
    std::string result;
    if (neg) result.push_back('-');
    for (auto it = out.rbegin(); it != out.rend(); ++it) {
        result.push_back(*it);
    }
    return result;
}

// Format a double rounded to integer with comma thousands separators
// (mirrors Python's "{:,.0f}" format).
std::string fmt_rate(double r) {
    if (!(r > 0.0)) return "0";
    long long rounded = static_cast<long long>(r + 0.5);
    return fmt_int_commas(rounded);
}

const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::BASIC:     return "basic";
        case Strategy::CONTRA:    return "contra";
        case Strategy::BACKTRACK: return "backtrack";
    }
    return "basic";
}

void print_grid(const Picture& pic) {
    const int H = pic.height();
    const int W = pic.width();
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            std::int8_t v = pic.pixels[static_cast<std::size_t>(r) * W + c];
            char ch;
            if (v == EMPTY) ch = '.';
            else if (v == FULL) ch = '#';
            else ch = '?';
            std::putchar(ch);
        }
        std::putchar('\n');
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(stderr);
        return 1;
    }

    std::string filename;
    bool print_progress = false;
    long long max_solutions = -1;  // -1 = no cap
    bool have_max = false;
    long long print_every = 0;     // 0 = use progressive default
    bool anytime = false;          // keep lookahead probing on for whole search
    long long estimate_dives = 0;  // >0: Knuth-estimate solution count, don't solve

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--print") == 0) {
            print_progress = true;
        } else if (std::strcmp(a, "--anytime") == 0) {
            anytime = true;
        } else if (std::strcmp(a, "--estimate") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--estimate requires a dive count\n");
                return 1;
            }
            try {
                estimate_dives = std::stoll(argv[++i]);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--estimate: invalid integer: %s\n", e.what());
                return 1;
            }
        } else if (std::strcmp(a, "--max") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--max requires a value\n");
                return 1;
            }
            try {
                max_solutions = std::stoll(argv[++i]);
                have_max = true;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--max: invalid integer: %s\n", e.what());
                return 1;
            }
        } else if (std::strcmp(a, "--print-every") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--print-every requires a value\n");
                return 1;
            }
            try {
                print_every = std::stoll(argv[++i]);
                if (print_every <= 0) {
                    std::fprintf(stderr, "--print-every: value must be > 0\n");
                    return 1;
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--print-every: invalid integer: %s\n", e.what());
                return 1;
            }
        } else if (a[0] == '-' && a[1] != '\0') {
            std::fprintf(stderr, "Unknown option: %s\n", a);
            print_usage(stderr);
            return 1;
        } else if (filename.empty()) {
            filename = a;
        } else {
            std::fprintf(stderr, "Unexpected positional arg: %s\n", a);
            return 1;
        }
    }

    if (filename.empty()) {
        print_usage(stderr);
        return 1;
    }

    PuzzleClues clues;
    try {
        clues = load_clues(filename);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    if (!clues_valid(clues)) {
        std::fprintf(stderr, "Invalid clues: %s\n", filename.c_str());
        return 1;
    }

    std::printf("Puzzle size: %zu rows x %zu cols\n", clues.rows.size(), clues.cols.size());
    std::fflush(stdout);

    if (estimate_dives > 0) {
        std::printf("Estimating solution count via %lld Knuth dives...\n", estimate_dives);
        std::fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        const char* se = std::getenv("ESTIMATE_SEED");
        unsigned long seed = se ? std::strtoul(se, nullptr, 10) : 0xC0FFEEUL;
        double est = estimate_solutions(clues.rows, clues.cols, static_cast<long>(estimate_dives), seed);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("Estimated total solutions: %.4e   (%.1fs)\n", est, secs);
        if (est > 0.0) {
            double per_day = 6.4e9;  // ~74k/s observed
            std::printf("At ~74,000 solutions/s that is ~%.2e years to fully enumerate.\n",
                        est / per_day / 365.25);
        }
        return 0;
    }

    std::printf("Solving...\n");
    std::fflush(stdout);

    auto start = std::chrono::steady_clock::now();

    long long solution_count = 0;
    long long print_count_threshold = 10;
    int print_count = 0;

    auto callback = [&](const Picture& pic) -> bool {
        ++solution_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double rate = (elapsed > 0.0) ? (static_cast<double>(solution_count) / elapsed) : 0.0;

        if (print_progress) {
            std::printf("\n=== Solution %s found (%s/s, elapsed %.1fs) ===\n",
                        fmt_int_commas(solution_count).c_str(),
                        fmt_rate(rate).c_str(),
                        elapsed);
            print_grid(pic);
            std::fflush(stdout);
        } else if (print_every > 0) {
            if (solution_count % print_every == 0) {
                std::printf("%s (%s/s, %.1fs) ",
                            fmt_int_commas(solution_count).c_str(),
                            fmt_rate(rate).c_str(),
                            elapsed);
                std::fflush(stdout);
                ++print_count;
                if (print_count == 10) {
                    std::putchar('\n');
                    print_count = 0;
                }
            }
        } else if (solution_count % print_count_threshold == 0) {
            std::printf("%s (%s/s, %.1fs) ",
                        fmt_int_commas(solution_count).c_str(),
                        fmt_rate(rate).c_str(),
                        elapsed);
            std::fflush(stdout);
            print_count_threshold = static_cast<long long>(print_count_threshold * 1.1);
            if (print_count_threshold <= 0) print_count_threshold = 1;
            ++print_count;
            if (print_count == 10) {
                std::putchar('\n');
                print_count = 0;
            }
        }

        if (have_max && solution_count >= max_solutions) {
            return false;  // stop solver
        }
        return true;
    };

    Strategy strategy = Strategy::BASIC;
    solve(clues.rows, clues.cols, callback, &strategy, anytime);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double rate = (elapsed > 0.0) ? (static_cast<double>(solution_count) / elapsed) : 0.0;

    std::putchar('\n');
    std::printf("\nTime: %.4fs\n", elapsed);
    std::printf("Found %s solution(s) (%s/s)\n",
                fmt_int_commas(solution_count).c_str(),
                fmt_rate(rate).c_str());
    std::printf("Strategy: %s\n", strategy_name(strategy));

    return 0;
}
