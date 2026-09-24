#include <chrono>
#include <csignal>
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
    std::fprintf(stream, "  --balance K      Score branch cells by K*min - max of the two probe\n");
    std::fprintf(stream, "                   fills (K=6 suggested): much smaller trees on hard\n");
    std::fprintf(stream, "                   unique puzzles; --max N runs on many-solution\n");
    std::fprintf(stream, "                   puzzles stop at a different point. Ignored by --anytime\n");
    std::fprintf(stream, "  --print-every N  Log count + rate + elapsed every N solutions\n");
    std::fprintf(stream, "                   (default: progressive batches starting at 10, x1.1)\n");
    std::fprintf(stream, "  --learn          Learn a clause from every contradiction (count mode)\n");
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

// SIGTERM / SIGINT stop the search at its next branch node instead of
// killing the process, so a run cut off by a timeout still reports how far
// it got (explored fraction, and in enumerating mode the solutions so far).
extern "C" void on_stop_signal(int) { request_stop(); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(stderr);
        return 1;
    }
    std::signal(SIGTERM, on_stop_signal);
    std::signal(SIGINT, on_stop_signal);

    std::string filename;
    bool print_progress = false;
    long long max_solutions = -1;  // -1 = no cap
    bool have_max = false;
    long long print_every = 0;     // 0 = use progressive default
    bool anytime = false;          // keep lookahead probing on for whole search
    double balance_k = 0.0;        // --balance K: K*min - max branch score
    long long estimate_dives = 0;  // >0: Knuth-estimate solution count, don't solve
    bool learn = false;            // --learn: clause-learning mode

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--print") == 0) {
            print_progress = true;
        } else if (std::strcmp(a, "--anytime") == 0) {
            anytime = true;
        } else if (std::strcmp(a, "--learn") == 0) {
            learn = true;
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
        } else if (std::strcmp(a, "--balance") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--balance requires a value\n");
                return 1;
            }
            try {
                balance_k = std::stod(argv[++i]);
                if (!(balance_k > 0.0)) {
                    std::fprintf(stderr, "--balance: value must be > 0\n");
                    return 1;
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "--balance: invalid number: %s\n", e.what());
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

    // The clock is read only when a progress line is printed: reading it on
    // every solution was 2-4% of a solution-dense enumeration (hard/6689,
    // 2.2M solutions in 0.9s).
    double p_elapsed = 0.0, p_rate = 0.0;
    auto stamp = [&]() {
        auto now = std::chrono::steady_clock::now();
        p_elapsed = std::chrono::duration<double>(now - start).count();
        p_rate = (p_elapsed > 0.0) ? (static_cast<double>(solution_count) / p_elapsed) : 0.0;
    };

    // Running estimate of the total: solutions found / fraction of the
    // search space completed (see explored_fraction()).
    auto fmt_duration = [](double s) -> std::string {
        char buf[32];
        if (s < 120.0) std::snprintf(buf, sizeof buf, "%.0fs", s);
        else if (s < 7200.0) std::snprintf(buf, sizeof buf, "%.1fmin", s / 60.0);
        else if (s < 2.0 * 86400.0) std::snprintf(buf, sizeof buf, "%.1fh", s / 3600.0);
        else if (s < 2.0 * 365.25 * 86400.0) std::snprintf(buf, sizeof buf, "%.1fd", s / 86400.0);
        else std::snprintf(buf, sizeof buf, "%.3gy", s / (365.25 * 86400.0));
        return std::string(buf);
    };
    // Running estimate of the total and of the time to finish: solutions
    // found / fraction of the search space completed, and elapsed scaled
    // the same way (see explored_fraction()).
    // Alongside the whole-run average, an exponential moving average of the
    // recent progress (mass covered per second, solutions per unit of mass)
    // between prints: it tracks the region the search is in now, so when
    // the fraction stops moving its ETA grows instead of staying put.
    double last_frac = 0.0, last_elapsed = 0.0;
    long long last_solutions = 0;
    double ema_mass_rate = 0.0, ema_density = 0.0;
    bool ema_ready = false;
    auto est_suffix = [&]() -> std::string {
        const double frac = explored_fraction();
        if (!(frac > 0.0)) return std::string();
        char buf[192];
        int n = std::snprintf(buf, sizeof buf, " est~%.2e %.3g%% eta~%s", static_cast<double>(solution_count) / frac,
                              frac * 100.0, fmt_duration(p_elapsed / frac - p_elapsed).c_str());
        const double dt = p_elapsed - last_elapsed, dm = frac - last_frac;
        if (dt > 0.0) {
            const double mass_rate = dm / dt;
            const double density = dm > 0.0 ? static_cast<double>(solution_count - last_solutions) / dm : ema_density;
            const double a = 0.3;
            ema_mass_rate = ema_ready ? (1.0 - a) * ema_mass_rate + a * mass_rate : mass_rate;
            ema_density = ema_ready ? (1.0 - a) * ema_density + a * density : density;
            ema_ready = true;
            const double remaining = 1.0 - frac;
            const double ema_total = static_cast<double>(solution_count) + remaining * ema_density;
            if (ema_mass_rate > 0.0)
                std::snprintf(buf + n, sizeof buf - n, " ema~%.2e eta~%s", ema_total, fmt_duration(remaining / ema_mass_rate).c_str());
            else
                std::snprintf(buf + n, sizeof buf - n, " ema~%.2e eta~stalled", ema_total);
        }
        last_frac = frac; last_elapsed = p_elapsed; last_solutions = solution_count;
        return std::string(buf);
    };

    auto callback = [&](const Picture& pic) -> bool {
        ++solution_count;

        if (print_progress) {
            stamp();
            std::printf("\n=== Solution %s found (%s/s, elapsed %.1fs) ===\n",
                        fmt_int_commas(solution_count).c_str(),
                        fmt_rate(p_rate).c_str(),
                        p_elapsed);
            print_grid(pic);
            std::fflush(stdout);
        } else if (print_every > 0) {
            if (solution_count % print_every == 0) {
                stamp();
                std::printf("%s (%s/s, %.1fs%s) ",
                            fmt_int_commas(solution_count).c_str(),
                            fmt_rate(p_rate).c_str(),
                            p_elapsed, est_suffix().c_str());
                std::fflush(stdout);
                ++print_count;
                if (print_count == 10) {
                    std::putchar('\n');
                    print_count = 0;
                }
            }
        } else if (solution_count == 1 || solution_count % print_count_threshold == 0) {
            // The first solution is always announced, so a run cut off by a
            // timeout still shows whether any solution was found.
            stamp();
            std::printf("%s (%s/s, %.1fs%s) ",
                        fmt_int_commas(solution_count).c_str(),
                        fmt_rate(p_rate).c_str(),
                        p_elapsed, est_suffix().c_str());
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
    // An exhaustive run counts: independent regions of the grid are
    // counted separately and multiplied, so no solution is visited one by
    // one. Runs that need the solutions themselves (--print, --max N,
    // --anytime) enumerate.
    const bool count_mode = !print_progress && !have_max && !anytime;
    std::string count_str;
    if (count_mode) set_progress_interval(60.0);
    // --learn is a count-mode flag: the enumerating runs ignore it.
    solve(clues.rows, clues.cols, callback, &strategy, anytime, balance_k, count_mode, &count_str, learn && count_mode);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double rate = (elapsed > 0.0) ? (static_cast<double>(solution_count) / elapsed) : 0.0;

    std::putchar('\n');
    std::printf("\nTime: %.4fs\n", elapsed);
    if (stop_requested()) {
        // No `Found` line: the count is not known. The tooling reads the
        // absence as a timeout.
        std::printf("Stopped by signal after %.1fs; explored %.8f%% of the search space\n",
                    elapsed, explored_fraction() * 100.0);
        if (count_mode) print_count_progress(elapsed);
        std::printf("Position: %s\n", stop_position().c_str());
        if (!count_mode)
            std::printf("Solutions so far: %s (%s/s)\n", fmt_int_commas(solution_count).c_str(), fmt_rate(rate).c_str());
        std::printf("Strategy: %s\n", strategy_name(strategy));
        return 0;
    }
    if (count_mode) {
        std::string with_commas;
        int n = 0;
        for (auto it = count_str.rbegin(); it != count_str.rend(); ++it) {
            if (n > 0 && n % 3 == 0) with_commas.insert(with_commas.begin(), ',');
            with_commas.insert(with_commas.begin(), *it);
            ++n;
        }
        std::printf("Found %s solution(s) (counted)\n", with_commas.c_str());
    } else {
        std::printf("Found %s solution(s) (%s/s)\n",
                    fmt_int_commas(solution_count).c_str(),
                    fmt_rate(rate).c_str());
    }
    if (!count_mode && explored_fraction() < 1.0 - 1e-12 && solution_count > 0)
        std::printf("Explored %.3g%% of the search space; estimated total ~%.3e solutions, ~%s to finish\n",
                    explored_fraction() * 100.0, static_cast<double>(solution_count) / explored_fraction(),
                    fmt_duration(elapsed / explored_fraction() - elapsed).c_str());
    std::printf("Strategy: %s\n", strategy_name(strategy));

    return 0;
}
