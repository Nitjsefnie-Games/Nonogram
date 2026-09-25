CXX = g++
# The inline-param bumps let the compiler inline std::vector::push_back's fast
# path (it was outlined as a real call, ~2.5% call overhead on the trail/queue
# pushes in the hot loop); measured ~+4% on the --max 300000 anytime benchmark.
CXXFLAGS = -O3 -march=native -flto -std=c++17 -Wall -Wextra -pedantic \
           --param max-inline-insns-auto=120 --param max-inline-insns-single=600 \
           -funroll-loops -DNDEBUG
# Static link: solve time is unchanged, but the corpus pipeline runs one
# process per puzzle and most puzzles solve in well under a millisecond, so
# dynamic loading dominated their wall time (easy_large/7368: 2.9ms -> 1.7ms
# per process, page faults 234 -> 149, measured over 200 runs).
LDFLAGS = -static

SRCS = main.cpp lines.cpp picture.cpp puzzle_io.cpp search.cpp
OBJS = $(SRCS:.cpp=.o)

PGO_DIR = pgo-data

all: solver

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Every object depends on every header: the headers are few and a stale .o
# built against an old struct layout links (LTO warns about ODR) and misbehaves.
%.o: %.cpp $(wildcard *.hpp)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) solver solver-stats

# Diagnostic build: DEBUG_CACHE_STATS=1 prints line-cache and probe
# statistics to stderr. Separate binary because the counters cost ~3% of
# instructions on the hot path when compiled in.
stats: solver-stats
solver-stats: $(SRCS) $(wildcard *.hpp) Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -DNONOGRAM_STATS $(SRCS) -o $@

# Flag variants of the PGO build measured 2026-09-25 (81ed2f17e, instructions
# on the four benchmark puzzles, then cycles on the shielded core):
# -fno-unroll-loops +1.2-1.8%; -fno-tracer mixed within 0.5%; the inlining
# budget raised to 200/1000 -0.1 to -0.3% instructions on all four but
# cycles mixed (hard/3867 +3.2% median, hard/30532 -6.7%, 12130 +4.9%,
# five rounds under load), and 300/1500 compiles to the same code.
# Both stages link to the SAME output name: gcc derives the .gcda file names
# from it (<output>-<source>.gcda), so an instrumented "solver-instrumented"
# produced profiles the "solver" build never found -- and -Wmissing-profile
# is only a warning, so the PGO build silently became a plain build.
# -Werror=missing-profile makes that mismatch fail the build instead.
pgo: pgo-clean
	@echo "=== PGO Stage 1: build instrumented binary ==="
	mkdir -p $(PGO_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -fprofile-generate=$(PGO_DIR) $(SRCS) -o solver
	@echo "=== PGO Stage 2: train on the puzzles that carry the corpus wall ==="
	# hard/3867 and hard/30532 are 80% of the corpus's summed solve time,
	# easy_large/7382 and easy_medium/12130 the largest of the rest; the
	# node caps keep the training under 15 s. The state cache's yield gate
	# is held inert so the training tree, and with it the profile, is the
	# same on every build.
	STATE_CACHE_YIELD=0 MAX_NODES=150000 ./solver ../nonograms/hard/3867 > /dev/null
	STATE_CACHE_YIELD=0 MAX_NODES=300000 ./solver ../nonograms/hard/30532 > /dev/null
	STATE_CACHE_YIELD=0 ./solver ../nonograms/easy_large/7382 > /dev/null
	STATE_CACHE_YIELD=0 ./solver ../nonograms/easy_medium/12130 > /dev/null
	STATE_CACHE_YIELD=0 ./solver ../nonograms/easy_large/4 > /dev/null
	STATE_CACHE_YIELD=0 ./solver ../nonograms/easy_medium/63 > /dev/null
	STATE_CACHE_YIELD=0 ./solver ../nonograms/trivial/1 > /dev/null
	@echo "=== PGO Stage 3: build optimized binary ==="
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -fprofile-use=$(PGO_DIR) -Werror=missing-profile $(SRCS) -o solver

pgo-clean:
	rm -rf $(PGO_DIR) solver-instrumented

.PHONY: all clean pgo pgo-clean stats
