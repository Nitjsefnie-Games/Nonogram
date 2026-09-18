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

# Both stages link to the SAME output name: gcc derives the .gcda file names
# from it (<output>-<source>.gcda), so an instrumented "solver-instrumented"
# produced profiles the "solver" build never found -- and -Wmissing-profile
# is only a warning, so the PGO build silently became a plain build.
# -Werror=missing-profile makes that mismatch fail the build instead.
pgo: pgo-clean
	@echo "=== PGO Stage 1: build instrumented binary ==="
	mkdir -p $(PGO_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -fprofile-generate=$(PGO_DIR) $(SRCS) -o solver
	@echo "=== PGO Stage 2: train on representative puzzles ==="
	./solver ../nonograms/easy_medium/108 --max 200000 > /dev/null
	./solver ../nonograms/easy_medium/108 > /dev/null
	MAX_NODES=300000 ./solver ../nonograms/partially_solved/3867 > /dev/null
	./solver ../nonograms/partially_solved/pikachu --anytime --max 30000 > /dev/null
	./solver ../nonograms/easy_large/4 > /dev/null
	./solver ../nonograms/easy_medium/63 > /dev/null
	./solver ../nonograms/trivial/1 > /dev/null
	@echo "=== PGO Stage 3: build optimized binary ==="
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -fprofile-use=$(PGO_DIR) -Werror=missing-profile $(SRCS) -o solver

pgo-clean:
	rm -rf $(PGO_DIR) solver-instrumented

.PHONY: all clean pgo pgo-clean stats
