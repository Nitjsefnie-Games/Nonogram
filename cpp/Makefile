CXX = g++
# The inline-param bumps let the compiler inline std::vector::push_back's fast
# path (it was outlined as a real call, ~2.5% call overhead on the trail/queue
# pushes in the hot loop); measured ~+4% on the --max 300000 anytime benchmark.
CXXFLAGS = -O3 -march=native -flto -std=c++17 -Wall -Wextra -pedantic \
           --param max-inline-insns-auto=120 --param max-inline-insns-single=600 \
           -funroll-loops

SRCS = main.cpp lines.cpp picture.cpp puzzle_io.cpp search.cpp
OBJS = $(SRCS:.cpp=.o)

PGO_DIR = pgo-data

all: solver

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Every object depends on every header: the headers are few and a stale .o
# built against an old struct layout links (LTO warns about ODR) and misbehaves.
%.o: %.cpp $(wildcard *.hpp)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) solver

# Both stages link to the SAME output name: gcc derives the .gcda file names
# from it (<output>-<source>.gcda), so an instrumented "solver-instrumented"
# produced profiles the "solver" build never found -- and -Wmissing-profile
# is only a warning, so the PGO build silently became a plain build.
# -Werror=missing-profile makes that mismatch fail the build instead.
pgo: pgo-clean
	@echo "=== PGO Stage 1: build instrumented binary ==="
	mkdir -p $(PGO_DIR)
	$(CXX) $(CXXFLAGS) -fprofile-generate=$(PGO_DIR) $(SRCS) -o solver
	@echo "=== PGO Stage 2: train on representative puzzles ==="
	./solver ../nonograms/medium/108 --max 200000 > /dev/null
	./solver ../nonograms/partially_solved/pikachu --anytime --max 30000 > /dev/null
	./solver ../nonograms/easy_large/4 > /dev/null
	./solver ../nonograms/easy_medium/63 > /dev/null
	./solver ../nonograms/trivial/1 > /dev/null
	@echo "=== PGO Stage 3: build optimized binary ==="
	$(CXX) $(CXXFLAGS) -fprofile-use=$(PGO_DIR) -Werror=missing-profile $(SRCS) -o solver

pgo-clean:
	rm -rf $(PGO_DIR) solver-instrumented

.PHONY: all clean pgo pgo-clean
