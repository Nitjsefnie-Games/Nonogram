CXX = g++
CXXFLAGS = -O3 -march=native -flto -std=c++17 -Wall -Wextra -pedantic

# Faster allocator for the line cache's per-miss key/result allocations
# (~+3% in both default and --anytime modes; glibc malloc was ~9% of the
# anytime profile). Linked only if present, so the build still works without it.
MALLOC_LIB := $(shell test -e /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 && echo -l:libtcmalloc_minimal.so.4)

SRCS = main.cpp lines.cpp picture.cpp puzzle_io.cpp search.cpp
OBJS = $(SRCS:.cpp=.o)

PGO_DIR = pgo-data

all: solver

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(MALLOC_LIB)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) solver

pgo: pgo-clean
	@echo "=== PGO Stage 1: build instrumented binary ==="
	mkdir -p $(PGO_DIR)
	$(CXX) $(CXXFLAGS) -fprofile-generate=$(PGO_DIR) $(SRCS) -o solver-instrumented
	@echo "=== PGO Stage 2: train on representative puzzles ==="
	./solver-instrumented ../nonograms/medium/108 --max 200000 > /dev/null
	./solver-instrumented ../nonograms/easy_large/4 > /dev/null
	./solver-instrumented ../nonograms/easy_medium/63 > /dev/null
	./solver-instrumented ../nonograms/trivial/1 > /dev/null
	@echo "=== PGO Stage 3: build optimized binary ==="
	$(CXX) $(CXXFLAGS) -fprofile-use=$(PGO_DIR) $(SRCS) -o solver $(MALLOC_LIB)
	rm -f solver-instrumented

pgo-clean:
	rm -rf $(PGO_DIR) solver-instrumented

.PHONY: all clean pgo pgo-clean
