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
	$(CXX) $(CXXFLAGS) -fprofile-use=$(PGO_DIR) $(SRCS) -o solver
	rm -f solver-instrumented

pgo-clean:
	rm -rf $(PGO_DIR) solver-instrumented

.PHONY: all clean pgo pgo-clean
