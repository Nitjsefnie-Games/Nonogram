CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra -pedantic

SRCS = main.cpp lines.cpp picture.cpp puzzle_io.cpp search.cpp
OBJS = $(SRCS:.cpp=.o)

all: solver

solver: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) solver

.PHONY: all clean
