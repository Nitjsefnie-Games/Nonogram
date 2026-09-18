#!/usr/bin/env python3
"""Generate the n-Dom puzzle (webpbn 8098 is 9-Dom) in this repo's format.

The series from the webpbn solver survey (https://webpbn.com/survey/dom.html):
a (2n+1) x (2n+1) grid where line solving does nothing and the search tree is
everything, scalable to any size. Usage: bench/ndom.py N > puzzle
"""
import sys

def ndom(n):
    rows = ["3"] + ["1", "3 1"] * (n - 1) + ["1", "1"]
    cols = ["1", "1"] + ["1 3", "1"] * (n - 1) + ["3"]
    return "\n".join(rows) + "\n---\n" + "\n".join(cols) + "\n"

if __name__ == "__main__":
    sys.stdout.write(ndom(int(sys.argv[1])))
