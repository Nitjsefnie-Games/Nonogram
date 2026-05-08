from lines import check_line


def load_clues(filename):
    with open(filename) as f:
        content = [x.strip() for x in f]
    cols = []
    rows = []
    for line in content:
        if not line:
            cols.append([])
        elif line[0] == '#':
            continue
        elif line == '---':
            rows, cols = cols, rows
        else:
            cols.append([int(x) for x in line.split()])
    return rows, cols


def clues_valid(rows, cols):
    return (sum(map(sum, rows)) == sum(map(sum, cols))
            and all(check_line(r, len(cols)) for r in rows)
            and all(check_line(c, len(rows)) for c in cols))
