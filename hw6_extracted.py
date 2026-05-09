from typing import List, Optional, Tuple, Set
Pixel = int
EMPTY, FULL, UNKNOWN = 0, 1, 2

Clue = List[int]
Nonogram = List[List[int]]

class Picture:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.pixels = [[UNKNOWN for _ in range(width)] for _ in range(height)]
        self.solved_rows: Set[int] = set()
        self.solved_cols: Set[int] = set()
        self.pixels2: List[List[int]] = list()

    def reverse(self) -> Nonogram:

        self.reversed = [[row[index] for row
                          in self.pixels] for index
                         in range(len(self.pixels[0]))]
        return self.reversed

    def write_column(self, column: int, values: List[int]) -> None:
        for row_index, value in enumerate(values):
            self.pixels[row_index][column] = value

    def get_column(self, column: int) -> List[int]:
        return [self.pixels[index][column]
                for index, _ in enumerate(self.pixels)]

def short_gen_line(clues: List[int]) -> List[int]:
    result = []
    for clue in clues:
        result += [FULL] * clue + [EMPTY]
    return result[:-1]

def load_picture(filename: str) -> Picture:
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    result = list()
    for line in content:
        row = list()
        for character in line:
            if character == '#':
                row.append(FULL)
            elif character == '.':
                row.append(EMPTY)
            else:
                row.append(UNKNOWN)
        result.append(row)
    pic = Picture(len(content), len(content[0]))
    pic.pixels = result
    return pic

def save_picture(pic: Picture, filename: str) -> None:
    with open(filename, mode='w') as f:
        for line in pic.pixels:
            for pixel in line:
                if pixel == EMPTY:
                    f.write('.')
                elif pixel == FULL:
                    f.write('#')
                else:
                    f.write('?')
            f.write('\n')

def load_clues(filename: str) -> Tuple[List[Clue], List[Clue]]:
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    cols: List[Clue] = list()
    for line in content:
        if not line:
            cols.append([])
        elif line[0] == '#':
            pass
        elif line == "---":

            rows = cols
            cols = list()
        else:
            numbers_in_string: List[str] = line.split(" ")
            numbers: List[int] = [int(x) for x in numbers_in_string]
            cols.append(numbers)
    return (rows, cols)

def get_clues(pic: Picture) -> Optional[Tuple[List[Clue], List[Clue]]]:
    rows: List[Clue] = []
    for row in pic.pixels:
        if 2 in row:
            return None
        rows.append(gen_line_clues(row))

    cols: List[Clue] = []
    for col in pic.reverse():
        cols.append(gen_line_clues(col))

    return (rows, cols)

def clues_valid(rows: List[Clue], cols: List[Clue]) -> bool:
    if sum(map(sum, rows)) != sum(map(sum, cols)):
        return False
    for row in rows:
        if not check_line([], row, len(cols), False):
            return False
    for col in cols:
        if not check_line([], col, len(rows), False):
            return False
    return True

def gen_line_clues(line: List[int]) -> List[int]:
    result: List[int] = []
    if not line:
        return result

    line_ = [EMPTY] + line[::-1]
    count = 0


    while line_:
        if line_.pop() == FULL:
            count += 1
        else:
            if count > 0:
                result.append(count)
            count = 0
    return result

def gen_lines(clue: Clue, size: int) -> List[List[Pixel]]:

    return gen_lines_(clue, size, 0, [])

def gen_lines_(clue: Clue,
               size: int,
               depth: int,
               generated: List[List[Pixel]],
               mini: int = 0,
               filter_: Optional[List[Pixel]] = None
               ) -> List[List[Pixel]]:
    if depth == 0:
        if not clue:
            return [[EMPTY] * size]
        current = [FULL] * clue[0]
        generated = []
        for i in range(mini, size - sum(clue) - len(clue) + 2):
            line = [EMPTY] * i
            line.extend(current)
            generated.append(line)
        if filter_:
            generated = filter_gen_lines(filter_, generated)
        if len(clue) > 1:
            return gen_lines_(clue, size, 1, generated, 0, filter_)

    elif depth == len(clue):
        for line in generated:
            line.extend([EMPTY] * (size - len(line)))
        if not filter_:
            return generated
        return filter_gen_lines(filter_, generated)

    else:
        current = clue[depth] * [FULL]
        result = []
        size_needed = size - sum(clue[depth:]) - len(clue[depth:]) + 1
        while generated:
            line = generated.pop()
            for i in range(size_needed - len(line)):
                result.append(line + [EMPTY] * (i + 1) + current)
        if filter_:
            result = filter_gen_lines(filter_, result)
        return gen_lines_(clue, size, depth + 1, result, 0,
                          filter_)
    result_ = []
    while generated:
        line = generated.pop()
        result_.append(line + [EMPTY] * (size - len(line)))
    if not filter_:
        return result_
    return filter_gen_lines(filter_, result_)

def gen_lines_with_prefix(clue: Clue, size: int,
                          prefix: List[Pixel]) -> List[List[Pixel]]:
    return gen_lines_with_prefix_(clue, size, prefix)

def gen_lines_with_prefix_(clue: Clue,
                           size: int,
                           prefix: List[Pixel],
                           filter_: Optional[List[int]] = None
                           ) -> List[List[Pixel]]:
    prefix_2 = prefix[:]
    if not check_line(prefix_2, clue, size, True):
        return []
    current_clue = gen_line_clues(prefix_2)
    if current_clue == clue:
        return [prefix_2 + [EMPTY] * (size - len(prefix_2))]
    return gen_lines_(clue, size, len(current_clue),
                      [prefix_2[:-1]], len(prefix_2), filter_)

def filter_gen_lines(filter_: List[int],
                     lines: List[List[int]]) -> List[List[int]]:
    ones = {i for i, x in enumerate(filter_) if x != EMPTY}
    nulls = {i for i, x in enumerate(filter_) if x != FULL}
    result = []
    while lines:
        line = lines.pop()
        ones_generated = {i for i, x in enumerate(line) if x != EMPTY}
        if ones_generated.issubset(ones):
            nulls_generated = {i for i, x in enumerate(line) if x != FULL}
            if nulls_generated.issubset(nulls):
                result.append(line)
    return result

def solve(rows: List[Clue], cols: List[Clue]) -> Optional[Picture]:
    if not clues_valid(rows, cols):
        return None
    pic = Picture(len(rows), len(cols))
    solve_easiest(rows, cols, pic)
    if pic.width > pic.height:
        return solve_(rows, cols, pic, 0)
    return solve_(rows, cols, pic, 1)

def solve_easiest(rows: List[Clue],
                  cols: List[Clue],
                  pic: Picture
                  ) -> None:
    mapped_cols = list(
        map(lambda x: sum(x) + len(x) - 1 if len(x) > 1 else sum(x),
            cols))
    a = max(mapped_cols)
    highest_col = mapped_cols.index(a)
    solve_one(cols[highest_col], highest_col, True, pic)
    mapped_rows = list(
        map(lambda x: sum(x) + len(x) - 1 if len(x) > 1 else sum(x),
            rows))
    b = max(mapped_rows)
    highest_row = mapped_rows.index(b)
    solve_one(rows[highest_row], highest_row, False, pic)

def solve_one(clue: List[int], index: int,
              is_col: bool, pic: Picture) -> bool:
    line = pic.get_column(index) if is_col else pic.pixels[index]
    size = pic.height if is_col else pic.width
    check_two: List[bool] = list(map(lambda x: UNKNOWN == x, line))
    if all(check_two):
        generated = gen_lines(clue, size)
    elif not any(check_two):
        pic.solved_cols.add(index) if is_col else pic.solved_rows.add(index)
        return True
    else:
        if line[0] == UNKNOWN and line[-1] != UNKNOWN:
            line = line[::-1]
            generated = gen_lines_with_prefix_(
                clue[::-1], size,
                line[:line.index(UNKNOWN)],
                line)
            for x in generated:
                x.reverse()
        else:
            generated = gen_lines_with_prefix_(
                clue, size, line[:line.index(UNKNOWN)], line)

    if not generated:
        return False
    if len(generated) == 1:
        if is_col:
            pic.solved_cols.add(index)
            pic.write_column(index, generated[0])
        else:
            pic.solved_rows.add(index)
            pic.pixels[index] = generated[0]
        return True
    write_intersection(generated, index, is_col, pic)
    return True

def write_intersection(generated: List[List[int]],
                       index: int, is_col: bool,
                       pic: Picture) -> None:
    if is_col:
        for j in range(pic.height):
            mapped_index = list(map(lambda x: x[j], generated))
            if all(mapped_index):
                if generated:
                    pic.pixels[j][index] = FULL
            elif not any(mapped_index):
                pic.pixels[j][index] = EMPTY
        return None

    for j in range(pic.width):
        mapped_index = list(map(lambda x: x[j], generated))
        if all(mapped_index):
            if generated:
                pic.pixels[index][j] = FULL
        elif not any(mapped_index):
            pic.pixels[index][j] = EMPTY

def solve_(rows: List[Clue],
           cols: List[Clue],
           pic: Picture,
           depth: int
           ) -> Optional[Picture]:
    if not solve_check(pic, rows, cols):
        return None

    if all(map(lambda x: 2 not in x, pic.pixels)):
        return pic

    pic.pixels2 = [x[:] for x in pic.pixels]

    for i, row in enumerate(rows):
        if depth == 0:
            break
        if i in pic.solved_rows:
            continue
        if not solve_one(row, i, False, pic):
            return None

    for i, col in enumerate(cols):
        if i in pic.solved_cols:
            continue
        if not solve_one(col, i, True, pic):
            return None

    if depth > 1 and pic.pixels == pic.pixels2:
        for i, line in enumerate(pic.pixels):
            if i not in pic.solved_rows:
                solutions = gen_lines_with_prefix_(rows[i], pic.width,
                                                   line[:line.index(UNKNOWN)],
                                                   line)
                break
        for solution in solutions:
            pic2 = Picture(pic.height, pic.width)
            pic2.solved_rows = set(pic.solved_rows)
            pic2.solved_cols = set(pic.solved_cols)
            pic2.pixels = [x[:] for x in pic.pixels2]
            pic2.pixels[i] = solution
            result = solve_(rows, cols, pic2, depth)
            if result:
                return result
            else:
                del pic2
    return solve_(rows, cols, pic, depth + 1)

def solve_check(pic: Picture,
                row_clues: List[List[int]],
                col_clues: List[List[int]]) -> bool:
    for i, clue in enumerate(row_clues):
        line = pic.pixels[i][:]
        if not check_line(line, clue, len(line)):
            return False

    for i, clue in enumerate(col_clues):
        line = pic.get_column(i)
        if not check_line(line, clue, len(line)):
            return False
    return True

def check_line(line: List[int],
               clues: Clue,
               size: int,
               modify: bool = False) -> bool:
    zero_count = size - sum(clues)
    if line.count(EMPTY) > zero_count:
        return False
    if 2 in line:
        line = line[:line.index(UNKNOWN)]
    generated = gen_line_clues(line)
    if len(generated) == 0:
        size_needed = clues[-1] if len(clues) == 1 else sum(clues) + \
            len(clues) - 1
        if len(line) + size_needed > size:
            return False
        if len(line) + size_needed == size:
            if modify:
                line.extend(short_gen_line(clues))
            return True
        return True
    if len(generated) > len(clues) \
       or (line[-1] == EMPTY and generated != clues[:len(generated)]) \
       or generated[:-1] != clues[:len(generated) - 1] \
       or generated[-1] > clues[len(generated) - 1]:
        return False

    if modify:
        clues_used = len(generated)
        if generated != clues[:clues_used]:
            line.extend([FULL] * (clues[clues_used - 1] - generated[-1]))
            generated = gen_line_clues(line)
        if generated == clues:
            return True
        if line[-1] == FULL:
            line.append(0)
        size_needed = clues[-1] if len(clues[clues_used:]) == 1 else \
            sum(clues[clues_used:]) + len(clues[clues_used:]) - 1
        if clues_used == len(clues) - 1 \
           and size_needed == size - len(line):
            line.extend([FULL] * size_needed)
        generated = gen_line_clues(line)
        clues_used = len(generated)
        size_needed = clues[-1] if len(clues[clues_used:]) == 1 else \
            sum(clues[clues_used:]) + len(clues[clues_used:]) - 1
        if size - len(line) == size_needed:
            line.extend(short_gen_line(clues[clues_used:]))
    return True

TEST_FILENAME = "_test_nonograms_"

def test_1() -> None:
    image1 = ("..#....\n"
              ".#.#..#\n"
              "#######\n"
              "..##..#\n"
              "...###.\n"
              "..#.#..\n")

    with open(TEST_FILENAME, "w") as file:
        file.write(image1)

    pic1 = load_picture(TEST_FILENAME)
    assert pic1.width == 7
    assert pic1.height == 6
    assert pic1.pixels == [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
    ]

    image2 = '.#?\n'

    with open(TEST_FILENAME, "w") as file:
        file.write(image2)

    pic2 = load_picture(TEST_FILENAME)
    assert pic2.width == 3
    assert pic2.height == 1
    assert pic2.pixels == [[0, 1, 2]]

    save_picture(pic1, TEST_FILENAME)
    result = ""
    with open(TEST_FILENAME) as file:
        for line in file:
            result += line.rstrip() + '\n'
        assert result == image1

    save_picture(pic2, TEST_FILENAME)
    with open(TEST_FILENAME) as file:
        lines = file.readlines()
        assert len(lines) == 1
        assert lines[0].rstrip() == '.#?'

    clues = ("# This is a file with clues.\n"
             "# Rows come first:\n"
             "1\n"
             "1 1 1\n"
             "# Comments can be anywhere.\n"
             "7\n"
             "2 1\n"
             "3\n"
             "1 1\n"
             "---\n"
             "1\n"
             "2\n"
             "1 2 1\n"
             "4\n"
             "1 2\n"
             "1 1\n"
             "3\n")

    with open(TEST_FILENAME, "w") as file:
        file.write(clues)

    rows, cols = load_clues(TEST_FILENAME)
    assert rows == [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    assert cols == [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]

    with open(TEST_FILENAME, "w") as file:
        file.write("\n---\n\n")

    rows, cols = load_clues(TEST_FILENAME)
    assert rows == [[]]
    assert cols == [[]]

def test_2() -> None:
    with_unknown = Picture(2, 2)
    with_unknown.pixels[0][0] = FULL
    with_unknown.pixels[0][1] = EMPTY
    with_unknown.pixels[1][0] = FULL

    assert get_clues(with_unknown) is None

    pic = Picture(6, 7)
    pic.pixels = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
    ]

    clues = get_clues(pic)
    assert clues is not None
    rows, cols = clues
    assert rows == [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    assert cols == [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]

def test_3() -> None:
    rows = [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]
    assert clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [7], [1, 1], [1], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1]]

    assert not clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 1, 1, 1], [4], [1, 2], [1, 1], [3]]

    assert not clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [6], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]

    assert not clues_valid(rows, cols)

    assert not clues_valid([[1], [1]], [[1, 1]])

    assert clues_valid([[]], [[]])
    assert clues_valid([[1], []], [[], [1]])

    assert clues_valid([[2], [], [2]], [[1, 1], [2]])

def test_4() -> None:
    assert sorted(gen_lines([1, 3, 2], 9)) == sorted([
        [1, 0, 1, 1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 1, 0, 1, 1]
    ])

    assert sorted(gen_lines([1], 4)) == sorted([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    assert gen_lines([], 10) == [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    assert gen_lines([1, 1, 1, 1], 6) == []

    assert gen_lines([1, 1, 1, 1], 7) == [[1, 0, 1, 0, 1, 0, 1]]

    assert gen_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 64) \
        == [[1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    assert len(gen_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 67)) == 286

def test_5() -> None:
    assert sorted(gen_lines_with_prefix([1, 3, 2], 9, [1, 0, 1])) == sorted([
        [1, 0, 1, 1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 1, 1],
    ])

    assert sorted(gen_lines_with_prefix([1], 4, [0])) == sorted([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    assert gen_lines_with_prefix([], 10, [0, 0, 0]) \
        == [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    assert gen_lines_with_prefix([1, 1, 1, 1], 7, [0]) == []

    assert gen_lines_with_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 1000, [1, 1]) == []

    assert len(gen_lines_with_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     100,
                                     [0 for _ in range(32)])) == 1001

def test_6() -> None:
    pic = solve([[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]],
                [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]])
    assert pic is not None
    assert pic.width == 7
    assert pic.height == 6
    assert pic.pixels == [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
    ]

    assert solve([[2], [], [2]], [[1, 1], [2]]) is None

    pic = solve([[2], [], [2]], [[1, 1], [1, 1]])
    assert pic is not None
    assert pic.width == 2
    assert pic.height == 3
    assert pic.pixels == [[1, 1], [0, 0], [1, 1]]

def test_bonus() -> None:
    pic = solve([[1, 3, 1, 1, 1], [1, 1, 1, 2, 2, 2], [1, 3, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1]],
                [[5], [], [5], [1, 1, 1], [1, 1, 1], [1, 1],
                 [], [1], [5], [], [1], [5], [], [1], [5]])
    assert pic is not None

if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_bonus()

reviewed by: Stanislav Boboň on 2021-01-29 14:05 CET
