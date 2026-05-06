import copy
import time


def is_valid(board, row, col, num):
    """Check if placing num at board[row][col] is valid."""
    # Check row
    if num in board[row]:
        return False
    # Check column
    if num in [board[r][col] for r in range(9)]:
        return False
    # Check 3x3 box
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if board[r][c] == num:
                return False
    return True

def has_valid_givens(board):
    """Return False if the starting board already contradicts itself."""
    for row in range(9):
        for col in range(9):
            num = board[row][col]
            if num == 0:
                continue
            board[row][col] = 0
            ok = is_valid(board, row, col, num)
            board[row][col] = num
            if not ok:
                return False
    return True

def get_candidates(board, row, col):
    """Return all valid numbers for one empty cell."""
    return [num for num in range(1, 10) if is_valid(board, row, col, num)]

def find_best_empty(board):
    """Find the empty cell with the fewest valid candidates."""
    best_pos = None
    best_candidates = None

    for row in range(9):
        for col in range(9):
            if board[row][col] != 0:
                continue

            candidates = get_candidates(board, row, col)
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_pos = (row, col)
                best_candidates = candidates
                if len(best_candidates) == 0:
                    return best_pos, best_candidates

    return best_pos, best_candidates

def solve(board, deadline=None):
    """
    Solve sudoku using backtracking.
    board: 9x9 list of ints, 0 means empty.
    Returns True if solved, False if unsolvable.
    """
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("Sudoku solve timed out")

    pos, candidates = find_best_empty(board)
    if pos is None:
        return True
    if not candidates:
        return False

    row, col = pos
    for num in candidates:
        board[row][col] = num
        if solve(board, deadline=deadline):
            return True
        board[row][col] = 0

    return False

def get_solution(board, max_seconds=None):
    """
    Takes a 9x9 board (0 for empty), returns solved board or None.
    Does not modify the original board.
    """
    b = copy.deepcopy(board)
    if not has_valid_givens(b):
        return None

    deadline = None
    if max_seconds is not None:
        deadline = time.perf_counter() + max_seconds

    try:
        if solve(b, deadline=deadline):
            return b
    except TimeoutError:
        return None

    return None

def print_board(board):
    """Utility to print board nicely in terminal."""
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            row_str += (str(val) if val != 0 else ".") + " "
        print(row_str)
