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

def solve(board):
    """
    Solve sudoku using backtracking.
    board: 9x9 list of ints, 0 means empty.
    Returns True if solved, False if unsolvable.
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def get_solution(board):
    """
    Takes a 9x9 board (0 for empty), returns solved board or None.
    Does not modify the original board.
    """
    import copy
    b = copy.deepcopy(board)
    if solve(b):
        return b
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