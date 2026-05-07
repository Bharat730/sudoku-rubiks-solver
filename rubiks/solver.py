import kociemba
from rubiks.detector import build_cube_string

# ── Move descriptions for step-by-step display ────────────────────────────────

MOVE_DESC = {
    'U':  "Top face clockwise",
    "U'": "Top face counter-clockwise",
    'U2': "Top face 180°",
    'D':  "Bottom face clockwise",
    "D'": "Bottom face counter-clockwise",
    'D2': "Bottom face 180°",
    'F':  "Front face clockwise",
    "F'": "Front face counter-clockwise",
    'F2': "Front face 180°",
    'B':  "Back face clockwise",
    "B'": "Back face counter-clockwise",
    'B2': "Back face 180°",
    'L':  "Left face clockwise",
    "L'": "Left face counter-clockwise",
    'L2': "Left face 180°",
    'R':  "Right face clockwise",
    "R'": "Right face counter-clockwise",
    'R2': "Right face 180°",
}

def solve_cube(faces):
    """
    faces: dict {U, R, F, D, L, B} -> 9-char color string each
    Returns: list of move strings e.g. ['U', "R'", 'F2', ...]
    or raises ValueError with a message if cube state is invalid.
    """
    try:
        cube_str = build_cube_string(faces)
        print(f"[Rubiks] Cube string: {cube_str}")

        # Validate length
        if len(cube_str) != 54:
            raise ValueError(f"Invalid cube string length: {len(cube_str)}")

        # Check each face letter appears exactly 9 times
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            count = cube_str.count(face)
            if count != 9:
                raise ValueError(
                    f"Face {face} appears {count} times (expected 9). "
                    f"Color detection error."
                )

        solution = kociemba.solve(cube_str)
        moves    = solution.strip().split()
        print(f"[Rubiks] Solution: {solution}")
        return moves

    except ValueError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Solver error: {str(e)}")

def get_move_details(moves):
    """
    Returns list of dicts with move + description for frontend display.
    """
    return [
        {
            'move': m,
            'desc': MOVE_DESC.get(m, m)
        }
        for m in moves
    ]