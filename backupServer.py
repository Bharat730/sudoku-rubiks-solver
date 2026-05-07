import cv2
import numpy as np
import base64
import socket
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from sudoku.detector import detect_grid, extract_cells
from sudoku.recognizer import extract_board, get_model
from sudoku.solver import get_solution, print_board
from itertools import combinations
from rubiks.detector import extract_face_colors, face_to_string, validate_face
from rubiks.solver import solve_cube, get_move_details

app = Flask(__name__)
HISTORY_FILE = Path("solve_history.jsonl")
MAX_HISTORY_ITEMS = 20
CAPTURE_DIR = Path("captures")
SUDOKU_WARP_SIZE = 900
MIN_GIVENS_TO_SOLVE = 16
SOLVER_TIME_LIMIT_SECONDS = 2.5

# ── Preload model on startup ──────────────────────────────────────────────────
print("Loading digit recognition model...")
get_model()
print("Model ready.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(data_url):
    """Convert base64 image from browser to OpenCV BGR frame."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def encode_image(frame):
    """Convert OpenCV frame to base64 string for sending to browser."""
    _, buffer = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def timestamp_id():
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def save_image(path, frame):
    """Save an OpenCV image, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)

def save_capture_set(puzzle_type, frame, warped=None, prefix=None):
    """Save raw and optionally warped images from one capture."""
    prefix = prefix or timestamp_id()
    base = CAPTURE_DIR / puzzle_type

    raw_path = base / "raw" / f"{prefix}.jpg"
    save_image(raw_path, frame)

    paths = {"raw_image": str(raw_path)}
    if warped is not None:
        warped_path = base / "warped" / f"{prefix}.jpg"
        save_image(warped_path, warped)
        paths["warped_image"] = str(warped_path)

    return paths

def save_history(entry):
    """Append one solved puzzle entry to local JSONL history."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **entry
    }
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def load_history(limit=MAX_HISTORY_ITEMS, puzzle_type=None):
    """Load most recent solved puzzle entries."""
    if not HISTORY_FILE.exists():
        return []

    entries = []
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if puzzle_type is None or entry.get("type") == puzzle_type:
                entries.append(entry)

    return entries[-limit:][::-1]

def print_solved_grid(original_board, solved_board):
    """Print solved Sudoku grid with givens and filled values."""
    print("[Sudoku] Solved grid:")
    for r in range(9):
        row = []
        for c in range(9):
            value = solved_board[r][c]
            row.append(str(value) if original_board[r][c] != 0 else f"({value})")
        print(" ".join(row))

def is_complete_valid_board(board):
    """Return True when a solved Sudoku board satisfies all constraints."""
    target = set(range(1, 10))
    if any(set(row) != target for row in board):
        return False
    if any(set(board[r][c] for r in range(9)) != target for c in range(9)):
        return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            values = {
                board[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
            }
            if values != target:
                return False
    return True

def solve_with_ocr_repair(board, max_removed=1):
    """Solve a board, optionally dropping a few OCR givens if needed."""
    givens = [
        (r, c)
        for r in range(9)
        for c in range(9)
        if board[r][c] != 0
    ]
    if len(givens) < MIN_GIVENS_TO_SOLVE:
        return board, None, []

    solved = get_solution(board, max_seconds=SOLVER_TIME_LIMIT_SECONDS)
    if solved is not None and is_complete_valid_board(solved):
        return board, solved, []

    for remove_count in range(1, max_removed + 1):
        if len(givens) - remove_count < MIN_GIVENS_TO_SOLVE:
            break

        for cells in combinations(givens, remove_count):
            candidate = [row[:] for row in board]
            for r, c in cells:
                candidate[r][c] = 0

            solved = get_solution(candidate, max_seconds=SOLVER_TIME_LIMIT_SECONDS)
            if solved is not None and is_complete_valid_board(solved):
                return candidate, solved, [
                    {"row": r, "col": c, "value": board[r][c]}
                    for r, c in cells
                ]

    return board, None, []

def overlay_solution(frame, original_board, solved_board, contour, img_size=450):
    """Draw solved digits onto frame over empty cells."""
    from sudoku.detector import order_points
    overlay   = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cell_size = img_size // 9

    for r in range(9):
        for c in range(9):
            if original_board[r][c] == 0 and solved_board[r][c] != 0:
                x = c * cell_size + cell_size // 4
                y = r * cell_size + int(cell_size * 0.75)
                cv2.putText(
                    overlay, str(solved_board[r][c]),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    img_size / 450, (0, 255, 0),
                    max(2, img_size // 225), cv2.LINE_AA
                )

    pts = order_points(contour)
    dst = np.array([
        [0, 0], [img_size-1, 0],
        [img_size-1, img_size-1], [0, img_size-1]
    ], dtype="float32")
    M_inv       = cv2.getPerspectiveTransform(dst, pts)
    h, w        = frame.shape[:2]
    warped_back = cv2.warpPerspective(overlay, M_inv, (w, h))

    mask   = warped_back.astype(bool)
    result = frame.copy()
    result[mask] = warped_back[mask]
    return result

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    puzzle_type = request.args.get("type")
    return jsonify({
        "success": True,
        "history": load_history(puzzle_type=puzzle_type)
    })

@app.route("/solve/sudoku", methods=["POST"])
def solve_sudoku():
    try:
        data  = request.get_json()
        frame = decode_image(data["image"])
        capture_id = timestamp_id()
        capture_paths = save_capture_set("sudoku", frame, prefix=capture_id)

        print(f"[Sudoku] Frame size: {frame.shape}")
        print(f"[Sudoku] Saved raw capture: {capture_paths['raw_image']}")

        # Detect grid
        warped, contour, M = detect_grid(frame, size=SUDOKU_WARP_SIZE)
        if warped is None:
            return jsonify({
                "success": False,
                "error": "No Sudoku grid detected. Make sure the full grid is visible and well lit.",
                "capture": capture_paths
            })

        capture_paths.update(save_capture_set(
            "sudoku", frame, warped=warped, prefix=capture_id
        ))

        print(f"[Sudoku] Grid detected, warped size: {warped.shape}")
        print(f"[Sudoku] Saved warped grid: {capture_paths['warped_image']}")

        # Extract and recognize digits
        cells = extract_cells(warped)
        board = extract_board(cells)

        print(f"[Sudoku] Detected board:")
        for row in board:
            print(row)

        # Check if board is all zeros (recognition failed)
        total_given = sum(1 for r in board for c in r if c != 0)
        print(f"[Sudoku] Total given digits detected: {total_given}")

        if total_given < MIN_GIVENS_TO_SOLVE:
            return jsonify({
                "success": False,
                "error": f"Only {total_given} digits detected. Need at least {MIN_GIVENS_TO_SOLVE} clear digits before solving. Try filling the frame with the grid and holding steady.",
                "capture": capture_paths
            })

        # Solve
        repaired_board, solved, dropped_digits = solve_with_ocr_repair(board)
        if solved is None:
            return jsonify({
                "success": False,
                "error": f"Could not solve puzzle. {total_given} digits detected — some may be wrong. Try again.",
                "original_board": board,
                "capture": capture_paths
            })

        if dropped_digits:
            print(f"[Sudoku] Dropped likely OCR mistakes: {dropped_digits}")
            board = repaired_board
            total_given = sum(1 for r in board for c in r if c != 0)

        print("[Sudoku] Solved successfully!")
        print_solved_grid(board, solved)

        history_entry = save_history({
            "type": "sudoku",
            "original_board": board,
            "solved_board": solved,
            "digits_found": total_given
        })

        # Overlay solution on original frame
        result_frame = overlay_solution(frame, board, solved, contour, img_size=warped.shape[0])
        result_b64   = encode_image(result_frame)
        warped_b64   = encode_image(warped)

        return jsonify({
            "success":        True,
            "result_image":   result_b64,
            "warped_image":   warped_b64,
            "original_board": board,
            "solved_board":   solved,
            "digits_found":   total_given,
            "capture":        capture_paths,
            "history_entry":  history_entry,
            "history":        load_history(puzzle_type="sudoku")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Server error: {str(e)}"})

@app.route("/debug/save", methods=["POST"])
def debug_save():
    """Save warped grid and all 81 cells to disk for inspection."""
    try:
        data  = request.get_json()
        frame = decode_image(data["image"])
        capture_id = timestamp_id()
        capture_paths = save_capture_set("sudoku_debug", frame, prefix=capture_id)
        warped, contour, M = detect_grid(frame, size=SUDOKU_WARP_SIZE)
        if warped is None:
            return jsonify({"success": False, "error": "No grid detected", "capture": capture_paths})
        capture_paths.update(save_capture_set(
            "sudoku_debug", frame, warped=warped, prefix=capture_id
        ))
        cv2.imwrite("debug_warped.jpg", warped)
        cells = extract_cells(warped)
        import os
        os.makedirs("debug_cells", exist_ok=True)
        for i, cell in enumerate(cells):
            cv2.imwrite(f"debug_cells/cell_{i:02d}.jpg", cell)
        return jsonify({
            "success": True,
            "message": "Saved debug_warped.jpg, debug_cells/, and capture images",
            "capture": capture_paths
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/scan/rubiks", methods=["POST"])
def scan_rubiks():
    """Scan one face of the Rubik's cube."""
    try:
        data   = request.get_json()
        frame  = decode_image(data["image"])
        face   = data.get("face", "U")

        colors, annotated = extract_face_colors(frame)
        print(f"[Rubiks] Face {face}: {colors}")

        if not validate_face(colors):
            return jsonify({
                "success": False,
                "error": "Some colors not detected. Hold cube steady with good lighting."
            })

        face_str      = face_to_string(colors)
        annotated_b64 = encode_image(annotated)

        return jsonify({
            "success":   True,
            "colors":    face_str,
            "grid":      colors,
            "annotated": annotated_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/solve/rubiks", methods=["POST"])
def solve_rubiks():
    """Solve the cube given all 6 scanned faces."""
    try:
        data  = request.get_json()
        faces = data.get("faces", {})

        moves   = solve_cube(faces)
        details = get_move_details(moves)

        # Save to history
        save_history({
            "type":  "rubiks",
            "moves": moves,
            "count": len(moves)
        })

        return jsonify({
            "success": True,
            "moves":   [m["move"] for m in details],
            "details": details,
            "count":   len(moves)
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Get local IP so user knows what to open on phone
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  Server running!")
    print(f"  Open on your phone: http://{local_ip}:5000")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=("cert.pem", "key.pem"))
