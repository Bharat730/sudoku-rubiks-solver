import cv2
import numpy as np
import base64
import socket
from flask import Flask, render_template, request, jsonify
from sudoku.detector import detect_grid, extract_cells
from sudoku.recognizer import extract_board, get_model
from sudoku.solver import get_solution, print_board

app = Flask(__name__)

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
                    1.0, (0, 255, 0), 2, cv2.LINE_AA
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

@app.route("/solve/sudoku", methods=["POST"])
def solve_sudoku():
    try:
        data  = request.get_json()
        frame = decode_image(data["image"])

        # Detect grid
        warped, contour, M = detect_grid(frame)
        if warped is None:
            return jsonify({"success": False, "error": "No Sudoku grid detected. Make sure the full grid is visible."})

        # Extract and recognize digits
        cells = extract_cells(warped)
        board = extract_board(cells)

        # Solve
        solved = get_solution(board)
        if solved is None:
            return jsonify({"success": False, "error": "Could not solve puzzle. Try better lighting or clearer digits."})

        # Overlay solution on original frame
        result_frame = overlay_solution(frame, board, solved, contour)
        result_b64   = encode_image(result_frame)
        warped_b64   = encode_image(warped)

        # Format board for display
        board_display  = board
        solved_display = solved

        return jsonify({
            "success":      True,
            "result_image": result_b64,
            "warped_image": warped_b64,
            "original_board": board_display,
            "solved_board":   solved_display
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/solve/rubiks", methods=["POST"])
def solve_rubiks():
    # Placeholder for Rubik's cube solver
    return jsonify({"success": False, "error": "Rubik's Cube solver coming soon!"})

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Get local IP so user knows what to open on phone
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"  Server running!")
    print(f"  Open on your phone: http://{local_ip}:5000")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)