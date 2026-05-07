import cv2
import numpy as np

# ── Color definitions in HSV ──────────────────────────────────────────────────
# Each color has a list of (lower, upper) HSV ranges
# Some colors like red wrap around 180 so need two ranges

COLOR_RANGES = {
    'U': [  # White
        (np.array([0,   0,   180]), np.array([180, 40,  255]))
    ],
    'D': [  # Yellow
        (np.array([20,  100, 100]), np.array([35,  255, 255]))
    ],
    'F': [  # Green
        (np.array([45,  80,  80]),  np.array([85,  255, 255]))
    ],
    'B': [  # Blue
        (np.array([95,  80,  80]),  np.array([130, 255, 255]))
    ],
    'L': [  # Orange
        (np.array([8,   150, 150]), np.array([20,  255, 255]))
    ],
    'R': [  # Red — wraps around hue=0
        (np.array([0,   150, 150]), np.array([8,   255, 255])),
        (np.array([170, 150, 150]), np.array([180, 255, 255]))
    ],
}

COLOR_DISPLAY = {
    'U': ('White',  (255, 255, 255)),
    'D': ('Yellow', (0,   220, 220)),
    'F': ('Green',  (0,   180, 0  )),
    'B': ('Blue',   (200, 100, 0  )),
    'L': ('Orange', (0,   140, 255)),
    'R': ('Red',    (0,   0,   220)),
}

# ── Color detection ───────────────────────────────────────────────────────────

def classify_color(bgr_pixel):
    """
    Given a BGR pixel value, return the face letter (U/D/F/B/L/R).
    Returns '?' if no color matched.
    """
    hsv = cv2.cvtColor(
        np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV
    )[0][0]

    best_face  = '?'
    best_score = 0

    for face, ranges in COLOR_RANGES.items():
        for (lo, hi) in ranges:
            # Check if hsv is within range
            in_range = all(lo[i] <= hsv[i] <= hi[i] for i in range(3))
            if in_range:
                # Score by saturation+value for confidence
                score = int(hsv[1]) + int(hsv[2])
                if score > best_score:
                    best_score = score
                    best_face  = face

    return best_face

def extract_face_colors(frame):
    """
    Given a frame with a Rubik's face centered in view,
    detect the 3x3 grid of sticker colors.

    Returns:
      colors: 3x3 list of face letters e.g. [['U','R','F'], ...]
      annotated: frame with color boxes drawn
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Define a centered square region (60% of smaller dimension)
    size   = int(min(h, w) * 0.6)
    x_start = (w - size) // 2
    y_start = (h - size) // 2
    cell   = size // 3

    colors = []
    for row in range(3):
        row_colors = []
        for col in range(3):
            # Center of each sticker
            cx = x_start + col * cell + cell // 2
            cy = y_start + row * cell + cell // 2

            # Sample a small region around center (avoid edges)
            sample_size = cell // 4
            region = frame[
                max(0, cy - sample_size): cy + sample_size,
                max(0, cx - sample_size): cx + sample_size
            ]

            if region.size == 0:
                row_colors.append('?')
                continue

            # Average color of region
            avg_bgr = region.mean(axis=(0, 1)).astype(np.uint8)
            face    = classify_color(avg_bgr)
            row_colors.append(face)

            # Draw box on annotated frame
            x1 = x_start + col * cell + 4
            y1 = y_start + row * cell + 4
            x2 = x_start + (col + 1) * cell - 4
            y2 = y_start + (row + 1) * cell - 4

            color_bgr = COLOR_DISPLAY.get(face, ('?', (128,128,128)))[1]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(annotated, face, (x1+8, y2-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        colors.append(row_colors)

    # Draw outer guide box
    cv2.rectangle(annotated,
                  (x_start, y_start),
                  (x_start + size, y_start + size),
                  (100, 255, 100), 3)

    return colors, annotated

def face_to_string(colors_3x3):
    """Convert 3x3 color list to 9-char string for kociemba."""
    return ''.join(c for row in colors_3x3 for c in row)

def build_cube_string(faces):
    """
    faces: dict with keys U,R,F,D,L,B each being a 9-char string
    kociemba expects: U R F D L B order
    """
    order = ['U', 'R', 'F', 'D', 'L', 'B']
    return ''.join(faces[f] for f in order)

def validate_face(colors_3x3):
    """Check if all 9 colors were detected (no '?')."""
    for row in colors_3x3:
        for c in row:
            if c == '?':
                return False
    return True