import cv2
import numpy as np

# ── Color definitions calibrated from your cube ───────────────────────────────
# Format: list of (lower_hsv, upper_hsv)
# Red wraps around H=0/180 so it needs two ranges

COLOR_RANGES = {
    'U': [  # White — low saturation (your white S=9-25, V=37-182)
        (np.array([0,   0,   35]),  np.array([180, 35,  255]))
    ],
    'D': [  # Yellow — H=20-27, high S
        (np.array([18,  80,  25]),  np.array([28,  255, 255]))
    ],
    'F': [  # Green — H=33-48, high S (clear gap above yellow)
        (np.array([33,  160, 25]),  np.array([50,  255, 255]))
    ],
    'B': [  # Blue — H=104-110, S=87-190, V=58-206 (your data)
        (np.array([103, 80,  50]),  np.array([116, 255, 255]))
    ],
    'L': [  # Orange — H=9-13, S=185-254, V=117-199 (your data)
        (np.array([7,   170, 100]), np.array([16,  255, 255]))
    ],
    'R': [  # Red — H=0-4, S=180-244, V=139-233 (your data)
        (np.array([0,   160, 100]), np.array([5,   255, 255])),
        (np.array([168, 160, 100]), np.array([180, 255, 255]))
    ],
}

COLOR_DISPLAY = {
    'U': ('White',  (255, 255, 255)),
    'D': ('Yellow', (0,   220, 220)),
    'F': ('Green',  (0,   180, 0  )),
    'B': ('Blue',   (200, 100, 0  )),
    'L': ('Orange', (0,   140, 255)),
    'R': ('Red',    (0,   0,   220)),
    '?': ('Unknown',(128, 128, 128)),
}

# ── Color detection ───────────────────────────────────────────────────────────

def classify_color(bgr_region):
    """
    Given a BGR region (numpy array), classify its color.
    Uses the median pixel to avoid outlier influence.
    Returns face letter or '?'.
    """
    # Use median for robustness
    median_bgr = np.median(bgr_region.reshape(-1, 3), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Too dark to classify
    if v < 25:
        return '?'

    # White: very low saturation (S < 50 from your calibration data)
    if s < 50:
        return 'U'

    # Check each color range
    for face, ranges in COLOR_RANGES.items():
        if face == 'U':
            continue
        for (lo, hi) in ranges:
            if (lo[0] <= h <= hi[0] and
                lo[1] <= s <= hi[1] and
                lo[2] <= v <= hi[2]):
                return face

    return '?'

def extract_face_colors(frame):
    """
    Given a frame with a Rubik's face centered in view,
    detect the 3x3 grid of sticker colors.

    Returns:
      colors: 3x3 list of face letters
      annotated: frame with color boxes drawn
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Centered square region — 60% of shorter dimension
    size    = int(min(h, w) * 0.6)
    x_start = (w - size) // 2
    y_start = (h - size) // 2
    cell    = size // 3

    colors = []
    for row in range(3):
        row_colors = []
        for col in range(3):
            # Cell boundaries
            x1c = x_start + col * cell
            y1c = y_start + row * cell
            x2c = x1c + cell
            y2c = y1c + cell

            # Sample inner 50% of cell to avoid grid lines/edges
            pad_x = cell // 4
            pad_y = cell // 4
            sx1 = max(0, x1c + pad_x)
            sy1 = max(0, y1c + pad_y)
            sx2 = min(w,  x2c - pad_x)
            sy2 = min(h,  y2c - pad_y)

            region = frame[sy1:sy2, sx1:sx2]
            if region.size == 0:
                row_colors.append('?')
                continue

            face = classify_color(region)
            row_colors.append(face)

            # Draw filled box on annotated frame
            color_bgr = COLOR_DISPLAY.get(face, COLOR_DISPLAY['?'])[1]
            cv2.rectangle(annotated, (x1c+3, y1c+3), (x2c-3, y2c-3), color_bgr, -1)
            cv2.rectangle(annotated, (x1c+3, y1c+3), (x2c-3, y2c-3), (0,0,0), 2)
            cv2.putText(annotated, face,
                        (x1c + cell//4, y2c - cell//4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0) if face == 'U' else (255, 255, 255),
                        2)

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
    faces: dict with keys U,R,F,D,L,B each being a 9-char string.
    kociemba expects: U R F D L B order.
    """
    order = ['U', 'R', 'F', 'D', 'L', 'B']
    return ''.join(faces[f] for f in order)

def validate_face(colors_3x3):
    """Check if all 9 stickers were detected (no '?')."""
    return all(c != '?' for row in colors_3x3 for c in row)