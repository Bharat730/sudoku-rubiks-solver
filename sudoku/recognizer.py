import cv2
import numpy as np
import pytesseract

# Point to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tesseract config — single character, only digits 1-9
TESS_CONFIG_BASE = '--oem 3 -c tessedit_char_whitelist=123456789'
TESS_CONFIG = f'--psm 10 {TESS_CONFIG_BASE}'
MIN_DIGIT_CONFIDENCE = 10

# ── Cell Processing ───────────────────────────────────────────────────────────

def preprocess_cell(cell_bgr):
    """Clean a single cell for OCR."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # Crop 20% border to remove grid lines
    h, w = gray.shape
    mh, mw = int(h * 0.20), int(w * 0.20)
    gray = gray[mh:h-mh, mw:w-mw]

    # Upscale — tesseract works better on larger images
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # OTSU threshold — clean black digit on white background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Add white border padding (tesseract needs padding)
    thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10,
                                cv2.BORDER_CONSTANT, value=255)
    return thresh

def preprocess_cell_variant(cell_bgr, mode="otsu", crop_ratio=0.18, size=120):
    """Prepare alternate OCR images for thin printed digits."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    mh, mw = int(h * crop_ratio), int(w * crop_ratio)
    gray = gray[mh:h-mh, mw:w-mw]
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if mode == "abs170":
        _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    elif mode == "adapt":
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 7
        )
    elif mode == "morph":
        _, inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.dilate(
            inv,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1
        )
        thresh = 255 - inv
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.copyMakeBorder(
        thresh, 15, 15, 15, 15,
        cv2.BORDER_CONSTANT, value=255
    )

def is_empty_cell(thresh):
    """Return True if a thresholded cell has no digit-sized component."""
    inv = cv2.bitwise_not(thresh)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area >= 120 and h >= 25 and w >= 8:
            return False

    return True

def is_empty_cell_image(cell_bgr):
    """Return True if the raw cell image has no dark digit-like blob."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    mh, mw = int(h * 0.20), int(w * 0.20)
    gray = gray[mh:h-mh, mw:w-mw]

    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use an absolute dark-pixel threshold so pale paper texture/grid shadows
    # do not become foreground the way OTSU sometimes makes them.
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area >= 120 and h >= 25 and w >= 8:
            return False

    return True

def read_digit_with_confidence(thresh, psm=10):
    """Read one digit from a preprocessed cell with Tesseract confidence."""
    data = pytesseract.image_to_data(
        thresh,
        config=f'--psm {psm} {TESS_CONFIG_BASE}',
        output_type=pytesseract.Output.DICT
    )

    best_digit = 0
    best_conf = -1

    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = (text or "").strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1

        if text.isdigit() and len(text) == 1 and 1 <= int(text) <= 9:
            if conf > best_conf:
                best_digit = int(text)
                best_conf = conf

    if best_conf >= MIN_DIGIT_CONFIDENCE:
        return best_digit, best_conf

    return 0, best_conf

def read_digit(thresh, psm=10):
    """Read one digit from a preprocessed cell using Tesseract confidence."""
    digit, _ = read_digit_with_confidence(thresh, psm=psm)
    return digit

def read_digit_multi_pass(cell_bgr):
    """Try several OCR views and return the best confident single digit."""
    base = preprocess_cell(cell_bgr)
    for psm in (10, 6, 8, 13):
        digit, _ = read_digit_with_confidence(base, psm=psm)
        if digit != 0:
            return digit

    attempts = (
        ("otsu", 13),
        ("otsu", 8),
        ("abs170", 10),
        ("abs170", 13),
        ("abs170", 8),
        ("adapt", 10),
        ("adapt", 13),
        ("morph", 10),
    )

    votes = {}
    confidence = {}
    for mode, psm in attempts:
        image = preprocess_cell_variant(cell_bgr, mode)
        digit, conf = read_digit_with_confidence(image, psm=psm)
        if digit != 0:
            votes[digit] = votes.get(digit, 0) + 1
            confidence[digit] = max(confidence.get(digit, -1), conf)

    if not votes:
        return 0

    return max(votes, key=lambda d: (votes[d], confidence[d]))

def predict_cell(cell_bgr):
    """Given a single cell image, return digit (0 = empty)."""
    if is_empty_cell_image(cell_bgr):
        return 0

    if is_empty_cell(preprocess_cell(cell_bgr)):
        return 0

    try:
        return read_digit_multi_pass(cell_bgr)
    except Exception:
        pass

    return 0

def extract_board(cells):
    """Takes list of 81 cell images, returns 9x9 board."""
    board = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = predict_cell(cells[i * 9 + j])
            row.append(digit)
        board.append(row)
    return board

# Kept for compatibility
def get_model():
    pass

def load_model(*args, **kwargs):
    pass

def train_model(*args, **kwargs):
    pass
