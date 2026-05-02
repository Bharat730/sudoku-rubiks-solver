import cv2
import numpy as np
import pytesseract

# Point to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Tesseract config — single character, only digits 1-9
TESS_CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'

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

def is_empty_cell(thresh):
    """Return True if cell has no digit using contour analysis."""
    contours, _ = cv2.findContours(
        cv2.bitwise_not(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = thresh.shape
    area = h * w

    for c in contours:
        ca = cv2.contourArea(c)
        if ca > area * 0.02:
            return False
    return True

def predict_cell(cell_bgr):
    """Given a single cell image, return digit (0 = empty)."""
    thresh = preprocess_cell(cell_bgr)

    if is_empty_cell(thresh):
        return 0

    try:
        text = pytesseract.image_to_string(
            thresh, config=TESS_CONFIG
        ).strip()
        if text and text.isdigit() and 1 <= int(text) <= 9:
            return int(text)
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