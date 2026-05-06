import cv2
import numpy as np

def preprocess(img):
    """Convert to grayscale, blur and threshold the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def find_sudoku_contour(thresh):
    """
    Find the sudoku grid contour using multiple strategies.
    Returns the best 4-point contour or None.
    """
    h, w = thresh.shape
    min_area = (h * w) * 0.05  # grid must be at least 5% of frame

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break  # remaining contours are too small

        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            # Check it's roughly square (aspect ratio between 0.7 and 1.3)
            x, y, cw, ch = cv2.boundingRect(approx)
            ratio = cw / ch if ch != 0 else 0
            if 0.7 < ratio < 1.3:
                return approx

    return None

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],     # top-left
        pts[np.argmin(diff)],  # top-right
        pts[np.argmax(s)],     # bottom-right
        pts[np.argmax(diff)]   # bottom-left
    ], dtype="float32")

def warp_perspective(img, contour, size=450):
    """Apply perspective transform to get a top-down view of the grid."""
    pts = order_points(contour)
    dst = np.array([
        [0, 0],
        [size-1, 0],
        [size-1, size-1],
        [0, size-1]
    ], dtype="float32")
    M      = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

def crop_inner_grid(warped, size=450):
    """Crop the actual dark Sudoku grid from a warped paper/screen image."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    candidates = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < (h * w) * 0.25:
            continue
        if x <= 2 and y <= 2 and x + cw >= w - 2 and y + ch >= h - 2:
            continue
        ratio = cw / ch if ch else 0
        if 0.75 <= ratio <= 1.35:
            candidates.append((area, x, y, cw, ch))

    if not candidates:
        return warped

    _, x, y, cw, ch = max(candidates, key=lambda item: item[0])
    inset = max(4, int(min(cw, ch) * 0.01))
    x1 = min(w - 1, x + inset)
    y1 = min(h - 1, y + inset)
    x2 = max(x1 + 1, x + cw - inset)
    y2 = max(y1 + 1, y + ch - inset)
    cropped = warped[y1:y2, x1:x2]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

def extract_cells(warped):
    """Split warped grid into 81 individual cell images."""
    cells   = []
    h, w    = warped.shape[:2]
    cell_h  = h // 9
    cell_w  = w // 9
    for row in range(9):
        for col in range(9):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            cells.append(warped[y1:y2, x1:x2])
    return cells

def draw_contour(frame, contour, color=(0, 255, 0), thickness=3):
    """Draw detected contour on frame for debugging."""
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, color, thickness)
    return frame

def detect_grid(frame, size=450):
    """
    Main detection function.
    Returns warped grid image, contour, and transform matrix.
    All None if no grid found.
    """
    thresh          = preprocess(frame)
    contour         = find_sudoku_contour(thresh)
    if contour is None:
        return None, None, None
    warped, M = warp_perspective(frame, contour, size=size)
    warped = crop_inner_grid(warped, size=size)
    return warped, contour, M
