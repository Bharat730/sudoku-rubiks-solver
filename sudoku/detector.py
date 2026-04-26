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
    """Find the largest square contour which is the sudoku grid."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],    # top-left
        pts[np.argmin(diff)], # top-right
        pts[np.argmax(s)],    # bottom-right
        pts[np.argmax(diff)]  # bottom-left
    ], dtype="float32")

def warp_perspective(img, contour, size=450):
    """Apply perspective transform to get a top-down view of the grid."""
    pts = order_points(contour)
    dst = np.array([
        [0, 0], [size-1, 0],
        [size-1, size-1], [0, size-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (size, size))
    return warped, M

def extract_cells(warped):
    """Split the warped grid into 81 individual cells."""
    cells = []
    h, w = warped.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = warped[y1:y2, x1:x2]
            cells.append(cell)
    return cells

def detect_grid(frame):
    """
    Main function: takes a frame, returns:
    - warped: top-down view of sudoku grid (or None)
    - contour: the detected contour (or None)
    - M: perspective transform matrix (or None)
    """
    thresh = preprocess(frame)
    contour = find_sudoku_contour(thresh)
    if contour is None:
        return None, None, None
    warped, M = warp_perspective(frame, contour)
    return warped, contour, M