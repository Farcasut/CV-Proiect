from xmlrpc.client import MAXINT, MININT

import cv2
import numpy as np
import sys
import os

DEBUG = True
DEBUG_IMAGES = []

# Empirically selected.
WARP_SIZE = 1000


def display_images(images, height=1080, width=1920):
    # Safety: ensure correct format
    if len(images) == 0:
        return
    idx = 0
    while True:
        img, title = images[idx]
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        img_resized = cv2.resize(img, (width, height))
        display = img_resized.copy()
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            display,
            f"{title} ({idx+1}/{len(images)})",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )
        cv2.imshow("Image Viewer", display)
        key = cv2.waitKey(0) & 0xFF
        if key in [83, ord('d'), ord('l')]:
            idx = (idx + 1) % len(images)
        elif key in [81, ord('a'), ord('h')]:
            idx = (idx - 1) % len(images)
        elif key in [27, ord('q')]:
            break
    cv2.destroyAllWindows()


def visualize_corners(img, corners):
    clone = img.copy()
    corners_int = corners.astype(int)
    cv2.polylines(clone, [corners_int], isClosed=True, color=(0, 255, 0), thickness=3)
    for i, (x, y) in enumerate(corners_int):
        cv2.circle(clone, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(clone, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return clone

def _extract_quad_from_hull(hull: np.ndarray) -> np.ndarray:
    pts = hull.reshape(-1, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([[tl], [tr], [br], [bl]], dtype=np.float32)

def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    # top left smallest x+y
    rect[0] = pts[np.argmin(s)]
    # top right  smallest x-y
    rect[1] = pts[np.argmin(diff)]
    # bottom-right largest x+y
    rect[2] = pts[np.argmax(s)]
    # bottom-left largest x-y
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_board_quad(img, scale):
    h, w = img.shape[:2]
    small = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    ## We rely on the fact that the game board is on a wooden table and prepare a hsv for the warm brownish color.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ## Close small gaps
    wood_maks = cv2.inRange(hsv,
                            np.array([8,  25, 80],  dtype=np.uint8),
                            np.array([32, 220, 255], dtype=np.uint8))
    board_mask = cv2.bitwise_not(wood_maks)

    k_close = np.ones((20, 20), np.uint8)
    k_open = np.ones((10, 10), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, k_close)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN,  k_open)
    ## Find the biggest contour (hopefully is the board LOL, because in some images the image contains a bit of foreground that is not part of the table)
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = contours[0]
    peri = cv2.arcLength(board_contour, True)
    approx = cv2.approxPolyDP(board_contour, 0.02 * peri, True)

    if len(approx) != 4:
        # Fallback: take the convex hull and keep only the 4 extreme points
        hull = cv2.convexHull(board_contour)
        ## We try to force the convex hull to have only 4 points.
        approx = _extract_quad_from_hull(hull)

    if approx is None or len(approx) != 4:
        raise RuntimeError(
            f"Board detection failed: expected 4-vertex polygon, got {len(approx)}.")

    # Scale corners back to original image resolution
    corners = approx.reshape(-1, 2).astype(np.float32) / scale
    corners = order_corners(corners)
    if DEBUG:
        enchanted_image = visualize_corners(img, corners)
        DEBUG_IMAGES.append((wood_maks, 'wood_mask'))
        DEBUG_IMAGES.append((board_mask, 'board_mask'))
        DEBUG_IMAGES.append((enchanted_image, 'enchanted_image'))
    return corners




def wrap_board(img, corners, output_size = WARP_SIZE):
    if corners is None:
        # This is pure hope. If the corners are missing I'm already fucked.
        corners = detect_board_quad(img, 0.5)
    n = output_size-1
    dst = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, homography, (output_size, output_size))
    if DEBUG:
        DEBUG_IMAGES.append((warped, 'warped'))
    return warped

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python board_detection.py <image.jpg> [output_dir]")
        sys.exit(1)
    img_path =  sys.argv[1]
    output_dir = sys.argv[2]

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image")
        sys.exit(1)

    corners = detect_board_quad(img, scale=0.25)
    warped = wrap_board(img, corners)
    display_images(DEBUG_IMAGES)