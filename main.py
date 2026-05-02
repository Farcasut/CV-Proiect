import os
import sys

import cv2
import numpy as np

DEBUG = True
DEBUG_IMAGES = []
MAPPED_BOARDS = []

# Empirically selected.
WARP_SIZE = 1000

SIFT_N_FEATURES = 3000
RANSAC_REPROJ_THR = 5.0
# Lowe ratio test
SIFT_RATIO = 0.75
MIN_INLIERS = 12  # minimum RANSAC inliers to trust

TEMPLATE_GRID_CX = 499.0
TEMPLATE_GRID_CY = 490.0

HEX_COL_STEP = 54.0
HEX_ROW_STEP = 45.0

## How many empty cells are on the original map by row.
ROW_CELL_COUNTS = [4, 7, 8, 9, 10, 9, 10, 9, 8, 7, 4]

# Cell indents
ROW_INDENT_STEPS = [3.0, 1.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 1.5, 3.0]


def group_games(directory):
    groups = dict()
    max = -1
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0]

            prefix, value = name.split("_")
            index = int(prefix)
            fullname = f"{directory}{filename}"
            if index not in groups:
                groups[index] = [fullname]
            else:
                groups[index].append(fullname)
            if index > max:
                max = index

    result = [[] for _ in range(max + 1)]
    for index, files in groups.items():
        files.sort()
        result[index] = files

    return result

def draw_grid_numbers(image, grid):
    output = image.copy()

    for number, (x, y) in grid.items():
        cv2.putText(output, str(number), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(output, (int(x), int(y)), 12, (0, 0, 255), 2)

    return output



def display_images(images, height=1080, width=1920):
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
        cv2.putText(display, f"{title} ({idx + 1}/{len(images)})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),
                    3)
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
        cv2.putText(clone, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return clone


def _extract_quad_from_hull(hull: np.ndarray) -> np.ndarray:
    pts = hull.reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([[tl], [tr], [br], [bl]], dtype=np.float32)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
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
    DEBUG_IMAGES.append((hsv, 'small'))
    ## Close small gaps
    wood_maks = cv2.inRange(hsv, np.array([8, 25, 80], dtype=np.uint8), np.array([32, 220, 255], dtype=np.uint8))
    board_mask = cv2.bitwise_not(wood_maks)

    k_close = np.ones((12, 12), np.uint8)
    k_open = np.ones((5, 5), np.uint8)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, k_close)
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, k_open)
    ## Find the biggest contour (hopefully is the board LOL, because in some images the image contains a bit of foreground that is not part of the table)
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        raise RuntimeError(f"Board detection failed: expected 4-vertex polygon, got {len(approx)}.")

    # Scale corners back to original image resolution
    corners = approx.reshape(-1, 2).astype(np.float32) / scale
    corners = order_corners(corners)
    if DEBUG:
        enchanted_image = visualize_corners(img, corners)
        DEBUG_IMAGES.append((wood_maks, 'wood_mask'))
        DEBUG_IMAGES.append((board_mask, 'board_mask'))
        DEBUG_IMAGES.append((enchanted_image, 'enchanted_image'))
    return corners


def wrap_board(img, corners, output_size=WARP_SIZE):
    if corners is None:
        # This is pure hope. If the corners are missing I'm already fucked.
        corners = detect_board_quad(img, 0.5)
    n = output_size - 1
    dst = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, homography, (output_size, output_size))
    if DEBUG:
        DEBUG_IMAGES.append((warped, 'warped'))
    return warped, homography


def _sift(warped, sift, template_keypoints, template_descriptors):
    query_keypoints, query_descriptors = sift.detectAndCompute(warped, None)
    if query_descriptors is None or len(query_keypoints) < 8:
        return None, 0

    matches = cv2.BFMatcher().knnMatch(template_descriptors, query_descriptors, k=2)
    good_matches = [m for m, second_best in matches if m.distance < SIFT_RATIO * second_best.distance]
    if len(good_matches) < MIN_INLIERS:
        return None, len(good_matches)

    template_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    query_pts = np.float32([query_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(template_pts, query_pts, cv2.RANSAC, RANSAC_REPROJ_THR)

    n_inliers = int(inlier_mask.ravel().sum()) if inlier_mask is not None else 0
    return (homography, n_inliers) if (homography is not None and n_inliers >= MIN_INLIERS) else (None, n_inliers)

def project_cell_map(grid_template, sift_homography):
    ids = list(grid_template.keys())
    pts = np.float32([[grid_template[i]] for i in ids])
    proj = cv2.perspectiveTransform(pts, sift_homography)
    return {cid: (float(p[0][0]), float(p[0][1]))
            for cid, p in zip(ids, proj)}


def build_cell_map_template():
    leftmost_wide = TEMPLATE_GRID_CX - 4.5 * HEX_COL_STEP
    top_row_y = TEMPLATE_GRID_CY - 5.0 * HEX_ROW_STEP
    cell_map = dict()
    cell_id = 1
    row_indent = zip(ROW_CELL_COUNTS, ROW_INDENT_STEPS)
    for row_idx, (n, indent) in enumerate(row_indent):
        y = top_row_y + row_idx * HEX_ROW_STEP
        left_x = leftmost_wide + indent * HEX_COL_STEP
        for col in range(n):
            cell_map[cell_id] = (left_x + col * HEX_COL_STEP, y)
            cell_id += 1
    return cell_map


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python <CV_PROJECT> <images_directory>")

    images_directory = sys.argv[1]
    games = group_games("/home/farcasut/Downloads/boards/")
    for index,game in enumerate(games):
        for index_2, img_path in enumerate(game):
            img = cv2.imread(img_path)
            if img is None:
                print("Could not read image")
                sys.exit(1)
            ## Sift.
            sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
            ## Read an prepare the template data.
            template = cv2.imread("./template/template_board.jpg")
            DEBUG_IMAGES.append((template, 'template'))
            template_keypoints, template_descriptors = sift.detectAndCompute(template, None)
            template_img_keypoints = cv2.drawKeypoints(template, template_keypoints, None,
                                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            grid_template = build_cell_map_template()
            template_grid_image = draw_grid_numbers(template, grid_template)
            DEBUG_IMAGES.append((template_img_keypoints, 'template_keypoints'))
            DEBUG_IMAGES.append((template_grid_image, 'template_grid'))

            ## Start the pipeline
            corners = detect_board_quad(img, scale=0.25)
            warped_image, warped_homography = wrap_board(img, corners, WARP_SIZE)

            sift_homography, n_inlines = _sift(warped_image, sift, template_keypoints, template_descriptors)

            if sift_homography is not None:
                cell_map = project_cell_map(grid_template, sift_homography)
            else:
                cell_map = dict(grid_template)

            mapped_board = draw_grid_numbers(warped_image, cell_map)
            DEBUG_IMAGES.append((mapped_board, f"Board {index}, {index_2}"))
            print(f"Board {index}, {index_2}")
            MAPPED_BOARDS.append((mapped_board, f"Board {index}, {index_2}"))
    if DEBUG:
        display_images(MAPPED_BOARDS)
        display_images(DEBUG_IMAGES)
