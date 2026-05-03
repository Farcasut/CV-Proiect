import argparse
import json
import os
import sys

import cv2
import numpy as np

from Hexboard import Hexboard

DEBUG = False
DEBUG_IMAGES = []
MAPPED_BOARDS = []
view_squares = []

TEST = []
# Empirically selected.
WARP_SIZE = 1000

SIFT_N_FEATURES = 3000
RANSAC_REPROJ_THR = 5.0
# Lowe ratio test
SIFT_RATIO = 0.75
MIN_INLIERS = 12

SCOREBOARD_COLOURS = ['R', 'G', 'O', 'B', 'Y', 'P']

TEMPLATE_GRID_CX = 499.0
TEMPLATE_GRID_CY = 490.0

HEX_COL_STEP = 54.0
HEX_ROW_STEP = 45.0

## How many empty cells are on the original map by row.
ROW_CELL_COUNTS = [4, 7, 8, 9, 10, 9, 10, 9, 8, 7, 4]

# Cell indents
ROW_INDENT_STEPS = [3.0, 1.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 1.5, 3.0]

# The size of a hex
PATCH_RADIUS = 18

SCORE_TABLES = {}


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


def read_scoreboards(directory):
    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            fullname = os.path.join(directory, filename)
            scores.append(fullname)
    scores.sort()
    return scores


def draw_grid_numbers(image, grid, radius=PATCH_RADIUS):
    output = image.copy()
    for number, (x, y) in grid.items():
        cv2.putText(output, str(number), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(output, (int(x), int(y)), radius, (0, 0, 255), 2)

    return output


def display_images(images, width=1920, height=1080):
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


def detect_board_quad(img, scale, k_close=np.ones((12, 12), np.uint8), k_open=np.ones((5, 5), np.uint8)):
    h, w = img.shape[:2]
    small = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    ## We rely on the fact that the game board is on a wooden table and prepare a hsv for the warm brownish color.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    DEBUG_IMAGES.append((hsv, 'small'))
    lower_wood = np.array([4, 10, 50], dtype=np.uint8)
    upper_wood = np.array([35, 255, 255], dtype=np.uint8)

    wood_mask = cv2.inRange(hsv, lower_wood, upper_wood)
    board_mask = cv2.bitwise_not(wood_mask)

    ## Clean up
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
        DEBUG_IMAGES.append((wood_mask, 'wood_mask'))
        DEBUG_IMAGES.append((board_mask, 'board_mask'))
        DEBUG_IMAGES.append((enchanted_image, 'enchanted_image'))
    return corners


def wrap_board(img, corners, width=WARP_SIZE, height=WARP_SIZE):
    if corners is None:
        # This is pure hope. If the corners are missing I'm already fucked.
        corners = detect_board_quad(img, 0.5)
    w = width - 1
    h = height - 1

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, homography, (width, height))

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
    return {cid: (float(p[0][0]), float(p[0][1])) for cid, p in zip(ids, proj)}


def build_cell_map_template():
    leftmost_wide = TEMPLATE_GRID_CX - 4.5 * HEX_COL_STEP
    top_row_y = TEMPLATE_GRID_CY - 5.0 * HEX_ROW_STEP
    cell_map = dict()
    cell_id = 1
    ## ROW_INDENT_STEPS = [3.0, 1.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 1.5, 3.0]
    row_indent = zip(ROW_CELL_COUNTS, ROW_INDENT_STEPS)
    for row_idx, (n, indent) in enumerate(row_indent):
        y = top_row_y + row_idx * HEX_ROW_STEP
        left_x = leftmost_wide + indent * HEX_COL_STEP
        for col in range(n):
            cell_map[cell_id] = (left_x + col * HEX_COL_STEP, y)
            cell_id += 1
    return cell_map


#### Find where tiles are placed.
def score_all_cells(current, previous, cell_grid):
    diff_gray = cv2.cvtColor(cv2.absdiff(current, previous), cv2.COLOR_BGR2GRAY)
    h, w = diff_gray.shape
    scores: dict[int, float] = {}
    for cell_id, (cx, cy) in cell_grid.items():
        cx, cy = int(round(cx)), int(round(cy))
        y0, y1 = max(0, cy - PATCH_RADIUS), min(h, cy + PATCH_RADIUS)
        x0, x1 = max(0, cx - PATCH_RADIUS), min(w, cx + PATCH_RADIUS)
        patch = diff_gray[y0:y1, x0:x1]
        if patch.size > 0:
            scores[cell_id] = float(patch.mean())

    return scores


def detect_color(hsv_img):
    # Define HSV ranges
    color_ranges = {"R": [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))],
                    "B": [((100, 150, 50), (140, 255, 255)),  ## Low saturated images like 2_01
                          ((90, 30, 150), (105, 80, 255))], "G": [((40, 70, 50), (80, 255, 255))],
                    "Y": [((20, 100, 100), (35, 255, 255))], "O": [((10, 100, 100), (20, 255, 255))],
                    "P": [((140, 50, 50), (160, 255, 255))]}

    # white_mask = cv2.inRange(hsv_img, np.array([0, 0, 200]), np.array([180, 50, 255]))
    # valid_mask = cv2.bitwise_not(white_mask)

    color_counts = {}
    for color, ranges in color_ranges.items():
        total = 0
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            # mask = cv2.bitwise_and(mask, valid_mask)
            total += cv2.countNonZero(mask)
        color_counts[color] = total

    # Pick the dominant color
    dominant_color = max(color_counts, key=color_counts.get)
    return dominant_color


def identify_piece_color(pieces, image):
    result = dict()
    size = 14
    for index, (x, y) in pieces.items():
        x = round(x)
        y = round(y)
        x1, y1 = x - size, y - size
        x2, y2 = x + size, y + size

        # Clamp to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        crop = image[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color = detect_color(hsv)
        result[index] = color

    return result


def find_new_cells(warped, cell_grid, previous_wrap):
    previous_wrap = previous_wrap if previous_wrap is not None else cv2.imread("./template/template_board.jpg")
    diff_scores = score_all_cells(warped, previous_wrap, cell_grid)
    ranked = sorted(diff_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [cell_id for cell_id, _ in ranked[:2]]


## scoreboard

def build_scoreboard_template_grid():
    center_x = 53
    center_y = 34
    step_x = 40
    step_y = 33
    template_grid = {}
    idx = 1
    for i in range(19):
        for j in range(6):
            ci_x = center_x + step_x * j
            ci_y = center_y + step_y * i
            template_grid[idx] = (ci_x, ci_y)
            idx = idx + 1
    return template_grid


def project_scoreboard_points(template_grid, sift_homography):
    keys = list(template_grid.keys())
    pts = np.float32([[template_grid[i]] for i in keys])
    proj = cv2.perspectiveTransform(pts, sift_homography)
    return {cid: (float(p[0][0]), float(p[0][1])) for cid, p in zip(keys, proj)}


def colour_signal(patch, colour):
    """
    Colour specific channel ratio for a BGR patch.

    Using channel ratios (e.g. R/(R+G+B)) instead of raw brightness
    makes the signal invariant to overall lighting level, so it works
    regardless of how brightly the card is lit.

      R / O  →  R channel dominates
      G      →  G channel dominates
      B      →  B channel dominates
      Y      →  R + G both high, B low
      P      →  R + B both high, G low
    """

    b = patch[:, :, 0].astype(np.float64)
    g = patch[:, :, 1].astype(np.float64)
    r = patch[:, :, 2].astype(np.float64)
    total = b + g + r + 1e-9

    if colour == 'R': return float((r / total).mean())
    if colour == 'G': return float((g / total).mean())
    if colour == 'O': return float((r / total).mean())
    if colour == 'B': return float((b / total).mean())
    if colour == 'Y': return float(((r + g) / (2 * total)).mean())
    if colour == 'P': return float(((r + b) / (2 * total)).mean())
    return 0.0


def extract_scores(warped_scoreboard: np.ndarray, cell_map: dict, patch_radius: int = 8) -> dict[str, int]:
    h_img, w_img = warped_scoreboard.shape[:2]
    row_indices = np.arange(19, dtype=np.float64)
    scores = {}

    for col_j, colour in enumerate(SCOREBOARD_COLOURS):
        col_signals = []
        for row_i in range(19):
            cell_idx = row_i * 6 + col_j + 1  # matches build_scoreboard_template_grid
            if cell_idx not in cell_map:
                col_signals.append(0.0)
                continue

            cx, cy = cell_map[cell_idx]
            cx, cy = int(round(cx)), int(round(cy))
            y0, y1 = max(0, cy - patch_radius), min(h_img, cy + patch_radius)
            x0, x1 = max(0, cx - patch_radius), min(w_img, cx + patch_radius)
            patch = warped_scoreboard[y0:y1, x0:x1]
            col_signals.append(colour_signal(patch, colour) if patch.size > 0 else 0.0)

        signals = np.array(col_signals)
        # Subtract a linear trend to remove the lighting gradient across the card.
        # Because only 1 of 19 rows is occupied, the fit tracks the empty-cell
        # background and the occupied cell stands out as a positive residual.
        background = np.polyval(np.polyfit(row_indices, signals, 1), row_indices)
        deviation = signals - background

        occupied_row = int(deviation.argmax())
        # Row 0 (top of card) = score 18; row 18 (bottom) = score 0
        scores[colour] = 18 - occupied_row

    return scores


def prepare_scoreboard_tables(score_board_tables_directory, template_path):
    scoreboards = read_scoreboards(score_board_tables_directory)
    template = cv2.imread(template_path)
    template = cv2.resize(template, (300, 700))
    scoreboard_template_grid = build_scoreboard_template_grid()
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
    template_keypoints, template_descriptors = sift.detectAndCompute(template, None)
    for idx_scoreboard, scoreboard in enumerate(scoreboards):
        scoreboard_table = cv2.imread(scoreboard)
        corners = detect_board_quad(scoreboard_table, 0.25)
        warped_image, _ = wrap_board(scoreboard_table, corners, 300, 700)
        sift_homography, _ = _sift(warped_image, sift, template_keypoints, template_descriptors)
        if sift_homography is not None:
            cell_map = project_scoreboard_points(scoreboard_template_grid, sift_homography)
        else:
            cell_map = scoreboard_template_grid

        score = extract_scores(warped_image, cell_map, patch_radius=8)
        print(json.dumps(score, sort_keys=True), os.path.basename(scoreboard))
        SCORE_TABLES[json.dumps(score, sort_keys=True)] = os.path.basename(scoreboard)
        if DEBUG:
            mapped_board = draw_grid_numbers(warped_image, cell_map, 2)
            TEST.extend([(mapped_board, os.path.basename(scoreboard))])
    if DEBUG:
        display_images(TEST, 300, 700)


def play_game(board_directory, output_directory,
              template_path):
    games = group_games(board_directory)
    template = cv2.imread(template_path)
    ## Sift
    sift = cv2.SIFT_create(nfeatures=SIFT_N_FEATURES)
    ## Prepare the template data.
    template_keypoints, template_descriptors = sift.detectAndCompute(template, None)
    template_img_keypoints = cv2.drawKeypoints(template, template_keypoints, None,
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    grid_template = build_cell_map_template()
    for board_game_index, game in enumerate(games):
        previous = template
        hexboard = Hexboard()
        ## Prepare a scoreboard.
        current_score = {key: 0 for key in SCOREBOARD_COLOURS}
        for move_index, img_path in enumerate(game):
            print(board_game_index, move_index)

            img = cv2.imread(img_path)
            if img is None:
                print("Could not read image")
                sys.exit(1)
            if DEBUG:
                template_grid_image = draw_grid_numbers(template, grid_template)
                DEBUG_IMAGES.append((template_img_keypoints, 'template_keypoints'))
                DEBUG_IMAGES.append((template_grid_image, 'template_grid'))

            corners = detect_board_quad(img, scale=0.25)
            warped_image, _ = wrap_board(img, corners, WARP_SIZE, WARP_SIZE)

            sift_homography, _ = _sift(warped_image, sift, template_keypoints, template_descriptors)

            if sift_homography is not None:
                cell_map = project_cell_map(grid_template, sift_homography)
            else:
                cell_map = dict(grid_template)

            mapped_board = draw_grid_numbers(warped_image, cell_map)
            new_piece = find_new_cells(warped_image, cell_map, previous)
            new_piece.sort()
            if DEBUG:
                print(warped_image.shape, previous.shape)
                print(new_piece)
                DEBUG_IMAGES.append((mapped_board, f"Board {board_game_index}, {move_index}"))
                MAPPED_BOARDS.append((mapped_board, f"Board {board_game_index}, {move_index}"))
            new_piece_pos = {new_piece[0]: cell_map.get(new_piece[0]), new_piece[1]: cell_map.get(new_piece[1])}
            pieces_colors = identify_piece_color(new_piece_pos, warped_image)
            added_pieces = draw_grid_numbers(mapped_board, new_piece_pos, 2)
            text_file = []
            text_file.append(f"{new_piece[0]} {pieces_colors[new_piece[0]]}")
            text_file.append(f"{new_piece[1]} {pieces_colors[new_piece[1]]}")
            color_scores = {}
            for index, piece_color in pieces_colors.items():
                q, r = hexboard.pos(index)
                score = hexboard.score_position(q, r, piece_color)
                if score > 0:
                    color_scores[piece_color] = color_scores.get(piece_color, 0) + score
            ## Place the new pieces on the board.
            hexboard.place(pieces_colors)

            for color, total_score in color_scores.items():
                current_score[color] = current_score.get(color, 0) + total_score

            print(color_scores)
            score_key = json.dumps(current_score, sort_keys=True)
            print(score_key)
            print(SCORE_TABLES.get(score_key, ""))

            order = {"R": 0, "G": 1, "O": 2, "B": 3, "Y": 4, "P": 5}
            score_line = [f"{total_score}{color}" for color, total_score in
                          sorted(color_scores.items(), key=lambda x: order.get(x[0], 999))]
            text_file.append(" ".join(score_line))
            predictions_file_name = f"{board_game_index}_{move_index + 1:02}.txt"
            file_path = os.path.join(output_directory, predictions_file_name)
            print(file_path)
            with open(file_path, "w") as predictions_file:
                predictions_file.write("\n".join(text_file))

            # ground_truth_file_name = os.path.join("/home/farcasut/Downloads/boards/", predictions_file_name)
            # with open(ground_truth_file_name, "r") as ground_truth_file:
            #     ground_truth_text = ground_truth_file.read()
            #     isEqual = ground_truth_text == "\n".join(text_file)
            #     if not isEqual:
            #         print("ERROR", predictions_file_name)
            # MAPPED_BOARDS.append((added_pieces, f"Pieces selected {board_game_index}, {move_index}"))
            previous = warped_image
    if DEBUG:
        display_images(DEBUG_IMAGES)
        display_images(view_squares)
        display_images(MAPPED_BOARDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scoreboard preparation and game processing.")

    parser.add_argument("--scoreboard_dir", type=str, required=True, help="Path to scoreboard tables directory")

    parser.add_argument("--scoreboard_template", default="./template/template_scoreboard_2.jpg", type=str,
                        required=False, help="Path to scoreboard template image")

    parser.add_argument("--board_dir", type=str, required=True, help="Path to game boards directory")

    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save predictions")

    parser.add_argument("--board_template", default="./template/template_board.jpg", type=str, required=False, help="Path to board template image")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prepare_scoreboard_tables(score_board_tables_directory=args.scoreboard_dir, template_path=args.scoreboard_template)

    play_game(board_directory=args.board_dir, output_directory=args.output_dir,
              template_path=args.board_template)
