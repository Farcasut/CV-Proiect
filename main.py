import cv2
import numpy as np
import sys
import os

def wrap_board(img, corners, output_size):
    pass

def display_image(img, height=1920, width=1080):
    img_resized = cv2.resize(img, (1920, 1080))
    cv2.imshow("Image", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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

    display_image(img, height=1920, width=1080)