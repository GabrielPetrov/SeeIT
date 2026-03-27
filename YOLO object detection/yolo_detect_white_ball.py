# ball_with_stick_single.py
#
# Usage:
#   python ball_with_stick_single.py --input real_test_photo.jpg --output result.jpg
#
# Install:
#   pip install ultralytics opencv-python numpy

import argparse
import math
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


def point_to_box_distance(px, py, x1, y1, x2, y2):
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


def ball_to_box_distance(ball_center, ball_radius, box):
    px, py = ball_center
    center_dist = point_to_box_distance(px, py, *box)
    return max(0.0, center_dist - ball_radius)


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def detect_stick_line(image):
    """
    Detect the long dark stick in the lower half of the image.
    Returns:
        ((x1, y1), (x2, y2)) or None
    """
    h, w = image.shape[:2]
    y_start = int(h * 0.45)
    roi = image[y_start:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dark object mask
    _, mask = cv2.threshold(blur, 85, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(h * 0.18),
        maxLineGap=25,
    )

    if lines is None:
        return None

    best_line = None
    best_score = -1

    for line in lines:
        x1, y1, x2, y2 = line[0]
        gx1, gy1 = x1, y1 + y_start
        gx2, gy2 = x2, y2 + y_start

        length = dist((gx1, gy1), (gx2, gy2))
        if length < h * 0.18:
            continue

        # Prefer diagonally upward lines in lower image
        dy = abs(gy2 - gy1)
        dx = abs(gx2 - gx1)
        if dx < 10 and dy < 10:
            continue

        lower_bonus = max(gy1, gy2) / h
        score = length * (1 + lower_bonus)

        if score > best_score:
            best_score = score
            best_line = ((gx1, gy1), (gx2, gy2))

    return best_line


def detect_ball_candidates(image):
    """
    Detect white/light ball candidates in lower half.
    Returns list of dicts.
    """
    h, w = image.shape[:2]
    y_start = int(h * 0.45)
    roi = image[y_start:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=35,
        param1=100,
        param2=18,
        minRadius=14,
        maxRadius=42,
    )

    candidates = []

    if circles is None:
        return candidates

    circles = np.round(circles[0]).astype(int)

    for x, y, r in circles:
        gx, gy = int(x), int(y + y_start)

        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(roi.shape[1], x + r)
        y2 = min(roi.shape[0], y + r)

        patch = roi[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_v = float(np.mean(hsv[:, :, 2]))

        # White / off-white preference
        if mean_s > 95:
            continue
        if mean_v < 110:
            continue

        candidates.append({
            "center": (gx, gy),
            "radius": int(r),
            "bbox": (
                max(0, gx - r),
                max(0, gy - r),
                min(w - 1, gx + r),
                min(h - 1, gy + r),
            ),
            "mean_s": mean_s,
            "mean_v": mean_v,
        })

    return candidates


def choose_ball_using_stick(image):
    """
    Pick the circle candidate that is closest to one endpoint of the stick.
    """
    stick = detect_stick_line(image)
    candidates = detect_ball_candidates(image)

    if not candidates:
        return None, stick

    if stick is None:
        # fallback: choose lowest-brightest reasonable candidate
        best = None
        best_score = -1
        h = image.shape[0]

        for c in candidates:
            score = c["radius"] * 2 + c["mean_v"] + (c["center"][1] / h) * 50 - c["mean_s"]
            if score > best_score:
                best_score = score
                best = c
        return best, stick

    p1, p2 = stick

    # real ball should be near one endpoint of the stick
    best = None
    best_score = float("inf")

    for c in candidates:
        center = c["center"]
        d1 = dist(center, p1)
        d2 = dist(center, p2)
        endpoint_dist = min(d1, d2)

        # prefer candidates near an endpoint, penalize weak white-ness
        score = endpoint_dist + c["mean_s"] * 0.5 - c["radius"] * 0.3

        if score < best_score:
            best_score = score
            best = c

    # Optional sanity threshold
    if best is not None and best_score > 140:
        return None, stick

    return best, stick


def choose_object(ball, detections):
    candidates = []
    for det in detections:

        d = ball_to_box_distance(ball["center"], ball["radius"], det["box"])
        candidates.append({**det, "ball_dist": d})

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["ball_dist"], -x["conf"]))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output_detected.jpg")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.12)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: failed to load image: {args.input}")
        sys.exit(1)

    ball, stick = choose_ball_using_stick(image)
    if ball is None:
        print("Error: ball not detected.")
        sys.exit(1)

    model = YOLO(args.model)
    results = model.predict(source=image, conf=args.conf, save=False, verbose=False)
    if not results:
        print("Error: no YOLO results.")
        sys.exit(1)

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "name": name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

    chosen = choose_object(ball, detections)

    display = image.copy()

    # Draw all detections
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["name"]
        conf = det["conf"]

        color = (130, 130, 255)
        
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display,
            f"{name} {conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Draw stick
    if stick is not None:
        (sx1, sy1), (sx2, sy2) = stick
        cv2.line(display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)

    # Draw ball
    bx, by = ball["center"]
    br = ball["radius"]
    cv2.circle(display, (bx, by), br, (0, 255, 0), 2)
    cv2.circle(display, (bx, by), 4, (255, 0, 255), -1)
    cv2.putText(
        display,
        "Detected ball",
        (bx - 30, by - br - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Draw chosen object
    if chosen is not None:
        x1, y1, x2, y2 = chosen["box"]
        name = chosen["name"]

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            display,
            f"Closest object: {name}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.line(display, (bx, by), (cx, cy), (255, 0, 0), 2)

        print(f"Ball center: ({bx}, {by}), radius: {br}")
        print(f"Chosen object: {name}")
        print(f"Distance ball->object: {chosen['ball_dist']:.2f}px")
    else:
        print(f"Ball center: ({bx}, {by}), radius: {br}")
        print("No valid object chosen.")

    ok = cv2.imwrite(args.output, display)
    if not ok:
        print(f"Error: failed to save output: {args.output}")
        sys.exit(1)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()