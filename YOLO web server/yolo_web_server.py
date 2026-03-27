import math
import os
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import uvicorn

from pathlib import Path
from datetime import datetime
# ----------------------------
# Configuration
# ----------------------------

MODEL_PATH = os.environ.get("BALL_DETECT_MODEL", "yolov8s.pt")
YOLO_CONF = float(os.environ.get("BALL_DETECT_CONF", "0.12"))

app = FastAPI(title="Ball Closest Object Detector")

model = YOLO(MODEL_PATH)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

IGNORE_CLASSES = {"sports ball", "baseball bat"}

# ----------------------------
# Core geometry helpers
# ----------------------------

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


# ----------------------------
# Stick / ball detection
# ----------------------------

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
    best = None
    best_score = float("inf")

    for c in candidates:
        center = c["center"]
        d1 = dist(center, p1)
        d2 = dist(center, p2)
        endpoint_dist = min(d1, d2)

        score = endpoint_dist + c["mean_s"] * 0.5 - c["radius"] * 0.3

        if score < best_score:
            best_score = score
            best = c

    if best is not None and best_score > 140:
        return None, stick

    return best, stick


# ----------------------------
# Object selection
# ----------------------------

def choose_object(ball, detections):
    candidates = []
    for det in detections:
        d = ball_to_box_distance(ball["center"], ball["radius"], det["box"])
        candidates.append({**det, "ball_dist": d})

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["ball_dist"], -x["conf"]))
    return candidates[0]

def choose_ball_from_yolo(result, model):
    """
    Find the best YOLO 'sports ball' detection and convert it
    into the same ball structure used by the rest of the code.
    """
    balls = []

    if result.boxes is None:
        return None

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = model.names[cls_id]

        if name != "sports ball":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = int(min(x2 - x1, y2 - y1) / 2)

        balls.append({
            "center": (cx, cy),
            "radius": radius,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
        })

    if not balls:
        return None

    # choose the most confident sports ball
    balls.sort(key=lambda b: b["conf"], reverse=True)
    return balls[0]

# def run_detection(image: np.ndarray) -> Dict[str, Any]:
#     if image is None:
#         raise ValueError("Invalid image")

#     ball, stick = choose_ball_using_stick(image)
#     if ball is None:
#         return {
#             "success": False,
#             "error": "ball_not_detected"
#         }

#     results = model.predict(source=image, conf=YOLO_CONF, save=False, verbose=False)
#     if not results:
#         return {
#             "success": False,
#             "error": "no_yolo_results"
#         }

#     result = results[0]
#     detections = []

#     if result.boxes is not None:
#         for box in result.boxes:
#             cls_id = int(box.cls[0].item())
#             conf = float(box.conf[0].item())
#             name = model.names[cls_id]

#             if name in IGNORE_CLASSES:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             detections.append({
#                 "name": name,
#                 "conf": conf,
#                 "box": (x1, y1, x2, y2),
#             })

#     chosen = choose_object(ball, detections)

#     response = {
#         "success": True,
#         "ball_center": {"x": int(ball["center"][0]), "y": int(ball["center"][1])},
#         "ball_radius": int(ball["radius"]),
#         "closest_object": chosen["name"] if chosen else None,
#         "distance_px": float(chosen["ball_dist"]) if chosen else None,
#     }

#     return response

def run_detection(image: np.ndarray) -> Dict[str, Any]:
    if image is None:
        raise ValueError("Invalid image")

    results = model.predict(source=image, conf=YOLO_CONF, save=False, verbose=False)
    if not results:
        return {
            "success": False,
            "error": "no_yolo_results"
        }

    result = results[0]

    ball = choose_ball_from_yolo(result, model)
    if ball is None:
        return {
            "success": False,
            "error": "sports_ball_not_detected"
        }

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names[cls_id]

            if name in IGNORE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "name": name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

    chosen = choose_object(ball, detections)

    response = {
        "success": True,
        "ball_center": {"x": int(ball["center"][0]), "y": int(ball["center"][1])},
        "ball_radius": int(ball["radius"]),
        "closest_object": chosen["name"] if chosen else None,
        "distance_px": float(chosen["ball_dist"]) if chosen else None,
        "ball_conf": float(ball["conf"]),
    }

    return response


# ----------------------------
# API endpoints
# ----------------------------

@app.get("/")
def root():
    return {
        "service": "ball-closest-object-detector",
        "status": "ok",
        "model": MODEL_PATH,
        "conf": YOLO_CONF,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    if not ext:
        ext = ".jpg"

    input_path = UPLOAD_DIR / f"input_{timestamp}{ext}"

    with open(input_path, "wb") as f:
        f.write(data)

    npbuf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        # result = run_detection(image)
        # return result
        result, debug_image = run_detection_with_debug(image)

        output_path = RESULT_DIR / f"result_{timestamp}.jpg"
        cv2.imwrite(str(output_path), debug_image)

        result["saved_result"] = str(output_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# def run_detection_with_debug(image: np.ndarray):
#     if image is None:
#         raise ValueError("Invalid image")

#     ball, stick = choose_ball_using_stick(image)
#     if ball is None:
#         return {"success": False, "error": "ball_not_detected"}, image

#     results = model.predict(source=image, conf=YOLO_CONF, save=False, verbose=False)
#     if not results:
#         return {"success": False, "error": "no_yolo_results"}, image

#     result = results[0]
#     detections = []

#     if result.boxes is not None:
#         for box in result.boxes:
#             cls_id = int(box.cls[0].item())
#             conf = float(box.conf[0].item())
#             name = model.names[cls_id]
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             detections.append({
#                 "name": name,
#                 "conf": conf,
#                 "box": (x1, y1, x2, y2),
#             })

#     chosen = choose_object(ball, detections)

#     display = image.copy()

#     for det in detections:
#         x1, y1, x2, y2 = det["box"]
#         cv2.rectangle(display, (x1, y1), (x2, y2), (130, 130, 255), 2)
#         cv2.putText(
#             display,
#             f"{det['name']} {det['conf']:.2f}",
#             (x1, max(20, y1 - 6)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (130, 130, 255),
#             1,
#             cv2.LINE_AA,
#         )

#     if stick is not None:
#         (sx1, sy1), (sx2, sy2) = stick
#         cv2.line(display, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)

#     bx, by = ball["center"]
#     br = ball["radius"]
#     cv2.circle(display, (bx, by), br, (0, 255, 0), 2)
#     cv2.circle(display, (bx, by), 4, (255, 0, 255), -1)

#     if chosen is not None:
#         x1, y1, x2, y2 = chosen["box"]
#         cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)

#     response = {
#         "success": True,
#         "closest_object": chosen["name"] if chosen else None,
#         "distance_px": float(chosen["ball_dist"]) if chosen else None,
#         "ball_center": {"x": int(bx), "y": int(by)},
#         "ball_radius": int(br),
#     }

#     return response, display

def run_detection_with_debug(image: np.ndarray):
    if image is None:
        raise ValueError("Invalid image")

    results = model.predict(source=image, conf=YOLO_CONF, save=False, verbose=False)
    if not results:
        return {"success": False, "error": "no_yolo_results"}, image

    result = results[0]

    ball = choose_ball_from_yolo(result, model)
    if ball is None:
        return {"success": False, "error": "sports_ball_not_detected"}, image

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = model.names[cls_id]

            if name in IGNORE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "name": name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

    chosen = choose_object(ball, detections)

    display = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (130, 130, 255), 2)
        cv2.putText(
            display,
            f"{det['name']} {det['conf']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (130, 130, 255),
            1,
            cv2.LINE_AA,
        )

    bx, by = ball["center"]
    br = ball["radius"]
    x1b, y1b, x2b, y2b = ball["bbox"]

    cv2.rectangle(display, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)
    cv2.circle(display, (bx, by), br, (0, 255, 0), 2)
    cv2.circle(display, (bx, by), 4, (255, 0, 255), -1)
    cv2.putText(
        display,
        f"YOLO ball {ball['conf']:.2f}",
        (x1b, max(20, y1b - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if chosen is not None:
        x1, y1, x2, y2 = chosen["box"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.line(display, (bx, by), (cx, cy), (255, 0, 0), 2)

        cv2.putText(
            display,
            f"Closest: {chosen['name']}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    response = {
        "success": True,
        "closest_object": chosen["name"] if chosen else None,
        "distance_px": float(chosen["ball_dist"]) if chosen else None,
        "ball_center": {"x": int(bx), "y": int(by)},
        "ball_radius": int(br),
        "ball_conf": float(ball["conf"]),
    }

    return response, display

if __name__ == "__main__":
    uvicorn.run("yolo_web_server:app", host="0.0.0.0", port=8000)