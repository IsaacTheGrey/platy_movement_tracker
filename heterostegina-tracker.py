import cv2
import os
import numpy as np
import math
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import threading

# Constants
MOVEMENT_THRESHOLD = 1      # Minimum movement in pixels to consider
FRAME_SKIP = 9              # Number of frames to skip between processed frames

# Detect circular wells in the static arena
def detect_wells(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=290,
        param1=16, param2=15,
        minRadius=145, maxRadius=145
    )
    wells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            wells.append((x, y, r))
    return wells

# Draw bounding box, well circle, and ID label
def draw_detection_box(frame, well_idx, x0, y0, contour, cx, cy, r, color):
    x, y, w_, h_ = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w_, y0 + y + h_), color, 2)
    cv2.putText(frame, f"Cell (Well {well_idx})", (x0 + x, y0 + y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.circle(frame, (cx, cy), r, (0, 255, 255), 2)  # high-contrast yellow circle

# Euclidean distance
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Process a single well for one frame
def process_well(well_idx, well, t, frame_w, frame_h,
                 gray_full, clip_limit, thresh_val, min_area, max_area,
                 kernel, frame, debug_mask, csv_writer, write_lock, color):
    global well_tracks, csv_file
    cx, cy, r = well
    x0, y0 = cx - r, cy - r
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(cx + r, frame_w), min(cy + r, frame_h)
    roi = gray_full[y0c:y1c, x0c:x1c]

    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(roi)

    # threshold
    _, bw = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # mask outside circle
    mask = np.zeros_like(bw)
    cv2.circle(mask, (r, r), r, 255, -1)
    bw = cv2.bitwise_and(bw, bw, mask=mask)

    # morphology
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=3)

    # update debug mask
    debug_mask[y0c:y1c, x0c:x1c] |= bw

    # connected components
    num_lbl, labels, stats, cents = cv2.connectedComponentsWithStats(bw)
    if num_lbl <= 1:
        return
    areas = stats[1:, cv2.CC_STAT_AREA]
    centers = cents[1:]
    lbls = np.arange(1, num_lbl)

    # filter by area
    candidates = [(lbl, cent, area) for lbl, cent, area in zip(lbls, centers, areas)
                  if min_area <= area <= max_area]
    if not candidates:
        return

    # choose candidate by nearest previous
    prev = well_tracks[well_idx]['last_centroid']
    if prev is None:
        chosen_lbl, (cb, cb2), _ = max(candidates, key=lambda x: x[2])
    else:
        distances = [(lbl, cent, area, calculate_distance(prev[0], prev[1], cent[0], cent[1]))
                     for lbl, cent, area in candidates]
        chosen_lbl, (cb, cb2), _, _ = min(distances, key=lambda x: x[3])

    # absolute centroid
    abs_x = int(x0c + cb)
    abs_y = int(y0c + cb2)

    # distance
    prev_full = well_tracks[well_idx]['last_full']
    dist = calculate_distance(prev_full[0], prev_full[1], abs_x, abs_y) if prev_full else 0.0
    well_tracks[well_idx]['last_full'] = (abs_x, abs_y)
    well_tracks[well_idx]['last_centroid'] = (cb, cb2)

    # write csv with timestamp
    with write_lock:
        csv_writer.writerow([well_idx, f"{t:.2f}", f"{dist:.2f}"])
        csv_file.flush()

    # draw box
    mask_choice = (labels == chosen_lbl).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_choice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        draw_detection_box(frame, well_idx, x0c, y0c, cnts[0], cx, cy, r, color)

# Setup
if __name__ == '__main__':
    video_path = r"C:\Users\federico97\Desktop\Adrian_heterostegina-depressa\full_1frame_per_minute.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video_path}")
    ret, first = cap.read()
    if not ret: raise RuntimeError("Cannot read first frame")

    wells = detect_wells(first)
    if not wells: raise RuntimeError("No wells found")
    h, w = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    # track state
    well_tracks = {i: {'last_centroid': None, 'last_full': None} for i in range(len(wells))}

    # CSV
    csv_file = open('well_distances.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['well', 'time_s', 'distance_px'])
    lock = threading.Lock()

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps/(FRAME_SKIP+1), (w, h))

    # windows
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    def nothing(x): pass
    cv2.createTrackbar('Thresh', 'Controls', 9, 255, nothing)
    cv2.createTrackbar('MinArea', 'Controls', 27, 1000, nothing)
    cv2.createTrackbar('MaxArea', 'Controls', 350, 5000, nothing)
    cv2.createTrackbar('Clip', 'Controls', 0, 500, nothing)

    cv2.namedWindow('Enhanced Gray', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Threshold Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)

    # static overlay
    overlay = np.zeros_like(first)
    for idx, (cx, cy, r) in enumerate(wells):
        cv2.circle(overlay, (cx, cy), r, (0, 255, 255), 2)
        cv2.putText(overlay, str(idx), (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    try:
        while True:
            for _ in range(FRAME_SKIP): cap.grab()
            ret, frame = cap.read()
            if not ret: break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # read controls
            tv = cv2.getTrackbarPos('Thresh', 'Controls')
            ma = cv2.getTrackbarPos('MinArea', 'Controls')
            xa = cv2.getTrackbarPos('MaxArea', 'Controls')
            cl = cv2.getTrackbarPos('Clip', 'Controls') / 100.0

            debug = np.zeros_like(gray)
            tasks = [
                (i, wells[i], t, w, h, gray, cl, tv, ma, xa,
                 kernel, frame, debug, csv_writer, lock,
                 (255, 0, 0) if i % 2 else (0, 255, 0)) for i in range(len(wells))
            ]
            with ThreadPoolExecutor() as ex:
                ex.map(lambda p: process_well(*p), tasks)

            # show windows
            enhanced_full = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8,8)).apply(gray)
            cv2.imshow('Enhanced Gray', enhanced_full)
            cv2.imshow('Threshold Mask', debug)

            # write and show detections with timestamp overlay
            out_frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
            cv2.putText(out_frame, f"Time: {t:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            out.write(out_frame)
            cv2.imshow('Detections', out_frame)

            if cv2.waitKey(1) == 27:
                break
    finally:
        csv_file.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
