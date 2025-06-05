import cv2
import os
import numpy as np
import math
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import threading

# Constants
MOVEMENT_THRESHOLD = 10    # Minimum movement in pixels to consider
ROLLING_WINDOW_SIZE = 2    # Number of frames for rolling mean

def detect_wells(frame):
    """Detect circular wells in the static arena."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=290,
        param1=16, param2=15,
        minRadius=160, maxRadius=170
    )
    wells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            wells.append((x, y, r))
    return wells

def draw_detection_box(frame, x0, y0, contour, cx, cy, r):
    """Draw bounding box around the chosen contour and circle for well."""
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), r, (255, 0, 0), 1)

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)

def calculate_rolling_speed(positions, fps):
    """Calculate rolling speed based on multiple positions."""
    if len(positions) < 2:
        return 0

    total_distance = 0
    total_time = 0

    # Calculate cumulative distance and time
    for i in range(1, len(positions)):
        prev_time, prev_x, prev_y = positions[i - 1]
        curr_time, curr_x, curr_y = positions[i]

        distance = calculate_distance(prev_x, prev_y, curr_x, curr_y)
        time_diff = curr_time - prev_time

        # Only count movement above threshold
        if distance >= MOVEMENT_THRESHOLD:
            total_distance += distance
            total_time += time_diff

    # Avoid division by zero
    if total_time > 0:
        return total_distance / total_time
    return 0

def process_well(well_idx, well, frame_idx, fps, frame_width, frame_height,
                 gray_full, clip_limit, thresh_val, min_area, max_area, kernel,
                 frame, debug_mask, csv_writer, write_lock):
    """
    Process a single well:
    1. Extract the grayscale ROI.
    2. Apply CLAHE *per well*.
    3. Threshold, find blobs, then pick the blob whose centroid is closest to
       last frame's centroid (or largest on first detection).
    """
    global well_tracks

    cx, cy, r = well
    # Define bounding-box of square ROI around the circular well
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    # Clip to image bounds
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, frame_width), min(y1, frame_height)
    roi_gray = gray_full[y0c:y1c, x0c:x1c]

    # ─── Apply CLAHE on this ROI ───────────────────────────────────────────────
    # Fixed tileGridSize per well, e.g., (8,8). Adjust if you want different granularity.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi_gray)

    # ─── Threshold the enhanced ROI (fixed or Otsu) ─────────────────────────────
    if thresh_val > 0:
        _, bw = cv2.threshold(roi_enhanced, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(roi_enhanced, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ─── Mask out anything outside the well circle ─────────────────────────────
    h, w = roi_enhanced.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    bw = cv2.bitwise_and(bw, bw, mask=mask)

    # ─── Morphological cleanup ─────────────────────────────────────────────────
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ─── Find connected components and filter by area ─────────────────────────
    num_lbl, labels, stats, cents = cv2.connectedComponentsWithStats(bw)
    if num_lbl <= 1:
        # No blobs detected this frame
        return

    all_areas     = stats[1:, cv2.CC_STAT_AREA]
    all_centroids = cents[1:]
    all_labels    = np.arange(1, num_lbl)

    # Build list of candidates within [min_area, max_area]
    candidates = []
    for lbl_i, area, centroid in zip(all_labels, all_areas, all_centroids):
        if min_area <= area <= max_area:
            candidates.append((lbl_i, (centroid[0], centroid[1]), area))

    if not candidates:
        return  # no blobs within size bounds

    # Retrieve previous centroid (relative to ROI) if exists
    prev_cent = well_tracks[well_idx]['last_centroid']

    if prev_cent is None:
        # First detection: pick the largest-area blob
        chosen_label, (cx_blob_rel, cy_blob_rel), _ = max(candidates, key=lambda x: x[2])
    else:
        # Pick the candidate whose centroid is closest to prev_cent
        distances = [
            (lbl_i,
             centroid,
             area,
             calculate_distance(prev_cent[0], prev_cent[1], centroid[0], centroid[1]))
            for lbl_i, centroid, area in candidates
        ]
        chosen_label, (cx_blob_rel, cy_blob_rel), _, _ = min(distances, key=lambda x: x[3])

    # Update memory
    well_tracks[well_idx]['last_centroid'] = (cx_blob_rel, cy_blob_rel)

    # Convert centroid back to full-frame coordinates
    abs_cx = int(x0c + cx_blob_rel)
    abs_cy = int(y0c + cy_blob_rel)
    t = frame_idx / fps

    # ─── Update tracking and write CSV ─────────────────────────────────────────
    well_tracks[well_idx]['positions'].append((t, abs_cx, abs_cy))
    rolling_speed = calculate_rolling_speed(
        list(well_tracks[well_idx]['positions']), fps
    )
    well_tracks[well_idx]['speeds'].append(rolling_speed)
    smoothed_speed = (
        sum(well_tracks[well_idx]['speeds']) / len(well_tracks[well_idx]['speeds'])
        if well_tracks[well_idx]['speeds'] else 0
    )

    with write_lock:
        csv_writer.writerow([
            well_idx,
            t,
            abs_cx,
            abs_cy,
            rolling_speed,
            smoothed_speed
        ])
        csv_file.flush()

    # ─── Build a mask for the single chosen blob (for visualization) ───────────
    single_blob_mask = (labels == chosen_label).astype(np.uint8) * 255
    debug_mask[y0c:y1c, x0c:x1c] = single_blob_mask

    cnts, _ = cv2.findContours(single_blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return
    c = cnts[0]  # only one contour after connectedComponents

    # ─── Draw box and centroid on the full frame ───────────────────────────────
    draw_detection_box(frame, x0c, y0c, c, cx, cy, r)
    cv2.circle(frame, (abs_cx, abs_cy), 4, (0, 0, 255), -1)

    # ─── Overlay speed text and movement path ─────────────────────────────────
    speed_text = f"{smoothed_speed:.1f} px/s"
    cv2.putText(frame, speed_text, (abs_cx + 10, abs_cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(well_tracks[well_idx]['positions']) > 1:
        _, last_x, last_y = well_tracks[well_idx]['positions'][-2]
        cv2.circle(frame, (last_x, last_y), MOVEMENT_THRESHOLD, (0, 255, 255), 1)
        cv2.line(frame, (last_x, last_y), (abs_cx, abs_cy), (255, 255, 0), 1)


# ── Setup ─────────────────────────────────────────────────────────────────────
exit_key = 'q'
video_path = r"C:\Users\federico97\Desktop\Adrian_heterostegina-depressa\000000.mp4"
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError(f"Could not open video {video_path}")

# Read first frame for well detection
ret, first_frame = capture.read()
if not ret:
    raise RuntimeError("Could not read first frame for well detection")

well_positions = detect_wells(first_frame)
if not well_positions:
    raise RuntimeError("No wells detected. Check parameters.")
print(f"Detected {len(well_positions)} wells")

# Get frame dimensions for ROI clipping
frame_height, frame_width = first_frame.shape[:2]

# Rewind to first frame
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Grab the frame rate
fps = capture.get(cv2.CAP_PROP_FPS)

# Initialize well tracks with deques for rolling calculations and 'last_centroid'
well_tracks = {
    i: {
        'positions':     collections.deque(maxlen=ROLLING_WINDOW_SIZE + 1),
        'speeds':        collections.deque(maxlen=ROLLING_WINDOW_SIZE),
        'last_centroid': None
    }
    for i in range(len(well_positions))
}

frame_idx = 0

csv_file = open("well_centroids.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["well", "time_s", "x_px", "y_px", "speed_px/s", "smoothed_speed_px/s"])
write_lock = threading.Lock()

print("Saving CSV to:", os.path.abspath(csv_file.name))

# ── GUI Setup ─────────────────────────────────────────────────────────────────
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
def nothing(x): pass
cv2.createTrackbar('Thresh',    'Controls', 50, 255, nothing)
cv2.createTrackbar('MinArea',   'Controls', 1, 500, nothing)
cv2.createTrackbar('MaxArea',   'Controls', 350, 1000, nothing)
cv2.createTrackbar('ClipLimit', 'Controls', 0, 1000, nothing)  # ×0.01

cv2.namedWindow("Detections",     cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 1300, 1300)

cv2.namedWindow("Enhanced Gray",   cv2.WINDOW_NORMAL)
cv2.resizeWindow("Enhanced Gray", 900, 900)

cv2.namedWindow("Threshold Mask",  cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold Mask", 800, 800)

# Static morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ── Main Loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 1) Read all sliders each frame
        thresh_val = cv2.getTrackbarPos('Thresh',    'Controls')
        min_area   = cv2.getTrackbarPos('MinArea',   'Controls')
        max_area   = cv2.getTrackbarPos('MaxArea',   'Controls')
        clip_limit = cv2.getTrackbarPos('ClipLimit', 'Controls') / 100.0

        # 2) Convert full frame to grayscale once
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3) Prepare a mask to show all wells' binaries
        debug_mask = np.zeros_like(gray_full)

        # 4) Per-well processing in parallel
        with ThreadPoolExecutor() as executor:
            tasks = [
                (
                    i,
                    well,
                    frame_idx,
                    fps,
                    frame_width,
                    frame_height,
                    gray_full,
                    clip_limit,
                    thresh_val,
                    min_area,
                    max_area,
                    kernel,
                    frame,
                    debug_mask,
                    csv_writer,
                    write_lock
                )
                for i, well in enumerate(well_positions)
            ]
            executor.map(lambda args: process_well(*args), tasks)

        # 5) Show results
        # For visualization purposes, we can display the per-well enhanced ROI masks
        # by showing gray_full (unmodified) alongside the debug_mask and detections.
        cv2.imshow("Enhanced Gray",   gray_full)
        cv2.imshow("Threshold Mask",  debug_mask)
        cv2.imshow("Detections",      frame)

        # Move to next frame
        frame_idx += 1

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            break

finally:
    # Release resources
    csv_file.close()
    capture.release()
    cv2.destroyAllWindows()
