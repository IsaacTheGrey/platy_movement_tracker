import cv2
import os
import numpy as np
import math
import collections
import csv
import time
from concurrent.futures import ThreadPoolExecutor

# ====== Configuration ======
VIDEO_PATH = r"c:\Users\federico97\Desktop\20241129_162150\000000.mp4"
OUTPUT_CSV = "well_speeds_grouped.csv"

# Frame skipping: process every Nth frame (e.g., 2 = every other frame)
SKIP_FRAMES = 2

# Tracking parameters
MOVEMENT_THRESHOLD = 15      # px
ROLLING_WINDOW_SIZE = 5
THRESH_VAL = 120             # fixed threshold
MIN_AREA = 10                # min blob area

# CLAHE parameters (precomputed once)
CLAHE_CLIP_LIMIT = 8.0
CLAHE_TILE_SIZE = (19, 19)

# Morphology kernel (static)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ====== Helper Functions ======
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def calculate_rolling_speed(positions):
    pos = list(positions)
    if len(pos) < 2:
        return 0.0
    total_dist, total_time = 0.0, 0.0
    for (t0, x0, y0), (t1, x1, y1) in zip(pos, pos[1:]):
        d = calculate_distance(x0, y0, x1, y1)
        dt = t1 - t0
        if d >= MOVEMENT_THRESHOLD:
            total_dist += d
            total_time += dt
    return total_dist / total_time if total_time > 0 else 0.0


def process_well_speed(well_idx, roi_info, frame_time, enhanced, wells_tracks, fps):
    x0, y0 = roi_info['x0'], roi_info['y0']
    mask = roi_info['mask']
    h, w = mask.shape
    roi = enhanced[y0:y0+h, x0:x0+w]

    _, bw = cv2.threshold(roi, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.bitwise_and(bw, bw, mask=mask)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    num_lbl, labels, stats, cents = cv2.connectedComponentsWithStats(bw)
    if num_lbl <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if areas[idx-1] < MIN_AREA:
        return 0.0

    cx_blob, cy_blob = cents[idx]
    abs_x, abs_y = x0 + cx_blob, y0 + cy_blob

    # update track
    track = wells_tracks[well_idx]
    track['positions'].append((frame_time, abs_x, abs_y))
    speed = calculate_rolling_speed(track['positions'])
    track['speeds'].append(speed)
    # use latest rolling speed or smoothed
    smoothed = sum(track['speeds']) / len(track['speeds']) if track['speeds'] else 0.0
    return smoothed

# ====== Main ======
if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_total = math.ceil(total_frames / SKIP_FRAMES)

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    # detect wells with full data
    circles = cv2.HoughCircles(
        cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT, dp=1, minDist=200,
        param1=16, param2=15, minRadius=140, maxRadius=150
    )
    if circles is None:
        raise RuntimeError("No wells detected; adjust parameters.")
    circles = np.uint16(np.around(circles))[0]
    # sort by x for consistent column order
    circles = sorted(circles, key=lambda c: c[0])
    num_wells = len(circles)

    # prepare CSV header
    header = [f"well_{i}" for i in range(num_wells)] + ["time_exp"]

    # precompute ROIs & circular masks
    h_full, w_full = first.shape[:2]
    well_rois = []
    for x, y, r in circles:
        x0, y0 = max(x-r, 0), max(y-r, 0)
        x1, y1 = min(x+r, w_full), min(y+r, h_full)
        mask_h, mask_w = y1-y0, x1-x0
        m = np.zeros((mask_h, mask_w), dtype=np.uint8)
        cv2.circle(m, (x-x0, y-y0), r, 255, -1)
        well_rois.append({'x0': x0, 'y0': y0, 'mask': m})

    # initialize tracks for speeds
    wells_tracks = {
        i: {
            'positions': collections.deque(maxlen=ROLLING_WINDOW_SIZE+1),
            'speeds': collections.deque(maxlen=ROLLING_WINDOW_SIZE)
        } for i in range(num_wells)
    }

    fps = cap.get(cv2.CAP_PROP_FPS)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)

    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        orig_idx = 0
        processed_idx = 0
        start = time.time()

        with ThreadPoolExecutor() as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                orig_idx += 1
                if orig_idx % SKIP_FRAMES != 1:
                    continue

                frame_time = orig_idx / fps
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                enhanced = clahe.apply(gray)

                # compute speeds in parallel
                futures = [executor.submit(process_well_speed, i, roi, frame_time, enhanced, wells_tracks, fps)
                           for i, roi in enumerate(well_rois)]
                speeds = [f.result() for f in futures]

                row = [f"{s:.1f}" for s in speeds] + [timestamp]
                writer.writerow(row)

                processed_idx += 1
                if processed_idx % 100 == 0 or processed_idx == processed_total:
                    elapsed = time.time() - start
                    rate = processed_idx / elapsed
                    eta = (processed_total - processed_idx) / rate if rate else float('inf')
                    print(f"Processed {processed_idx}/{processed_total}, {rate:.1f} fps, ETA {eta:.1f}s")

    cap.release()
    total_time = time.time() - start
    print(f"Completed in {total_time:.1f}s. CSV at {os.path.abspath(OUTPUT_CSV)}")
