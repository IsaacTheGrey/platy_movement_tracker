import cv2
import os
import numpy as np
import math
import collections
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import csv
import threading

# Constants
MOVEMENT_THRESHOLD = 15    # Minimum movement in pixels to consider
ROLLING_WINDOW_SIZE = 5   # Number of frames for rolling mean

def detect_wells(frame):
    """Detect circular wells in the static arena."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=200,
        param1=16, param2=15,
        minRadius=140, maxRadius=150
    )
    wells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            wells.append((x, y, r))
    return wells

def draw_detection_box(frame, x0, y0, contour, cx, cy, r):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), r, (255, 0, 0), 1)

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_rolling_speed(positions, fps):
    """Calculate rolling speed based on multiple positions."""
    if len(positions) < 2:
        return 0
    
    total_distance = 0
    total_time = 0
    
    # Calculate cumulative distance and time
    for i in range(1, len(positions)):
        prev_time, prev_x, prev_y = positions[i-1]
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

def process_well(well_idx, well, frame_idx, fps, frame_width, frame_height, enhanced_full, 
                 thresh_val, min_area, kernel, frame, debug_mask, csv_writer, write_lock):
    global well_tracks
    
    cx, cy, r = well
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    # clip to image bounds
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, frame_width), min(y1, frame_height)
    roi = enhanced_full[y0c:y1c, x0c:x1c]

    # --- make a circular mask the same size as roi ---
    h, w = roi.shape
    # center of circle in ROI coords:
    cy_roi, cx_roi = r, r
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx_roi, cy_roi), r, 255, -1)

    # threshold
    if thresh_val > 0:
        _, bw = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # --- apply circular mask so outside-circle is zeroed out ---
    bw = cv2.bitwise_and(bw, bw, mask=mask)

    # morphology cleanup
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_lbl, labels, stats, cents = cv2.connectedComponentsWithStats(bw)
    if num_lbl <= 1:
        return  # no blobs
    
    # get areas of each blob (skip background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if areas[idx-1] < min_area:
        return

    # centroid in ROI coords
    cx_blob, cy_blob = cents[idx]
    # convert to frame coords
    abs_cx = int(x0c + cx_blob)
    abs_cy = int(y0c + cy_blob)
    t = frame_idx / fps

    # Store current position
    well_tracks[well_idx]['positions'].append((t, abs_cx, abs_cy))
    
    # Calculate rolling speed
    rolling_speed = calculate_rolling_speed(
        list(well_tracks[well_idx]['positions']),
        fps
    )
    
    # Store speed for rolling mean calculation
    well_tracks[well_idx]['speeds'].append(rolling_speed)
    
    # Calculate final smoothed speed (mean of recent speeds)
    if well_tracks[well_idx]['speeds']:
        smoothed_speed = sum(well_tracks[well_idx]['speeds']) / len(well_tracks[well_idx]['speeds'])
    else:
        smoothed_speed = 0

    # Write to CSV
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
    
    # accumulate for debug
    debug_mask[y0c:y1c, x0c:x1c] = bw

    # find largest contour
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return

    # draw on the full frame
    draw_detection_box(frame, x0c, y0c, c, cx, cy, r)
    cv2.circle(frame, (abs_cx, abs_cy), 4, (0, 0, 255), -1)

    # Display ROLLING MEAN SPEED (changed from instant speed)
    speed_text = f"{smoothed_speed:.1f} px/s"
    cv2.putText(frame, speed_text, (abs_cx + 10, abs_cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Visual feedback for movement tracking
    if len(well_tracks[well_idx]['positions']) > 1:
        _, last_x, last_y = well_tracks[well_idx]['positions'][-2]
        # Draw movement threshold circle
        cv2.circle(frame, (last_x, last_y), MOVEMENT_THRESHOLD, (0, 255, 255), 1)
        # Draw movement path
        cv2.line(frame, (last_x, last_y), (abs_cx, abs_cy), (255, 255, 0), 1)

# ── Setup ─────────────────────────────────────────────────────────────────────
exit_key = 'q'
video_path = r"c:\Users\federico97\Desktop\20241129_162150\000000.mp4"
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError(f"Could not open video {video_path}")

ret, first_frame = capture.read()
if not ret:
    raise RuntimeError("Could not read first frame for well detection")

well_positions = detect_wells(first_frame)
if not well_positions:
    raise RuntimeError("No wells detected. Check parameters.")
print(f"Detected {len(well_positions)} wells")

# get frame dimensions for ROI clipping
frame_height, frame_width = first_frame.shape[:2]

# rewind
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# grabbing the frame rate
fps = capture.get(cv2.CAP_PROP_FPS)

# Initialize well tracks with deques for rolling calculations
well_tracks = {
    i: {
        'positions': collections.deque(maxlen=ROLLING_WINDOW_SIZE+1),
        'speeds': collections.deque(maxlen=ROLLING_WINDOW_SIZE)
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
cv2.createTrackbar('Thresh', 'Controls', 120, 255, nothing)
cv2.createTrackbar('MinArea', 'Controls', 10, 500, nothing)
cv2.createTrackbar('ClipLimit', 'Controls', 80, 100, nothing)  # ×0.1
cv2.createTrackbar('TileSize', 'Controls', 19, 50, nothing)

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 1300, 1300)

cv2.namedWindow("Enhanced Gray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Enhanced Gray", 900, 900)

cv2.namedWindow("Threshold Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold Mask", 800, 800)

# static morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Main Loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 1) Read all sliders each frame
        thresh_val = cv2.getTrackbarPos('Thresh', 'Controls')
        min_area = cv2.getTrackbarPos('MinArea', 'Controls')
        clip_limit = cv2.getTrackbarPos('ClipLimit', 'Controls') / 10.0
        tile_size = cv2.getTrackbarPos('TileSize', 'Controls')
        tile_size = max(1, tile_size)

        # 2) Configure CLAHE dynamically
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        # 3) Contrast-enhance the whole frame
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_full = clahe.apply(gray_full)

        # 4) Prepare a mask to show all wells' binaries
        debug_mask = np.zeros_like(enhanced_full)

        # 5) Per-well processing in parallel
        with ThreadPoolExecutor() as executor:
            # Build list of arguments for each well
            tasks = [(i, well, frame_idx, fps, frame_width, frame_height, 
                      enhanced_full, thresh_val, min_area, kernel, 
                      frame, debug_mask, csv_writer, write_lock) 
                     for i, well in enumerate(well_positions)]
            
            # Execute processing for all wells
            executor.map(lambda args: process_well(*args), tasks)

        # 6) Show results
        cv2.imshow("Enhanced Gray", enhanced_full)
        cv2.imshow("Threshold Mask", debug_mask)
        cv2.imshow("Detections", frame)

        # go to next frame
        frame_idx += 1

        # 7) Exit on key
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            break

finally:
    # release & destroy
    csv_file.close()
    capture.release()
    cv2.destroyAllWindows()
