import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def detect_wells(frame):
    """Detect circular wells in the static arena."""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0,255,0), 2)
    cv2.circle(frame, (cx, cy), r, (255,0,0), 1)

# ── Setup ─────────────────────────────────────────────────────────────────────
exit_key   = 'q'
video_path = r"c:\Users\federico97\Desktop\20241129_162150\000000.mp4"
capture    = cv2.VideoCapture(video_path)
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

# ── GUI Setup ─────────────────────────────────────────────────────────────────
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
def nothing(x): pass
cv2.createTrackbar('Thresh',    'Controls', 112, 255, nothing)
cv2.createTrackbar('MinArea',   'Controls',  10, 500, nothing)
cv2.createTrackbar('ClipLimit', 'Controls',  80, 100, nothing)  # ×0.1
cv2.createTrackbar('TileSize',  'Controls',  19,  50, nothing)

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detections", 800, 800)

cv2.namedWindow("Enhanced Gray", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Enhanced Gray", 900, 900)

cv2.namedWindow("Threshold Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold Mask", 800, 800)

# static morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# ── Main Loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 1) Read all sliders each frame
        thresh_val  = cv2.getTrackbarPos('Thresh',    'Controls')
        min_area    = cv2.getTrackbarPos('MinArea',   'Controls')
        clip_limit  = cv2.getTrackbarPos('ClipLimit','Controls') / 10.0
        tile_size   = cv2.getTrackbarPos('TileSize', 'Controls')
        tile_size   = max(1, tile_size)

        # 2) Configure CLAHE dynamically
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=(tile_size, tile_size))

        # 3) Contrast‐enhance the whole frame
        gray_full     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_full = clahe.apply(gray_full)

        # 4) Prepare a mask to show all wells’ binaries
        debug_mask = np.zeros_like(enhanced_full)

        # 5) Per‐well processing in parallel
        def process_well(well):
            cx, cy, r = well
            x0, y0 = cx - r, cy - r
            x1, y1 = cx + r, cy + r
            # clip to image bounds
            x0c, y0c = max(x0, 0), max(y0, 0)
            x1c, y1c = min(x1, frame_width), min(y1, frame_height)
            roi = enhanced_full[y0c:y1c, x0c:x1c]

            # threshold
            if thresh_val > 0:
                _, bw = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
            else:
                _, bw = cv2.threshold(roi, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # morphology cleanup
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  kernel, iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
            #bw = cv2.erode(bw, kernel, iterations=1)


            # accumulate for debug
            debug_mask[y0c:y1c, x0c:x1c] = bw

            # find largest contour
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < min_area:
                return

            # draw on the full frame
            draw_detection_box(frame, x0c, y0c, c, cx, cy, r)

        # fire off per‐well in a thread pool
        with ThreadPoolExecutor() as executor:
            executor.map(process_well, well_positions)

        # 6) Show results
        cv2.imshow("Enhanced Gray",    enhanced_full)
        cv2.imshow("Threshold Mask",   debug_mask)
        cv2.imshow("Detections",       frame)

        # 7) Exit on key
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            break

finally:
    # always release & destroy
    capture.release()
    cv2.destroyAllWindows()
