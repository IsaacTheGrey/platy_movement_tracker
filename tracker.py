import cv2
import numpy as np

def detect_wells(frame):
    """Detect circular wells in the static arena."""
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.medianBlur(gray, 5)
    circles  = cv2.HoughCircles(
                   blurred,
                   cv2.HOUGH_GRADIENT,
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

# ── Setup ─────────────────────────────────────────────────────────────────────
video_path = r"c:\Users\federico97\Desktop\20241129_162150\000000.mp4"
capture    = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError(f"Could not open video {video_path}")

ret, first_frame = capture.read()
if not ret:
    raise RuntimeError("Could not read first frame for well detection")

well_positions = detect_wells(first_frame)
print(f"Detected {len(well_positions)} wells")

# rewind for processing
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ── Controls & Enhancement ────────────────────────────────────────────────────
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
def nothing(x): pass

# Sliders for threshold and minimum area
cv2.createTrackbar('Thresh',  'Controls', 110, 255, nothing)
cv2.createTrackbar('MinArea', 'Controls',  10, 500, nothing)

# CLAHE for contrast boosting
clahe  = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(20,20))
# Morphology kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # 1) Convert & enhance contrast
    gray_full     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_full = clahe.apply(gray_full)

    # 2) Read sliders
    thresh_val = cv2.getTrackbarPos('Thresh',  'Controls')
    min_area   = cv2.getTrackbarPos('MinArea', 'Controls')

    # For visualizing the binary result across all wells
    debug_mask = np.zeros_like(gray_full)

    # 3) Process each well ROI, but draw onto the full frame
    for cx, cy, r in well_positions:
        x0, y0 = cx - r, cy - r
        roi     = enhanced_full[y0:cy+r, x0:cx+r]

        # threshold: fixed if >0 else Otsu
        if thresh_val > 0:
            _, bw = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY_INV)
        else:
            _, bw = cv2.threshold(
                roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )

        # clean up noise
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

        # for tuning: show the combined mask
        debug_mask[y0:cy+r, x0:cx+r] = bw

        # find contours & keep the largest
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            continue

        # draw the detection box on the full frame
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(
            frame,
            (x0 + x,    y0 + y),
            (x0 + x + w, y0 + y + h),
            (0,255,0), 2
        )
        # optional: draw the well outline
        cv2.circle(frame, (cx, cy), r, (255,0,0), 1)

    # 4) Display full-frame outputs
    cv2.imshow("Enhanced Gray",    enhanced_full)
    cv2.imshow("Threshold Mask",   debug_mask)
    cv2.imshow("Detections",       frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
capture.release()
cv2.destroyAllWindows()
