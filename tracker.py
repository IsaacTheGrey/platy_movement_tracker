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
                   param1=18, param2=16,
                   minRadius=140, maxRadius=155
               )
    wells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0]:
            wells.append((x, y, r))
    return wells

# ── Setup ─────────────────────────────────────────────────────────────────────
video_path      = r"c:\Users\federico97\Desktop\20241129_162150\000000.mp4"
capture         = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError(f"Could not open video {video_path}")

# get first frame → detect wells → build well mask
ret, first_frame = capture.read()
if not ret:
    raise RuntimeError("Could not read first frame for well detection")

well_positions = detect_wells(first_frame)
print(f"Detected {len(well_positions)} wells")

# draw wells for verification
for cx, cy, r in well_positions:
    cv2.circle(first_frame, (cx, cy), r, (0, 0, 255), 2)
    cv2.circle(first_frame, (cx, cy),   2, (0, 255,   0), 3)
cv2.imshow("Detected Wells", first_frame)

# create a binary mask of wells
well_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
for cx, cy, r in well_positions:
    cv2.circle(well_mask, (cx, cy), r, 255, thickness=-1)

# reset video
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# create trackbars
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
def nothing(x): pass
cv2.createTrackbar('VarThresh','Controls',32,100,nothing)
cv2.createTrackbar('MinArea',  'Controls', 10,500,nothing)

# initialize detector & CLAHE
object_detector = cv2.createBackgroundSubtractorMOG2(history=300)
clahe           = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(12,12))

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # get parameters
    var_thresh = cv2.getTrackbarPos('VarThresh', 'Controls')
    min_area   = cv2.getTrackbarPos('MinArea',   'Controls')
    object_detector.setVarThreshold(var_thresh)

    # 1) enhance contrast
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced   = clahe.apply(gray)
    enhanced   = cv2.GaussianBlur(enhanced, (5,5), 0)

    # 2) background subtraction
    fg_mask    = object_detector.apply(enhanced)

    # 3) restrict to wells
    masked_fg  = cv2.bitwise_and(fg_mask, fg_mask, mask=well_mask)

    # 4) detect contours
    contours, _ = cv2.findContours(
        masked_fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # 5) display
    cv2.imshow("Enhanced Gray", enhanced)
    cv2.imshow("Wells-Masked FG", masked_fg)
    cv2.imshow("Detections",   frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
