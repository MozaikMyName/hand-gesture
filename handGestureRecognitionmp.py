import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import math

MAX_HANDS = 2
MIN_DET_CONF = 0.6
MIN_TRACK_CONF = 0.6

GESTURE_BUFFER_LEN = 8
GESTURE_CONFIRM_COUNT = 4  

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=MAX_HANDS,
                       min_detection_confidence=MIN_DET_CONF,
                       min_tracking_confidence=MIN_TRACK_CONF)

gesture_buffer = deque(maxlen=GESTURE_BUFFER_LEN)

def landmarks_to_pixels(landmarks, w, h):
    """Convert normalized landmarks to list of (x,y) in pixels."""
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def hand_size_px(pts):
    """Estimasi ukuran tangan (diagonal bbox) untuk scale threshold."""
    arr = np.array(pts)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    diag = np.linalg.norm([x_max - x_min, y_max - y_min])
    return max(1.0, diag)

def finger_states(norm_landmarks, handedness_label):
    """
    Kembalikan list boolean [thumb, index, middle, ring, pinky]
    True = finger extended / up
    norm_landmarks: list of (x,y) normalized
    handedness_label: "Left" or "Right"
    """
    
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]  

    states = [False]*5

    t_tip_x = norm_landmarks[4][0]
    t_ip_x = norm_landmarks[3][0]
    if handedness_label.lower().startswith("r"):
        states[0] = t_tip_x > t_ip_x
    else:
        states[0] = t_tip_x < t_ip_x

    for i, (tip, pip) in enumerate(zip(tip_ids[1:], pip_ids[1:]), start=1):
        tip_y = norm_landmarks[tip][1]
        pip_y = norm_landmarks[pip][1]
        states[i] = tip_y < pip_y 

    return states  

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def detect_gesture_from_states(states, norm_landmarks, hand_px_pts):
    """
    states: [thumb, index, middle, ring, pinky] booleans
    norm_landmarks: list of (x,y) normalized
    hand_px_pts: list pixel coords -> untuk ukuran/threshold
    """
    thumb, idx, mid, ring, pinky = states
    size = hand_size_px(hand_px_pts)
    touch_thresh = 0.18 * size  
    
    idx_tip_px = hand_px_pts[8]
    thumb_tip_px = hand_px_pts[4]
    dist_thumb_index = euclid(thumb_tip_px, idx_tip_px)

    if dist_thumb_index < touch_thresh:
        return "OK"

    if thumb and idx and not mid and not ring and pinky:
        return "I Love You"

    if idx and not mid and not ring and pinky:
        return "Metal"

    if idx and mid and not ring and not pinky:
        return "Peace"

    if idx and not mid and not ring and not pinky and not thumb:
        return "Point"

    if thumb and not idx and not mid and not ring and not pinky:
        return "Thumbs Up"

    if not thumb and not idx and not mid and not ring and not pinky:
        return "Fist"

    if thumb and idx and mid and ring and pinky:
        return "Open Palm"

    return "Gesture tidak diketahui"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

p_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gestures_in_frame = []

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):
            label = hand_handedness.classification[0].label

            norm_lms = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            pts_px = landmarks_to_pixels(hand_landmarks, w, h)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            states = finger_states(norm_lms, label)
            gesture = detect_gesture_from_states(states, norm_lms, pts_px)
            gestures_in_frame.append(gesture)

            arr = np.array(pts_px)
            x_min, y_min = arr[:,0].min()-10, arr[:,1].min()-10
            x_max, y_max = arr[:,0].max()+10, arr[:,1].max()+10
            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,200,0), 2)
            cv2.putText(frame, f"{label} | {gesture}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if gestures_in_frame:
        choice = Counter(gestures_in_frame).most_common(1)[0][0]
        gesture_buffer.append(choice)
    else:
        gesture_buffer.append("")

    most_common, count = Counter(gesture_buffer).most_common(1)[0]
    display_gesture = most_common if (most_common != "" and count >= GESTURE_CONFIRM_COUNT) else ""

    if display_gesture:
        cv2.rectangle(frame, (10,10), (260,70), (50,50,50), -1)
        cv2.putText(frame, f"Gesture: {display_gesture}", (20,45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,255), 2)
    else:
        cv2.rectangle(frame, (10,10), (160,50), (50,50,50), -1)
        cv2.putText(frame, f"Gesture: ...", (20,35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,200), 1)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (w-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Hand Gesture (upgrade)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Hand Gesture (upgrade)", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
