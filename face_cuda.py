import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque


use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(use_cuda)

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)



LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Parameters for blink detection
# Change according to the requirements
EAR_HISTORY = deque(maxlen=5)
BLINK_THRESHOLD = 0.25
BLINK_DEBOUNCE_TIME = 0.01
BLINK_MIN_DURATION = 0.15
MOVE_X = 1.4
MOVE_Y = 1.4
last_blink_time = 0
blink_start_time = None


screen_width, screen_height = pyautogui.size()


def calculate_ear(eye_landmarks, landmarks):
    
    a = np.linalg.norm(np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) -
                       np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y]))
    b = np.linalg.norm(np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) -
                       np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y]))
    c = np.linalg.norm(np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) -
                       np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y]))
    return (a + b) / (2.0 * c)


def map_coordinates(x, y, frame_width, frame_height, screen_width, screen_height):
    return int(x / frame_width * screen_width), int(y / frame_height * screen_height)



cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)


    if use_cuda:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB).download()
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_height, frame_width, _ = frame.shape


    face_rest = face_detection.process(rgb_frame)
    result = face_mesh.process(rgb_frame)

    if face_rest.detections:
        for detection in face_rest.detections:
            mp_drawing.draw_detection(frame, detection)
            nose = detection.location_data.relative_keypoints[2]
            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

            screen_x, screen_y = map_coordinates(nose_x, nose_y, frame_width, frame_height, screen_width, screen_height)
            pyautogui.moveTo(screen_x * MOVE_X, screen_y * MOVE_Y)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_ear = calculate_ear(LEFT_EYE_LANDMARKS, face_landmarks.landmark)
            right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, face_landmarks.landmark)
            ear = (left_ear + right_ear) / 2.0

   
            EAR_HISTORY.append(ear)
            smoothed_ear = np.mean(EAR_HISTORY)


            if len(EAR_HISTORY) == EAR_HISTORY.maxlen:
                BLINK_THRESHOLD = max(np.mean(EAR_HISTORY) * 0.85, 0.35)

  
            if smoothed_ear < BLINK_THRESHOLD:
                if blink_start_time is None:
                    blink_start_time = time.time()
            else:
                if blink_start_time is not None:
                    blink_duration = time.time() - blink_start_time
                    if blink_duration > BLINK_MIN_DURATION and (time.time() - last_blink_time) > BLINK_DEBOUNCE_TIME:
                        print("Blink detected")
                        pyautogui.click()
                        last_blink_time = time.time()
                    blink_start_time = None

   
    cv2.imshow("Blink to Click (GPU Accelerated)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
