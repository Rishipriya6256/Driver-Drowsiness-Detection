import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import math
import os
from twilio.rest import Client

# ------------------ TWILIO CONFIG ------------------
import os

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = '+918686160750'   # Twilio number
EMERGENCY_NUMBER = '911'    # Recipient number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_emergency_sms():
    try:
        message = client.messages.create(
            body="Driver unresponsive for 5 minutes! Please check immediately!",
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_NUMBER
        )
        print("Emergency SMS sent:", message.sid)
    except Exception as e:
        print("Failed to send SMS:", e)

# ------------------ INITIALIZATIONS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

ALERT_DURATION_LIMIT = 300  # 5 minutes in seconds

# EAR, MAR, Head tilt functions same as before
def get_ear(landmarks, eye_indices, h, w):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

def get_mar(landmarks, mouth_indices, h, w):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = np.linalg.norm(np.array(points[13]) - np.array(points[19]))
    B = np.linalg.norm(np.array(points[14]) - np.array(points[18]))
    C = np.linalg.norm(np.array(points[15]) - np.array(points[17]))
    D = np.linalg.norm(np.array(points[12]) - np.array(points[16]))
    return (A + B + C) / (3.0 * D)

def get_head_tilt(landmarks, h, w):
    left_ear_tip = (int(landmarks[234].x * w), int(landmarks[234].y * h))
    right_ear_tip = (int(landmarks[454].x * w), int(landmarks[454].y * h))
    dx = right_ear_tip[0] - left_ear_tip[0]
    dy = right_ear_tip[1] - left_ear_tip[1]
    return abs(math.degrees(math.atan2(dy, dx)))

def start_alert():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_alert():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# ------------------ MAIN FUNCTION ------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    eye_thresh_val = 23
    tilt_thresh_val = 25
    mar_adj_val = 100

    baseline_mar = None
    calib_counter = 0
    mar_sum = 0
    frames_to_calibrate = 50
    yawning_prev = False
    drowsy_prev = False
    alert_start_time = None
    emergency_triggered = False

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            status = "Normal"
            drowsy = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                         291, 308, 402, 317, 14, 87, 178, 88, 95, 78]

                left_ear = get_ear(landmarks, left_eye, h, w)
                right_ear = get_ear(landmarks, right_eye, h, w)
                ear = (left_ear + right_ear) / 2.0
                mar = get_mar(landmarks, mouth, h, w)
                tilt_angle = get_head_tilt(landmarks, h, w)

                # --- Yawn calibration ---
                if baseline_mar is None:
                    mar_sum += mar
                    calib_counter += 1
                    if calib_counter >= frames_to_calibrate:
                        baseline_mar = mar_sum / frames_to_calibrate
                        print(f"Calibrated baseline MAR: {baseline_mar:.3f}")
                    status = "Calibrating..."
                    yawning = False
                else:
                    upper_thresh = baseline_mar * 1.45 * (mar_adj_val/100)
                    lower_thresh = baseline_mar * 1.25 * (mar_adj_val/100)
                    if mar > upper_thresh:
                        yawning = True
                        yawning_prev = True
                    elif mar < lower_thresh:
                        yawning = False
                        yawning_prev = False
                    else:
                        yawning = yawning_prev

                eye_closed = ear < (eye_thresh_val/100)
                head_tilted = tilt_angle > tilt_thresh_val
                drowsy = eye_closed or yawning or head_tilted

                if eye_closed:
                    status = "Eyes Closed"
                elif yawning:
                    status = "Yawning"
                elif head_tilted:
                    status = "Head Tilted"
                else:
                    status = "Normal"

                # --- Alert and Emergency SMS ---
                if drowsy and not drowsy_prev:
                    start_alert()
                    alert_start_time = time.time()
                    emergency_triggered = False
                elif not drowsy and drowsy_prev:
                    stop_alert()
                    alert_start_time = None
                    emergency_triggered = False

                if alert_start_time:
                    elapsed = time.time() - alert_start_time
                    if elapsed >= ALERT_DURATION_LIMIT and not emergency_triggered:
                        emergency_triggered = True
                        send_emergency_sms()
                        cv2.putText(frame, f"EMERGENCY! SMS Sent to {EMERGENCY_NUMBER}",
                                    (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

                drowsy_prev = drowsy

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0],
                                          mp_face_mesh.FACEMESH_CONTOURS)
                color = (0, 0, 255) if drowsy else (0, 255, 0)
                cv2.putText(frame,
                            f"EAR:{ear:.2f} MAR:{mar:.2f} Tilt:{tilt_angle:.1f}° Status:{status}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Driver Fatigue Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    stop_alert()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
