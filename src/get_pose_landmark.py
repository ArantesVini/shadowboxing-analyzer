import cv2
import numpy as np
import mediapipe as mp

from src.calculate_angle import calculate_angle
from src.save_pose_landmarks_to_csv import save_pose_landmarks_to_csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def get_pose_landmarks(cap, file_name):
    pose_landmarks_data = []
    frame_count = 0

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)
            landmarks = results.pose_landmarks

            if landmarks:
                shoulder_index = 11
                elbow_index = 13
                wrist_index = 15

                shoulder = np.array(
                    [
                        landmarks.landmark[shoulder_index].x,
                        landmarks.landmark[shoulder_index].y,
                    ]
                )
                elbow = np.array(
                    [
                        landmarks.landmark[elbow_index].x,
                        landmarks.landmark[elbow_index].y,
                    ]
                )
                wrist = np.array(
                    [
                        landmarks.landmark[wrist_index].x,
                        landmarks.landmark[wrist_index].y,
                    ]
                )
                shoulder_elbow_wrist_angle = calculate_angle(shoulder, elbow, wrist)

                landmark_data = {
                    "frame": frame_count,
                    "shoulder": shoulder,
                    "elbow": elbow,
                    "wrist": wrist,
                    "angle": shoulder_elbow_wrist_angle,
                }

                pose_landmarks_data.append(landmark_data)
                cv2.putText(
                    image,
                    f"Angle: {shoulder_elbow_wrist_angle:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Pose Estimation", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    save_pose_landmarks_to_csv(pose_landmarks_data, file_name)
