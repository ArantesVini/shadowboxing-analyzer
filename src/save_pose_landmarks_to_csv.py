import csv


def save_pose_landmarks_to_csv(landmarks, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Frame",
                "Shoulder_X",
                "Shoulder_Y",
                "Elbow_X",
                "Elbow_Y",
                "Wrist_X",
                "Wrist_Y",
            ]
        )

        for frame, landmark_data in enumerate(landmarks):
            writer.writerow(
                [
                    frame,
                    landmark_data["shoulder"][0],
                    landmark_data["shoulder"][1],
                    landmark_data["elbow"][0],
                    landmark_data["elbow"][1],
                    landmark_data["wrist"][0],
                    landmark_data["wrist"][1],
                ]
            )
