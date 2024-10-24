import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose class.
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points.
def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by the line segments 'ab' and 'bc'.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    # Calculate the angle.
    ba = a - b
    bc = c - b

    # Compute the cosine of the angle between vectors ba and bc.
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clamp the cosine value to the range [-1, 1] to avoid numerical errors.
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Compute the angle in degrees.
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return angle

# Initialize counter and stage variables.
counter = 0
stage = None  # Indicates whether the arm is 'up' or 'down'.

# Initialize video capture object.
cap = cv2.VideoCapture(0)

# Set up Mediapipe Pose detection.
with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to read from camera. Exiting...")
            break

        # Flip the frame horizontally for natural (mirror) viewing.
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe processing.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose detection.
        results = pose.process(image_rgb)

        # Extract landmarks.
        try:
            landmarks = results.pose_landmarks.landmark

            # Get image dimensions.
            image_height, image_width, _ = frame.shape

            # Extract coordinates for the right arm.
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Convert normalized coordinates to pixel values.
            shoulder = [int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)]
            elbow = [int(right_elbow.x * image_width), int(right_elbow.y * image_height)]
            wrist = [int(right_wrist.x * image_width), int(right_wrist.y * image_height)]

            # Calculate the angle at the elbow.
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize the angle on the elbow.
            cv2.putText(frame, str(int(angle)),
                        tuple(elbow),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Flexion detection logic.
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(f"Flexion count: {counter}")

            # Draw the skeleton using OpenCV functions.
            # Draw circles at key points.
            cv2.circle(frame, tuple(shoulder), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, tuple(elbow), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, tuple(wrist), 5, (0, 255, 0), cv2.FILLED)

            # Draw lines between the points.
            cv2.line(frame, tuple(shoulder), tuple(elbow), (0, 255, 0), 2)
            cv2.line(frame, tuple(elbow), tuple(wrist), (0, 255, 0), 2)

        except Exception as e:
            # In case landmarks are not detected.
            pass

        # Display the counter on the image using OpenCV functions.
        cv2.rectangle(frame, (0, 0), (225, 75), (245, 117, 16), -1)
        cv2.putText(frame, 'REPS', (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'STAGE', (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, stage if stage else '', (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the image.
        cv2.imshow('Flexion Detector', frame)

        # Break loop on 'q' key press.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()
