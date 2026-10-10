import cv2
import mediapipe as mp
import time


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        # Convert OpenCV image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Current time for MediaPipe tracking
        timestamp_ms = int(time.monotonic() * 1000)

        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Check if a hand was detected
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Draw all 21 landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        cv2.imshow("Touchless Toolbox", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()