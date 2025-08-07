# drowsiness_detector.py

from imutils.video import VideoStream
from imutils import face_utils
from eye_ear import eye_aspect_ratio
import pygame
import numpy as np
import imutils
import dlib
import cv2

# Initialize the alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3") # Replace with your alarm sound file

# Define two constants for the eye aspect ratio threshold and
# a counter for consecutive frames
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30

# Initialize the frame counter and a boolean to track the alarm state
COUNTER = 0
ALARM_ON = False

# Initialize dlib's face detector and then create the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()

while True:
    # Grab the frame from the video stream, resize it, and convert it to grayscale
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates and compute the EAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize the eyes on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    pygame.mixer.music.play(-1) # Play alarm indefinitely

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                pygame.mixer.music.stop()

        # Display the EAR on the frame for debugging
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # If the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()