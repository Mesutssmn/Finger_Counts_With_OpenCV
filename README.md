# Finger Counting System using MediaPipe and OpenCV

## Overview

This project implements a real-time finger counting system using the MediaPipe Hands module and OpenCV. The program detects hands, identifies open fingers, and displays the count for both the left and right hands individually.

## Steps

### 1. Import Necessary Libraries

  OpenCV (cv2) for video capture and display.
  
  NumPy (numpy) for numerical operations.
  
  Math (math) for distance calculations.
  
  MediaPipe (mediapipe) for hand detection and tracking.

### 2. Define Helper Functions

  euclidean_distance(a, b): Calculates the Euclidean distance between two points.
  
  count_open_fingers(hand_landmarks, handedness): Determines the number of open fingers for a detected hand.

### 3. Initialize MediaPipe Hands

  The Hands module is initialized with:
  
  min_detection_confidence=0.7: Minimum confidence level for detecting a hand.
  
  min_tracking_confidence=0.5: Minimum confidence level for tracking hand landmarks.
  
  max_num_hands=2: Allows detecting up to two hands.

### 4. Start Video Capture

  OpenCV captures video from the default webcam.
  
  The video frame size is set to 1280x720.
  
  A font scale of 0.7 is used for displaying text on the screen.

### 5. Process Video Frames

  Read each frame from the webcam.
  
  Convert the frame to RGB (required for MediaPipe processing).
  
  Use hands.process() to detect hands in the frame.

### 6. Analyze Detected Hands

  If hands are detected:
  
  Iterate over detected hands.
  
  Determine handedness (Left or Right).
  
  Identify open fingers using count_open_fingers().
  
  Draw hand landmarks using mp_drawing.draw_landmarks().
  
  Display the number of open fingers for each hand on the frame.

### 7. Display Results on Frame

  Show the number of open fingers for the left and right hands.
  
  Display the total number of open fingers detected.
  
### 8. Resize and Show the Frame
  
  Resize the frame to 1280x720 for display.
  
  Show the frame in a window titled Finger Count.

### 9. Exit Condition

  Press 'q' to exit the program.
  
  Release the webcam and close all OpenCV windows.

## Usage

Run the script using Python:

***python finger_count.py***

Make sure your webcam is connected and accessible.

## Requirements

Install the required libraries before running the script:

***pip install opencv-python numpy mediapipe***

Notes

The script detects both left and right hands accurately using the MediaPipe handedness classification.

The thumb is detected based on handedness to avoid false detections.

The text is displayed dynamically based on detected hands.

## Future Improvements

Rock-Paper-Scissors Games
