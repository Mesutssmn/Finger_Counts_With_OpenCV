import cv2
import numpy as np
import math
import mediapipe as mp

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two landmark points.
    
    Parameters:
    a (mp.framework.formats.landmark_pb2.NormalizedLandmark): First landmark point
    b (mp.framework.formats.landmark_pb2.NormalizedLandmark): Second landmark point
    
    Returns:
    float: Euclidean distance between the two points
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.5,
                       max_num_hands=2)

def count_open_fingers(hand_landmarks, handedness):
    """
    Count the number of open fingers for a detected hand.
    
    Parameters:
    hand_landmarks (list): List of 21 landmarks representing a hand
    handedness (str): 'Right' or 'Left' to indicate the hand's orientation
    
    Returns:
    list: List of names of open fingers
    """
    open_fingers = []
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Thumb check using handedness
    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[2]
    
    if handedness == "Right":
        # Right hand: thumb is open if tip is left of MCP (lower x)
        if thumb_tip.x < thumb_mcp.x:
            open_fingers.append(finger_names[0])
    else:
        # Left hand: thumb is open if tip is right of MCP (higher x)
        if thumb_tip.x > thumb_mcp.x:
            open_fingers.append(finger_names[0])

    # Other fingers: Open if tip is above PIP joint (y-coordinate is lower)
    if hand_landmarks[8].y < hand_landmarks[6].y:
        open_fingers.append(finger_names[1])
    if hand_landmarks[12].y < hand_landmarks[10].y:
        open_fingers.append(finger_names[2])
    if hand_landmarks[16].y < hand_landmarks[14].y:
        open_fingers.append(finger_names[3])
    if hand_landmarks[20].y < hand_landmarks[18].y:
        open_fingers.append(finger_names[4])

    return open_fingers

# Start video capture
cap = cv2.VideoCapture(0)

# Set the desired window size   
window_width = 1280
window_height = 720

# Set the font scale for text overlay
font_scale = 0.7  # Decreased from 1.0 to 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    total_open_fingers = 0
    left_text = ""
    right_text = ""
    left_text_size = (0, 0)  
    right_text_size = (0, 0) 

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (left or right hand)
            handedness = results.multi_handedness[hand_index].classification[0].label
            
            open_fingers = count_open_fingers(hand_landmarks.landmark, handedness)
            total_open_fingers += len(open_fingers)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare text display for left or right hand
            if handedness == "Left":
                left_text = f'Left: {len(open_fingers)} Fingers ({", ".join(open_fingers)})'
            else:
                right_text = f'Right: {len(open_fingers)} Fingers ({", ".join(open_fingers)})'

    # Display total open fingers at the top of the frame
    total_text = f'Total Open Fingers: {total_open_fingers}'
    total_text_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    cv2.putText(frame, total_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Display left-hand information
    if left_text:
        left_text_size = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.putText(frame, left_text, (50, 50 + total_text_size[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Display right-hand information
    if right_text:
        right_text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        cv2.putText(frame, right_text, (50, 50 + total_text_size[1] + left_text_size[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # Resize the frame to the desired window size
    resized_frame = cv2.resize(frame, (window_width, window_height))

    # Display the processed frame
    cv2.imshow('Finger Count', resized_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
