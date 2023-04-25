# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model
import pyautogui
import time
import threading

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

exit_thread = False
gesture_detected = False
className = ''
# Initialize the webcam
cap = cv2.VideoCapture(0)

# TEST

# Define function to check gesture


def check_gesture():
    global gesture_detected, className

    while not exit_thread:
        if className == 'thumbs down':
            print('DOWN')
            pyautogui.press('volumedown')
            gesture_detected = True
        elif className == 'thumbs up':
            print('UP')
            pyautogui.press('volumeup')
            gesture_detected = True
        elif className == 'stop':
            pyautogui.screenshot('my_screenshot.png')
            gesture_detected = True
        elif className == 'rock':
            pyautogui.hotkey('ctrl', 'shift', 'esc')
            gesture_detected = True
        elif className == 'okay':
            pyautogui.hotkey('winleft', 'shift', 's')
            gesture_detected = True
        elif className == 'fist':
            pyautogui.hotkey('fn', 'f5')
            gesture_detected = True
        else:

            gesture_detected = False

        time.sleep(1)

    print("Exiting thread")


# Start gesture detection thread
gesture_thread = threading.Thread(target=check_gesture)
gesture_thread.start()
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    # Perform action based on detected gesture
    if gesture_detected:

        print('dziala')
        gesture_detected = False
        className = ''

    # Exit program on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        exit_thread = True
        break
# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
