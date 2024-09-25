import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import distance as d
import numpy as np
import pyautogui


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]

left_click = False
right_click = False
swipe_start = None
pyautogui.FAILSAFE = False
alt_key_pressed = False
horizontal_swipe_threshold = 60  # Adjust this threshold as needed
# Get the screen resolution
screen_width, screen_height = pyautogui.size()

# Calculate the scaling factor between webcam and screen coordinates
webcam_width, webcam_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
x_scale = (screen_width / webcam_width)*1.7
y_scale = (screen_height / webcam_height)*1.7

landmarks_list = []

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    adjacency_matrix = np.zeros((21, 21))
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            for landmark in handlandmark.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmarks_list.append((x, y))

            # adjacency_matrix = d.create_adjacency_matrix(landmarks_list)


            # Length between thumb and little finger
            length = d.trace_shortest_path(adjacency_matrix,8,2)
            # Length between thumb and index finger
            length2 = d.trace_shortest_path(adjacency_matrix,12,2)

            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
            if lmList != []:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb
                x5, y5 = lmList[8][1], lmList[8][2]  # Index finger
                x17, y17 = lmList[20][1], lmList[20][2]  # Little finger

            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x5, y5), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x17, y17), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x5, y5), (255, 0, 0), 3)  # Line from thumb to index finger
            cv2.line(img, (x1, y1), (x17, y17), (255, 0, 0), 3)  # Line from thumb to little finger
            length = hypot(x5 - x1, y5 - y1)
            length2 = hypot(x17 - x1, y17 - y1)
            horizontal_distance = abs(x5 - x1)
            vol = max(min(np.interp(length, [15, 220], [volMin, volMax]), 0.0), -65.25)
            volume.SetMasterVolumeLevel(vol, None)
            is_hand_fully_open = False
            if length > 80 and length2 > 80:
                is_hand_fully_open = True

            # Mimic hand movement with cursor movement
            if is_hand_fully_open:
                # Get the current position of the hand's index finger tip
                index_finger_tip_x = lmList[12][1]
                index_finger_tip_y = lmList[12][2]
                index_finger_tip_x = int(index_finger_tip_x * x_scale)
                index_finger_tip_y = int(index_finger_tip_y * y_scale)
                index_finger_tip_x = index_finger_tip_x
                # Move the cursor to the position of the index finger tip
                pyautogui.moveTo(screen_width - index_finger_tip_x, index_finger_tip_y)
            if length < 80 and length2 < 80:
                is_hand_fully_open = False
            if length < 20:
                if not left_click:
                    left_click = True
                    pyautogui.click(button='left')
            else:
                if left_click:
                    left_click = False

            if length2 < 20:
                if not right_click:
                    right_click = True
                    pyautogui.click(button='right')
            else:
                if right_click:
                    right_click = False

            # if horizontal_distance > horizontal_swipe_threshold:
            #     # Hold Alt key
            #     if not alt_key_pressed:
            #         pyautogui.keyDown('alt')
            #         alt_key_pressed = True
            #
            #     # Simulate pressing and releasing Tab key
            #     pyautogui.press('tab')
            #     time.sleep(0.2)  # Adjust the delay as needed
            #     pyautogui.keyUp('tab')
            #
            # if alt_key_pressed and horizontal_distance <= horizontal_swipe_threshold:
            #     # Release Alt key
            #     pyautogui.keyUp('alt')
            #     alt_key_pressed = False

            print("Volume:", vol, "Horizontal Distance:", horizontal_distance, "Left Click:", left_click, "Right Click:", right_click)


    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
