# pip install opencv-python, mediapipe, pyautogui, pydirectinput, keyboard
from cv2 import waitKey, COLOR_RGB2BGR, INTER_AREA, resize, cvtColor, VideoCapture, imshow, circle
from mediapipe import solutions
import pyautogui, pydirectinput, keyboard
from math import sqrt

# Reduce delay / prevent exiting under certain conditions.
pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Set up model and video capture.
webcam = VideoCapture(0)
mp_hands = solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=0)

# Config
click_threshold = 15 # pixels | 30
frames_clicked_threshold = 5 # how many frames should a click be done over to be registered | 3
smoothing = 10 # Number of frames to average hand movement. Reduces jerkiness, increases latency | 10
drag_threshold = 10 # number of pixels of mouse movement required to start dragging | 10
sensitivity = 10 # directly scale movement values
sensitivity_exponent = 1.2 # polynomially scale values
scroll_threshold = 50 # pixels of movement required to scroll
show_video_feed = True
frames_to_skip = 0 # only performs CV action every nth frame allowing for less lag, reduces responsiveness
# ^ set this higher on slower computers, if powerful computer than 0 is fine

# Tracking
left_click_frame_count = 0
right_click_frame_count = 0
middle_click_frame_count = 0
scroll_click_frame_count = 0
left_last_x = None
left_last_y = None
right_last_x = None
right_last_y = None
left_x_movement = []
left_y_movement = []
right_x_movement = []
right_y_movement = []
scroll_hand_present = False

# Optimisation
frame_counter = 0
first_frame = True
hands_in_frame = None
resolution_scale = 4
resolution = (resolution_scale * 160, resolution_scale * 90)

# Constants
image_width = resolution[0]
image_height = resolution[1]
sensitivity /= (resolution_scale * 0.5)
click_threshold *= resolution_scale * 0.5

# State machine
left_locked = False
right_locked = False

def distance(x1, y1, x2, y2):
    x_dist = x2 - x1
    y_dist = y2 - y1
    return sqrt(x_dist**2 + y_dist**2)

while not keyboard.is_pressed("F9"):
    # Optimisation
    frame_counter += 1
    do_cv_this_frame = False
    if frame_counter > frames_to_skip:
        frame_counter = 0
        do_cv_this_frame = True

    # Priority for hands.
    left_hand_found = False
    right_hand_found = False

    # Take a picture from webcam
    if do_cv_this_frame or first_frame:
        if first_frame: first_frame = False
        success, image = webcam.read()
        image = resize(cvtColor(image, COLOR_RGB2BGR), resolution, interpolation = INTER_AREA)
        # image = resize(image, resolution, interpolation = INTER_AREA)
        hands_in_frame = hands.process(image)

    # If there are hand in the frame
    if hands_in_frame.multi_hand_landmarks:
        for index, hand in enumerate(hands_in_frame.multi_hand_landmarks):
            if hands_in_frame.multi_handedness[index].classification[0].label.lower() == "left" and not left_hand_found and not scroll_hand_present:
                left_hand_found = True
                # https://www.researchgate.net/publication/355402809/figure/fig1/AS:1080622231617545@1634651825721/Mediapipe-hand-landmarks.png
                index_finger_x = hand.landmark[8].x * image_width
                index_finger_y = hand.landmark[8].y * image_height
                
                pinky_finger_x = hand.landmark[20].x * image_width
                pinky_finger_y = hand.landmark[20].y * image_height

                middle_finger_x = hand.landmark[12].x * image_width
                middle_finger_y = hand.landmark[12].y * image_height

                thumb_x = hand.landmark[4].x * image_width
                thumb_y = hand.landmark[4].y * image_height

                position_x = hand.landmark[17].x * image_width
                position_y = hand.landmark[17].y * image_height

                index_thumb_distance = distance(index_finger_x, index_finger_y, thumb_x, thumb_y)
                pinky_thumb_distance = distance(pinky_finger_x, pinky_finger_y, thumb_x, thumb_y)
                middle_thumb_distance = distance(middle_finger_x, middle_finger_y, thumb_x, thumb_y)

                if show_video_feed:
                    circle(image, (int(index_finger_x), int(index_finger_y)), 10, (int(index_thumb_distance * 5), 0, int(255 - index_thumb_distance * 5)), 2)
                    circle(image, (int(pinky_finger_x), int(pinky_finger_y)), 10, (int(pinky_thumb_distance * 5), 0, int(255 - pinky_thumb_distance * 5)), 2)
                    circle(image, (int(middle_finger_x), int(middle_finger_y)), 10, (int(middle_thumb_distance * 5), 0, int(255 - middle_thumb_distance * 5)), 2)
                    circle(image, (int(thumb_x), int(thumb_y)), 10, (0, 255, 0), 2)

                if index_thumb_distance < click_threshold:
                    left_click_frame_count += 1
                    if left_click_frame_count == 1:
                        left_locked = True
                    if left_click_frame_count == frames_clicked_threshold:
                        pydirectinput.mouseDown(button="left")
                        print("Left click.")
                elif left_click_frame_count > 0:
                    print("Left release")
                    left_click_frame_count = 0
                    left_locked = False
                    pydirectinput.mouseUp(button="left")

                if pinky_thumb_distance < click_threshold:
                    right_click_frame_count += 1
                    if right_click_frame_count == 1:
                        left_locked = True
                    if right_click_frame_count == frames_clicked_threshold:
                        pydirectinput.mouseDown(button="right")
                elif right_click_frame_count > 0:
                    right_click_frame_count = 0
                    left_locked = False
                    pydirectinput.mouseUp(button="right")

                if middle_thumb_distance < click_threshold:
                    middle_click_frame_count += 1
                elif middle_click_frame_count > 0:
                    middle_click_frame_count = 0

                if left_last_x == None: left_last_x = position_x
                if left_last_y == None: left_last_y = position_y

                movement_x = ((position_x - left_last_x) * sensitivity * -1)
                if movement_x != 0: movement_x = abs((movement_x**sensitivity_exponent).real) * abs(movement_x) / movement_x
                movement_y = ((position_y - left_last_y) * sensitivity)
                if movement_y != 0: movement_y = abs((movement_y**sensitivity_exponent).real) * abs(movement_y) / movement_y
                if middle_click_frame_count > frames_clicked_threshold:
                    movement_x *= 20
                    movement_y *= 3

                left_x_movement.append(movement_x)
                left_y_movement.append(movement_y)

                if len(left_x_movement) > smoothing:
                    left_x_movement.pop(0)
                    left_y_movement.pop(0)

                smoothed_x_movement = int((sum(left_x_movement) / smoothing))
                smoothed_y_movement = int((sum(left_y_movement) / smoothing))

                if distance(smoothed_x_movement, 0, smoothed_y_movement, 0) > drag_threshold:
                    left_locked = False

                if not left_locked:
                    pydirectinput.move(smoothed_x_movement, smoothed_y_movement, _pause=False)

                left_last_x = position_x
                left_last_y = position_y

            elif hands_in_frame.multi_handedness[index].classification[0].label.lower() == "right" and not right_hand_found:
                scroll_hand_present = True
                right_hand_found = True
                # https://www.researchgate.net/publication/355402809/figure/fig1/AS:1080622231617545@1634651825721/Mediapipe-hand-landmarks.png
                index_finger_x = hand.landmark[8].x * image_width
                index_finger_y = hand.landmark[8].y * image_height

                thumb_x = hand.landmark[4].x * image_width
                thumb_y = hand.landmark[4].y * image_height

                position_x = hand.landmark[17].x * image_width
                position_y = hand.landmark[17].y * image_height

                index_thumb_distance = distance(index_finger_x, index_finger_y, thumb_x, thumb_y)

                if index_thumb_distance < click_threshold:
                    scroll_click_frame_count += 1
                    if scroll_click_frame_count == 1:
                        right_locked = True
                elif scroll_click_frame_count > 0:
                    scroll_click_frame_count = 0
                    right_locked = False

                if right_last_x == None: right_last_x = position_x
                if right_last_y == None: right_last_y = position_y

                movement_x = ((position_x - right_last_x) * sensitivity * -1)
                if movement_x != 0: movement_x = abs((movement_x**sensitivity_exponent).real) * abs(movement_x) / movement_x
                movement_y = ((position_y - right_last_y) * sensitivity)
                if movement_y != 0: movement_y = abs((movement_y**sensitivity_exponent).real) * abs(movement_y) / movement_y

                right_x_movement.append(movement_x)
                right_y_movement.append(movement_y)

                if len(right_x_movement) > smoothing:
                    right_x_movement.pop(0)
                    right_y_movement.pop(0)

                smoothed_x_movement = int((sum(right_x_movement) / smoothing))
                smoothed_y_movement = int((sum(right_y_movement) / smoothing))

                if distance(smoothed_x_movement, 0, smoothed_y_movement, 0) > drag_threshold:
                    right_locked = False

                if not left_locked and not right_locked and abs(smoothed_y_movement) > scroll_threshold:
                    if scroll_click_frame_count >= frames_clicked_threshold:
                        pyautogui.scroll(smoothed_y_movement)

                right_last_x = position_x
                right_last_y = position_y
            
            if not right_hand_found: scroll_hand_present = False
    if show_video_feed:
        imshow("Video Feed", image)
    waitKey(1)

pydirectinput.mouseUp(button="left")
pydirectinput.mouseUp(button="right")
