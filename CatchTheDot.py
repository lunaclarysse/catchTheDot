
import cv2
import mediapipe as mp
import random
import math


mp_hand = mp.solutions.hands
hands = mp_hand.Hands()

mp_drawing_utils = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# get current image
success, img = cap.read()
# height, width, channels of the image
h,w,c = img.shape

#generate random position of the goal within boundaries (note: position has to be at least 40 pixels from the border)
goal_x = random.randint(40, w-40)
goal_y = random.randint(40, h-40)

#the score keeps track of how many dots are catched
score = 0


while True:
    #create array that keeps track of all the distances from the different hands to the goal
    distances = []

    # get the current image
    success, img = cap.read()
    # mirror the image
    img = cv2.flip(img, 1)

    #checking whether an image was received correctly, otherwise stop
    if not success:
        break

    # display current score
    cv2.putText(img, f"Score: {str(score)}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 255), 3)

    # draw a green dot at the position of the goal
    cv2.circle(img, (goal_x, goal_y), 10, (0,255,0), cv2.FILLED)

    # detect the hands in the image
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # check whether there are hands detected in the image
    # multi_hand_landmarks = Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks and each landmark is composed of x, y and z
    if result.multi_hand_landmarks:
        # loop over all detected hands in the image (result.multi_hand_landmarks) and get the landmarks
        for hand_landmark in result.multi_hand_landmarks:
            #draw the landmarks of the detected hand
            mp_drawing_utils.draw_landmarks(
                                            img,
                                            hand_landmark,
                                            mp_hand.HAND_CONNECTIONS,
                                            )

            # loop over all the 21 hand landmarks
            for id, landmark in enumerate(hand_landmark.landmark):
            # id indicates which of the 21 hand landmarks,
            # landmark contains x,y and z which are normalized to [0,1]


                # id=8 is the top of the index finger
                if id == 8:
                    # height, width, channels of the image
                    h, w, c = img.shape

                    # compute the coordinates of the top of the index finger: center_x and center_y
                    cx, cy = int(landmark.x * w), int(landmark.y * h)

                    # draw dot on top of the index finger
                    cv2.circle(img, (cx, cy), 10, (250, 0, 0), cv2.FILLED)


                    # compute distance between goal and top of the index finger
                    # formula:  sqrt( (x_1-x_2)^2 +(y_1-y_2)^2)
                    distance = math.sqrt((goal_x - cx) ** 2 + (goal_y - cy) ** 2)

                    # add this distance to the list of distances from all different index fingers to the goal
                    distances.append(distance)

    # display image with all of its drawings
    cv2.imshow("Image",img)
    cv2.waitKey(1)

    # checking whether one of the detected hands has reached the goal
    # the goal is reached by an index finger if the distance is smaller than 10
    if len(distances) != 0 and min(distances) < 10:
        # if the goal is reached, generate new position of the goal
        goal_x = random.randint(40, w-40)
        goal_y = random.randint(40, h-40)

        # increase the score with 1
        score = score +1


cap.release()
cv2.destroyAllWindows()