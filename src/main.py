# Written by Nitish Kanna
# 30/06/2024, India

import cv2
import mediapipe as mp
import gesture

import traceback
import time

width, height = 640, 480
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print('Running OpenCV', cv2.__version__)
print('Running Mediapipe', mp.__version__)

# gesture properties in a symbolic representation
class Properties:
    RIGHT_HAND = 0
    LEFT_HAND = 1
    HAND_DOWN = 2
    HAND_UP = 3
    PALM_FRONT = 4
    PALM_BACK = 5
    THUMB_BENT = 6
    THUMB_STRAIGHT = 7

try:
    def model(myHands): 
        modelData = [] # for storing the gesture details
        fingers = gesture.Hand.finger_dict(myHands.hands[0])

        # check if hand is right or left
        if myHands.handTypes[0] == 'Right':
            modelData.append(Properties.RIGHT_HAND)
            xFactor = -1 # reverse the x axis
        else:
            modelData.append(Properties.LEFT_HAND)
            xFactor = 1

        # hand is pointing down if y coordinate of the tip 
        # of the index finger is greater than the y coordinate of
        # the wrist
        if fingers['index'][3][1] > fingers['wrist'][0][1]:
            modelData.append(Properties.HAND_DOWN)
            yFactor = -1 # reverse axis as hand is upside down
            xFactor *= -1
        else:
            modelData.append(Properties.HAND_UP)
            yFactor = 1 # don't reverse axis

        thumbTip = fingers['thumb'][3][0] * xFactor
        pinkyTip = fingers['pinky'][3][0] * xFactor

        if thumbTip < pinkyTip: # palm is facing the camera
            modelData.append(Properties.PALM_FRONT)
            xFactor *= -1
        else: # palm is not facing the camera
            modelData.append(Properties.PALM_BACK)

        # classify fingers as bent or straight
        indices = tuple(fingers.values())

        thumbTip = indices[0][3][0] * xFactor
        thumbKnuckle = indices[0][1][0] * xFactor

        if thumbTip < thumbKnuckle: # special condition for checking if thumb is bent
            modelData.append(Properties.THUMB_BENT)
        else:
            modelData.append(Properties.THUMB_STRAIGHT)
        
        # finger poses will have four numbers representing fingers index to pinky.
        # If a finger is bent then push 1 to fingerPoses or else push 0 to fingerPoses.
        fingerPoses = []
        for i in range(1, 5): # iterate only 4 times to ignore the wrist
            tip = indices[i][3][1] * yFactor
            knuckle = indices[i][1][1] * yFactor

            if tip > knuckle: 
                fingerPoses.append(1) # finger is bent
            else:
                fingerPoses.append(0) # finger is stretched out
        modelData.append(fingerPoses)
        return modelData

    hands = mp.solutions.hands.Hands(False, 1, 1, .5, .5)
    time.sleep(2) # wait for mediapipe log to be initialized

    gestures = dict() # store gesture model data and name

    n = int(input('Number of gestures$ '))
    i = 1

    while i <= n:
        gname = input(f'{i}. Gesture name$ ')
        if gname in gestures.keys():
            print('main: gesture name already exists!')
            continue

        print('Press q when you have your gesture in proper place')

        while cv2.waitKey(1) != ord('q'):
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1) # mirror the frame

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frameRGB)

            modelData = None
            if results.multi_hand_landmarks != None:
                myHands = gesture.Hand(results, width, height)
                myHands.markup(frame)
                modelData = model(myHands) # returns a model containing the gesture's data

            cv2.imshow('Capture', frame)
            cv2.moveWindow('Capture', 0, 0)
        
        if modelData == None:
            print('main: could not find any gestures.')
            continue
        elif modelData in gestures.values():
            print('main: gesture already exists')
            continue
        
        gestures[gname] = modelData
        i += 1
            

    print('Training complete')
    ready = input('Ready for recognition? ')

    while cv2.waitKey(1) != ord('q'):
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        
        modelFound = None
        if results.multi_hand_landmarks != None:
            myHands = gesture.Hand(results, width, height)
            myHands.markup(frame)
            modelFound = model(myHands) # returns a model containing the gesture's data

        if modelFound != None:
            gestFound = 'Unknown'
            for gestName, modelData in gestures.items():
                if modelData == modelFound:
                    gestFound = gestName
                    break
            
            cv2.putText(frame, gestFound, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('Capture', frame)
        cv2.moveWindow('Capture', 0, 0)
    cam.release()
except Exception:
    print(traceback.format_exc())