# Written by Nitish Kanna
# 30/06/2024, India

import cv2
import mediapipe as mp
import gesture

import traceback

width, height = 640, 480
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print('Running OpenCV', cv2.__version__)
print('Running Mediapipe', mp.__version__)

try:
    def train(frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        if results.multi_hand_landmarks != None:
            myHands = gesture.Hand(results, width, height)
            myHands.markup(frame)
            return model(myHands)
        else:
            return None

    def model(myHands): 
        modelDat = dict()
        fingers = gesture.Hand.finger_dict(myHands.hands[0])

        # check if hand is right or left
        if myHands.handTypes[0] == 'Right':
            modelDat['Type'] = 1
            xFactor = -1
        else:
            modelDat['Type'] = 0
            xFactor = 1

        # hand is pointing down if y coordinate of the tip 
        # of the index finger is greater than the y coordinate of
        # the wrist
        if fingers['index'][3][1] > fingers['wrist'][0][1]:
            modelDat['YPos'] = 1
            yFactor = -1 # reverse axis as hand is upside down
            xFactor *= -1
        else:
            modelDat['YPos'] = 0
            yFactor = 1 # don't reverse axis

        thumbTip = fingers['thumb'][3][0] * xFactor
        pinkyTip = fingers['pinky'][3][0] * xFactor

        if thumbTip < pinkyTip: # palm is facing the camera
            modelDat['XPos'] = 1
            xFactor *= -1
        else: # palm is not facing the camera
            modelDat['XPos'] = 0

        # classify fingers as bent or straight
        indices = tuple(fingers.values())
        indicator = ''

        thumbTip = indices[0][3][0] * xFactor
        thumbKnuckle = indices[0][1][0] * xFactor

        if thumbTip < thumbKnuckle: # special condition for checking if thumb is bent
            indicator += '1'
        else:
            indicator += '0'

        for i in range(1, 5): # iterate only 4 times to ignore the wrist
            tip = indices[i][3][1] * yFactor
            knuckle = indices[i][1][1] * yFactor

            if tip > knuckle: 
                indicator += '1' # indicates that the finger is bent
            else:
                indicator += '0'
        modelDat['Fingers'] = indicator
        
        return modelDat

    gestures = dict()
    n = int(input('Number of gestures$ '))
    hands = mp.solutions.hands.Hands(False, 1, 1, .5, .5)
    i = 0

    while i != n:
        i += 1

        gname = input(f'{i}. Gesture name$ ')
        print('Press q when you have your gesture in proper place')

        while cv2.waitKey(1) != ord('q'):
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            model_result = train(frame)
            if model_result != None:
                gestures[gname] = model_result

            cv2.imshow('Capture', frame)
            cv2.moveWindow('Capture', 0, 0)
        
        if gname not in gestures.keys():
            print('main: could not find any gestures.')
            i -= 1
            

    print('Training completed. Now recognizing')
    ready = input('Ready for recognition? ')

    while cv2.waitKey(1) != ord('q'):
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        modelFound = train(frame)
        unknown = True

        if modelFound != None:
            for gestName, modelDat in gestures.items():
                if modelDat == modelFound:
                    unknown = False
                    break
            if unknown == True:
                gestName = 'unknown'
            
            cv2.putText(frame, gestName, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('Capture', frame)
        cv2.moveWindow('Capture', 0, 0)

    cam.release()
except Exception:
    print(traceback.format_exc())

'''
Right - 1
Left - 0

Hand down - 1
Hand up - 0

Palm back - 1
Palm front - 0

Finger bent - 1
Finger straight - 0
'''