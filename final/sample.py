from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np

# Parameters
width, height = 1280, 720
gestureThreshold = 300
folderPath = "presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image
ink_color = (0, 0, 255)  # Initial ink color (red)

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Zoom variables
zoom_level = 1.0  # Initial zoom level
min_zoom = 0.5  # Minimum zoom level
max_zoom = 2.0  # Maximum zoom level

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, ink_color, cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, ink_color, cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        # Color selection gesture
        if fingers == [1, 1, 0, 0, 0]:
            ink_color = (0, 0, 255)  # Red color
        elif fingers == [1, 1, 1, 0, 0]:
            ink_color = (0, 255, 0)  # Green color
        elif fingers == [1, 1, 1, 1, 0]:
            ink_color = (255, 0, 0)  # Blue color

        # Zoom gestures
        if fingers:
            if fingers == [1, 0, 1, 0, 0]:  # Two fingers up
                zoom_level += 0.1  # Increase zoom level
                if zoom_level > max_zoom:
                    zoom_level = max_zoom

            if fingers == [1, 0, 0, 0, 0]:  # Thumb up
                zoom_level -= 0.1  # Decrease zoom level
                if zoom_level < min_zoom:
                    zoom_level = min_zoom

    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], ink_color, 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    # Resize the image based on the zoom level
    imgCurrentZoomed = cv2.resize(imgCurrent, None, fx=zoom_level, fy=zoom_level)

    # Display the zoomed image
    cv2.imshow("Slides", imgCurrentZoomed)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
