import cv2
from random import randrange

# trained data for recognising the faces
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video for face detection
video = cv2.VideoCapture(0)

# iterate frames in the video
while True:

    # capture frame
    isSuccessful, frame = video.read()

    grayScaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceCoordinates = trainedFaceData.detectMultiScale(grayScaledFrame)

    for i in range(len(faceCoordinates)):
        (x0, y0, w, h) = faceCoordinates[i]
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 4)
        cv2.rectangle(grayScaledFrame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 4)

    cv2.imshow("Face Detector", frame)
    # cv2.imshow("Face Detector", grayScaledFrame)

    key = cv2.waitKey(1)

    if key == 27:
        break

video.release()
