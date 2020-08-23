import cv2
from random import randrange

# trained data for recognising the faces in a pic
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#  input
initialImage = cv2.imread('wonderwoman.jpeg')

# Image details
# print(initialImage)
# print(len(initialImage))
# print(len(initialImage[0]))


# Convert the image to grey scale
greyScaledImage = cv2.cvtColor(initialImage, cv2.COLOR_BGR2GRAY)

# finding out the coordinates of identified face rectangle in the pic
faceCoordinates = trainedFaceData.detectMultiScale(greyScaledImage)
print(faceCoordinates)

for i in range(len(faceCoordinates)):
    (x0, y0, w, h) = faceCoordinates[i]
    cv2.rectangle(initialImage, (x0, y0), (x0 + w, y0 + h), (randrange(256), randrange(256), randrange(256)), 4)


# Display image
cv2.imshow("FACE DETECTOR", initialImage)

# To avoid immediate window close
cv2.waitKey()

print("Job Done!!")
