import cv2

from random import randrange

#Dowlanding pre-trained data from opencv
trained_face_Data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#To capture video from webcam
webcam = cv2.VideoCapture(0)

#For capturing videos
# webcam = cv2.VideoCapture(video.mp4)

 ##Iterate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Color changing
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect faces
    face_cordinates = trained_face_Data.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the faces
    for (x, y, w,h) in face_cordinates:
           cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    cv2.imshow('82',frame)
    key = cv2.waitKey(1)
    #Stop if q key is pressed
    if key==81 or key==113:
        break


