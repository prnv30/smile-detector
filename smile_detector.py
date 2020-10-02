import cv2


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#pull the webcam feed
webcam = cv2.VideoCapture(0)

#display Frames
while True:

    #read current frame from webcam and store it in 'frame'
    (bool_frame_read, frame) = webcam.read()

    #if theres error reading frame, break
    if not bool_frame_read:
        break

    #change to grayscale(for optimization)
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in grayscale frame
    faces = face_detector.detectMultiScale(frame_grayscale)

    #run face detection within each faces
    for (x, y, w, h) in faces:

        #draw rectangle around the face(in the original frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 200, 50), 3)

        #get the sub frame of the face
        the_face = frame[y:y+h, x:x+w]

        #change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #detect smiles in the grayscale frame
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        #check if smile(s) detected
        if len(smiles)>0:
            cv2.putText(frame, 'Smiling', (x,y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0,0), 1, cv2.LINE_AA)

    #show current frame
    cv2.imshow('Smile Detector', frame)

    #display(impersonates a key press every 1ms)
    cv2.waitKey(1)

#cleanup
webcam.release()
cv2.destroyAllWindows()
