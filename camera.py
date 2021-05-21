import cv2
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

# Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")

# Recognize face
cascPath = 'camera/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Define network and classes
network = torch.load('networks/JNet18_nepoch50_lr0.001_batchsize25_loaderbalanced.pth',
                     map_location=torch.device('cpu'))
classes = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Set parameters for text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
lineType = 2

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop image
            input = frame_gray[y:y + h, x:x + w] / 255

            # Change size for correct input size
            input = cv2.resize(input, (48, 48), interpolation=cv2.INTER_CUBIC)

            # Expand dimension for tensor
            input = np.expand_dims(input, axis=(0, 1))
            input = torch.from_numpy(input.astype('float32'))

            # Calculate prediction
            output = network(input)
            pred = classes[int(output.data.max(1, keepdim=False)[1])]

            # Add text to image
            cv2.putText(frame,pred,
                        (x,int(y-10)),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    # Display the resulting frame
    cv2.imshow('preview', frame)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
