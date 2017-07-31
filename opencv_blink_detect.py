import cv2
import sys
import dlib
import numpy as np
import argparse
from imutils import face_utils
import imutils
from scipy.spatial.distance import euclidean as dist

# Run this file as such 'python opencv_blink_detect.py -p sp.dat'

#Defining EAR
##EAR is the ratio between width and height of eye
EYE_AR_THRESH = 0.43
EYE_AR_CONSEC_FRAMES = 2

L_COUNTER = 0  # Counts number of frame left eye has been closed
R_COUNTER = 0  # Counts number of frame right eye has been closed

TOTAL_BLINK_COUNTER = 0
L_BLINK_COUNTER = 0
R_BLINK_COUNTER = 0

#Method to convert rectanngle to Bounding Box
def rect2bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

#Converting Shape to numpy array
def shape2np(shape, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates


def draw_details(rect, shape, image):
    #Rectangle over the face
    (x, y, w, h) = rect2bb(rect)

    #Points of the facial landmarks
    shape = shape2np(shape)

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)

    #Drawing rectangle over the face
    #Drawing a point over each landmark
    for (x,y) in shape:
        cv2.circle(image, (x,y), 2, (0, 0, 255), -1)
    return image


def find_features(image):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shapes.append(shape)
        image = draw_details(rect, shape, image)
    return image, rects, shapes


def calculate_EAR(eye):
    if len(eye)==6:
        width = dist(eye[0],eye[3])
        A = dist(eye[1],eye[5])
        B = dist(eye[2],eye[4])
        EAR = (A+B)/width
        return EAR
    else:
        print "Error in eye shape"
        return -1


def get_eyes(features):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    features = shape2np(features)
    l_eye = features[lStart:lEnd]
    r_eye = features[rStart:rEnd]
    return l_eye, r_eye




#ArgParser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="Path to landmark predictor")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


cam = cv2.VideoCapture(0)

frame_no=0
x=list()
y=list()

while True:
    ret, img = cam.read()
    img, rects, feature_array = find_features(img)
    n_faces = len(rects)
    if n_faces!=0:
        features = feature_array[0]         # Currently only calculating blink for the First face
        l_eye, r_eye = get_eyes(features)
        l_EAR = calculate_EAR(l_eye)
        r_EAR = calculate_EAR(r_eye)
        L_COUNTER += l_EAR <= EYE_AR_THRESH
        R_COUNTER += r_EAR <= EYE_AR_THRESH

        if L_COUNTER == EYE_AR_CONSEC_FRAMES:
            L_COUNTER = 0
            TOTAL_BLINK_COUNTER += 1  # Blink has been Detected in the Left eye
            L_BLINK_COUNTER += 1
        if R_COUNTER == EYE_AR_CONSEC_FRAMES:
            R_COUNTER = 0
            TOTAL_BLINK_COUNTER += 1  # Blink has been Detected in the  Right eye
            R_BLINK_COUNTER += 1

        cv2.putText(img, "Blinks: {} , {} ".format(L_BLINK_COUNTER, R_BLINK_COUNTER), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "EAR: {:.2f} , {:.2f} ".format(l_EAR,r_EAR), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        x.append(frame_no);
        y.append(l_EAR);
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 1048603: #Escape clicked.Exit program
        break
cv2.destroyAllWindows()







