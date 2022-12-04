
import numpy as np
import cv2
import sklearn
import pickle
from django.conf import settings
import os
from keras.models import model_from_json
import mediapipe as mp
from math import *


STATIC_DIR = settings.STATIC_DIR
print("----------------------------------" , STATIC_DIR)

face_detector_model = cv2.dnn.readNetFromCaffe(os.path.join(STATIC_DIR , 'models/deploy.prototxt.txt'),
                                               os.path.join(STATIC_DIR ,'models/res10_300x300_ssd_iter_140000.caffemodel'))
model_file = open(os.path.join(STATIC_DIR , 'models\\ndtry_better_model.json'), "r")
model_json = model_file.read()
model = model_from_json(model_json)
model.load_weights(os.path.join(STATIC_DIR , 'models\\ndtry_better_model_weights.h5'))


def aligner(coordinates , img):
    w,h,_= img.shape
    left_eye_left = coordinates[130]
    right_eye_right = coordinates[359]
    lx = left_eye_left[0]
    ly = left_eye_left[1]
    rx = right_eye_right[0]
    ry = right_eye_right[1]
    nose_tip = coordinates[94]
    scale = 1
    angle = (atan((ry-ly) / (rx-lx))*180)/ pi
    M = cv2.getRotationMatrix2D((nose_tip[0] , nose_tip[1]), angle, scale)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    return rotated_img

def bg_remove(image):
    face_mesh = mp.solutions.face_mesh.FaceMesh()
    
    #image = cv2.resize(image , dsize = (224,224))
    #image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    height,width, _ = image.shape
    
    
    result = face_mesh.process(image)
    all_landmarks = result.multi_face_landmarks[0].landmark
    coordinates = []
    for i in all_landmarks:
        x_and_y = []
        x_and_y.append((int)(i.x*width))
        x_and_y.append((int)(i.y*height))
        coordinates.append(x_and_y)
    
    routes = [
              (coordinates[332][0] ,coordinates[332][1])  ,
              (coordinates[297][0] ,coordinates[297][1])  ,
              (coordinates[338][0] ,coordinates[338][1])  ,
              (coordinates[10][0] ,coordinates[10][1])  ,
              (coordinates[109][0] ,coordinates[109][1])  ,
              (coordinates[67][0] ,coordinates[67][1])  ,
              (coordinates[103][0] ,coordinates[103][1])  ,
              (coordinates[54][0] ,coordinates[54][1])  ,
              (coordinates[21][0] ,coordinates[21][1])  ,
              (coordinates[162][0] ,coordinates[162][1])  ,
              (coordinates[127][0] ,coordinates[127][1])  ,
              (coordinates[234][0] ,coordinates[234][1])  ,
              (coordinates[93][0] ,coordinates[93][1])  ,
              (coordinates[132][0] ,coordinates[132][1])  ,
              (coordinates[58][0] ,coordinates[58][1])  ,
              (coordinates[172][0] ,coordinates[172][1])  ,
              (coordinates[136][0] ,coordinates[136][1])  ,
              (coordinates[150][0] ,coordinates[150][1])  ,
              (coordinates[149][0] ,coordinates[149][1])  , 
              (coordinates[176][0] ,coordinates[176][1]),
              (coordinates[148][0] ,coordinates[148][1]),
              (coordinates[152][0] ,coordinates[152][1]), 
              (coordinates[377][0] ,coordinates[377][1]),
              (coordinates[400][0] ,coordinates[400][1]),
              (coordinates[378][0] ,coordinates[378][1]),
              (coordinates[379][0] ,coordinates[379][1])  ,
              (coordinates[365][0] ,coordinates[365][1])  ,
              (coordinates[397][0] ,coordinates[397][1])  ,
              (coordinates[288][0] ,coordinates[288][1])  ,
              (coordinates[361][0] ,coordinates[361][1])  ,
              (coordinates[323][0] ,coordinates[323][1])  ,
              (coordinates[454][0] ,coordinates[454][1])  ,
              (coordinates[356][0] ,coordinates[356][1])  ,
              (coordinates[389][0] ,coordinates[389][1])  ,
              (coordinates[251][0] ,coordinates[251][1])  ,
              (coordinates[284][0] ,coordinates[284][1])  ,
              (coordinates[332][0] ,coordinates[332][1])  ,
             ]
    cv2.line(image , (coordinates[176][0] ,coordinates[176][1])  ,(coordinates[148][0] ,coordinates[148][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[148][0] ,coordinates[148][1])  ,(coordinates[152][0] ,coordinates[152][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[152][0] ,coordinates[152][1])  ,(coordinates[377][0] ,coordinates[377][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[377][0] ,coordinates[377][1])  ,(coordinates[400][0] ,coordinates[400][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[400][0] ,coordinates[400][1])  ,(coordinates[378][0] ,coordinates[378][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[378][0] ,coordinates[378][1])  ,(coordinates[379][0] ,coordinates[379][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[379][0] ,coordinates[379][1])  ,(coordinates[365][0] ,coordinates[365][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[365][0] ,coordinates[365][1])  ,(coordinates[397][0] ,coordinates[397][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[397][0] ,coordinates[397][1])  ,(coordinates[288][0] ,coordinates[288][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[288][0] ,coordinates[288][1])  ,(coordinates[361][0] ,coordinates[361][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[361][0] ,coordinates[361][1])  ,(coordinates[323][0] ,coordinates[323][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[323][0] ,coordinates[323][1])  ,(coordinates[454][0] ,coordinates[454][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[454][0] ,coordinates[454][1])  ,(coordinates[356][0] ,coordinates[356][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[356][0] ,coordinates[356][1])  ,(coordinates[389][0] ,coordinates[389][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[389][0] ,coordinates[389][1])  ,(coordinates[251][0] ,coordinates[251][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[251][0] ,coordinates[251][1])  ,(coordinates[284][0] ,coordinates[284][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[284][0] ,coordinates[284][1])  ,(coordinates[332][0] ,coordinates[332][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[332][0] ,coordinates[332][1])  ,(coordinates[297][0] ,coordinates[297][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[297][0] ,coordinates[297][1])  ,(coordinates[338][0] ,coordinates[338][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[338][0] ,coordinates[338][1])  ,(coordinates[10][0]  ,coordinates[10][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[10][0]  ,coordinates[10][1])   ,(coordinates[109][0] ,coordinates[109][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[109][0] ,coordinates[109][1])  ,(coordinates[67][0]  ,coordinates[67][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[67][0]  ,coordinates[67][1])   ,(coordinates[103][0] ,coordinates[103][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[103][0] ,coordinates[103][1])  ,(coordinates[54][0]  ,coordinates[54][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[54][0]  ,coordinates[54][1])   ,(coordinates[21][0]  ,coordinates[21][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[21][0]  ,coordinates[21][1])   ,(coordinates[162][0] ,coordinates[162][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[162][0] ,coordinates[162][1])  ,(coordinates[127][0] ,coordinates[127][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[127][0] ,coordinates[127][1])  ,(coordinates[234][0] ,coordinates[234][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[234][0] ,coordinates[234][1])  ,(coordinates[93][0]  ,coordinates[93][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[93][0]  ,coordinates[93][1])   ,(coordinates[132][0] ,coordinates[132][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[132][0] ,coordinates[132][1])  ,(coordinates[58][0]  ,coordinates[58][1])  , (255,255,0), 1)
    cv2.line(image , (coordinates[58][0]  ,coordinates[58][1])   ,(coordinates[172][0] ,coordinates[172][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[172][0] ,coordinates[172][1])  ,(coordinates[136][0] ,coordinates[136][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[136][0] ,coordinates[136][1])  ,(coordinates[150][0] ,coordinates[150][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[150][0] ,coordinates[150][1])  ,(coordinates[149][0] ,coordinates[149][1]) , (255,255,0), 1)
    cv2.line(image , (coordinates[149][0] ,coordinates[149][1])  ,(coordinates[176][0] ,coordinates[176][1]) , (255,255,0), 1)
    
    mask = np.zeros((height,width))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out , coordinates


def get_new_coordinates(image):
    face_mesh = mp.solutions.face_mesh.FaceMesh()
#     image = cv2.resize(image , dsize = (224,224))
    rgb_image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    height,width, _ = image.shape
    result = face_mesh.process(image)
    
    all_landmarks = result.multi_face_landmarks[0].landmark
    coordinates = []
    for i in all_landmarks:
        x_and_y = []
        x_and_y.append((int)(i.x*width))
        x_and_y.append((int)(i.y*height))
        coordinates.append(x_and_y)
    return coordinates

def normalizer(coordinates):
    for i in coordinates:
        i[0] =(int)  (i[0] - coordinates[94][0])
        i[1] = (int) (i[1] - coordinates[94][1])
    return coordinates


def preprocess(image):
    image , coor = bg_remove(image)
#     image = aligner(coor , image)
    real_coor = get_new_coordinates(image)
    real_coor = normalizer(real_coor)
    return image , coor




def predictor(image):
    IMAGE_SHAPE = (224, 224)
    IMAGE_SHAPE+(3,)
    img_resize = cv2.resize(image , IMAGE_SHAPE)
    predicted = model.predict(np.array([img_resize]))
    ind = np.argmax(predicted , axis = 1)
    index = ind[0]

    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++=========================",predicted[0][index])
    return index,predicted[0][index]




def pipeline_model(path):
    # pipeline model
    img = cv2.imread(path)
    image = img.copy()
    h,w = img.shape[:2]
    # face detection
    img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    
    # machcine results
    machinlearning_results = dict(face_detect_score = [], 
                                 face_name = [],
                                 face_name_score = [],
                                 emotion_name = [],
                                 emotion_name_score = [],
                                 count = [])
    count = 1
    if len(detections) > 0:
        for i , confidence in enumerate(detections[0,0,:,2]):
            if confidence > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                startx,starty,endx,endy = box.astype(int)
                score = -1
                score = -1
                cnt = -1
                cnt1 = -1
                 
                    #Detected Face in Cropped Variable................................
                try:
                    cropped,cc = preprocess(image)
                    cnt,score = predictor(cropped)
                except:
                    print()
                    print("Not Croppable .......... Moving Ahead with original image")
                
                cnt1,score1 = predictor(image)
                    
                if score1 > score:
                    cnt=cnt1
                    score=score1
                print()
                print("Child Number               == " , cnt+1)
                print("Face Verification Score    == " , score*100 , " %")
                print()
                break            
    return image , cnt, score





