# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:27:28 2020

@author: ja2
"""


import numpy as np
import sys, getopt
import cv2
import os



from keras.models import load_model
from keras.preprocessing import image
import pandas as pd


MODEL_ROOT_PATH="./nets/"



model_age = load_model(MODEL_ROOT_PATH+'age.hdf5')
model_gender = load_model(MODEL_ROOT_PATH+'gender.hdf5')	

shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
 
destdir = 'c:/facecomp1/'
destdir = 'c:/faces2/'
files = [ f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f)) ] 

all_images=[]
predicted_classes=[]
k=0

os.chdir(destdir)
for file in files:
        k+=1
        print(k)
        face_image=file
        frame = cv2.imread(face_image)
        img=frame
        img = img[...,::-1]
        inputs = img.copy() / 255.0
        img_keras = cv2.resize(inputs, (shape[1],shape[2]))    

        img_keras = np.expand_dims(img_keras, axis=0)
        pred_age_keras = model_age.predict(img_keras)[0]
        prob_age_keras = np.max(pred_age_keras)
        age_keras = pred_age_keras.argmax() 

        pred_gender_keras = model_gender.predict(img_keras)[0]
        prob_gender_keras = np.max(pred_gender_keras)
        cls_gender_keras = pred_gender_keras.argmax()        
        tt=[age_keras,cls_gender_keras,pred_gender_keras ]        
        if k==1:
            arrayf=tt        
        else:
            arrayf=np.vstack([ arrayf,tt])
       
df = pd.DataFrame(arrayf)
names=pd.DataFrame(files)
named=pd.concat([names, df],axis=1,ignore_index=True)
named.to_csv('c:/objects/age-genderuk.csv', encoding='utf-8', index=False)
