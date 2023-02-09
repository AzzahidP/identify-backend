import os
from os import listdir
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np

import pickle
import cv2

MyFaceNet = load_model('facenet_keras.h5')

database = {}

folder='runs/detect/exp/crops/ce/' #Folder gambar yang udah dicrop
database = {}

for filename in listdir(folder):
    
    path = folder + filename
    gbr = cv2.imread(folder + filename)

    gbr1 = cv2.cvtColor(gbr, cv2.COLOR_BGR2RGB) # konversi dari OpenCV ke PIL

    gbr1 = Image.fromarray(gbr1)                       
    gbr1 = gbr1.resize((160,160))
    gbr1 = asarray(gbr1)

    gbr1 = gbr1.astype('float32')
    mean, std = gbr1.mean(), gbr1.std()
    gbr1 = (gbr1 - mean) / std

    gbr1 = expand_dims(gbr1, axis=0)
    signature = MyFaceNet.predict(gbr1)

    print(gbr1)

    database[os.path.splitext(filename)[0]]=signature

myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()

print(database)