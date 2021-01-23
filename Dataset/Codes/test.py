import os
'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))'''
import modelcnnnew
import numpy as np
import keras
from keras import backend as K
from  keras import losses
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
def Imagearr(root_folder):
	labels=[]
	folders = [os.path.join(root_folder, x) for x in ('Uninfected', 'Parasite')]
	all_images = [img for folder in folders for img in load_images_from_folder(folder)]
	for folder in folders:
		if folder.split('/')[-1]=='Uninfected':
			for img in load_images_from_folder(folder):
				labels.append(1)
		elif folder.split('/')[-1]=='Parasite':
			for img in load_images_from_folder(folder):
				labels.append(0)           
	return all_images,labels
test_impath='../Test/'
pat = '../weights.h5'
m=modelcnnnew.cnn_oc()
m.load_weights(pat) 
x_test,y_test=Imagearr(test_impath)
X_test=np.array(x_test)/255.0;
y_pred=m.predict_classes(X_test)
l=list(y_pred)
sc=accuracy_score(y_test, y_pred)
sc=np.around(sc, decimals=3)
print(sc)




