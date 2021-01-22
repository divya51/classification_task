
import os
'''os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))'''
import modelcnnnew
import numpy as np
import keras
import os
import cv2
import glob
import itertools
from keras import backend as K
from  keras import losses
from keras.preprocessing.image import ImageDataGenerator

import pickle 
train_images_path = '../Train/'
val_images_path= '../Val/'
weights_path   = '../weights.h5'
train_batch_size =32
val_batch_size=16


def load_images_from_folder(folder):
	images = [];
	for filename in os.listdir(folder):
		if filename.endswith(".png"):
			img = cv2.imread(os.path.join(folder, filename))
			img = cv2.resize(img, ( 224 , 224 ))
			if img is not None:
				images.append(img)
	return images


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


m=modelcnnnew.cnn_oc()
m.compile(loss='binary_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint_callback = keras.callbacks.ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

x_train,y_train=Imagearr(train_images_path)
train_img_num=len(x_train)
print('train_im'+str(len(x_train)))
print('train_seg'+str(len(y_train)))

x_val,y_val=Imagearr(val_images_path)
val_img_num=len(x_val)
print('val_im'+str(len(x_val)))
print('val_seg'+str(len(y_val)))
train_datagen=ImageDataGenerator(rescale=1./255) 
val_datagen=ImageDataGenerator(rescale=1./255)
G  =  train_datagen.flow(np.array(x_train), y_train, batch_size=train_batch_size )
G2  = val_datagen.flow(np.array(x_val), y_val, batch_size=val_batch_size)
history=m.fit_generator( G , steps_per_epoch=train_img_num/train_batch_size,  validation_data=G2 , validation_steps=val_img_num//val_batch_size ,  epochs=70, callbacks=[early_stopping_callback,checkpoint_callback])
checkname='trainHistory'
checkname+='_'+str(len(x_train))
with open(checkname, 'wb') as file_pi:
	pickle.dump(history.history, file_pi)




