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
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
import cv2
import seaborn as sns

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
test_impath='../Test/'
pat = 'models_dir/'+'cnnweights.h5'
m=modelcnnnew.cnn_oc()
m.load_weights(pat) 
x_test,y_test=Imagearr(test_impath)
X_test=np.array(x_test)/255.0
y_pred=m.predict_classes(X_test)
#y_pred=m.predict(X_test)
'''l=list(y_pred)
sc=accuracy_score(y_test, y_pred)
sc=np.around(sc, decimals=3)
print('classification accuracy', sc)'''
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
TP, FP, FN, TN = confusion_matrix.ravel()

specificity = TN / (TN + FP) # True Negative Rate

plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = plt.cm.coolwarm);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0:.3f}'.format(metrics.accuracy_score(y_test, y_pred))
plt.title(all_sample_title, size = 15);
fig1=plt.gcf()
fig1.savefig('models_dir/'+'cnn_confusion_matrix.png')
print('Accuracy: {0:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))





