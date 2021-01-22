import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
import modelcnn, LoadBatches
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
#fold_no=sys.argv[1]
test_impath='/home2/data/divya_degala/cnn_oc/prob_plot/Rahul_Deekshith_M/';
pat = "./weights/f23cnn.h5"
m=modelcnn.cnn_oc()
m.load_weights(pat) 
x_test,name_list=LoadBatches.load_images_from_folder(test_impath)
#print(name_list)
name_list=np.array(name_list)
print(np.shape(np.array(x_test)))
print(len(name_list))
#print(np.shape(np.array(y_test)))
X_test=np.array(x_test)/255.0;
#y_pred=m.predict_classes(X_test)
y_pred=m.predict(X_test)
print(y_pred)
l=list(y_pred)
print(l.count(1))
'''sc=accuracy_score(y_test, y_pred)
sc=np.around(sc, decimals=3)
print(sc)'''
plt.plot(name_list,y_pred)
scipy.io.savemat('/tmp/Rahul_Deekshith_M.mat',{'p':y_pred,'fn':name_list});
#plt.xlim(0,len(name_list))
#plt.ylim(0,1)
plt.title('plot of predicted probabilities')
plt.xlabel('image no')
plt.ylabel('Predicted probability of glottis')
plt.show()



