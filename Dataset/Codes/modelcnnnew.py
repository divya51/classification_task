from keras.models import *
from keras.layers import *


def cnn_oc():
    model = Sequential()
    model.add(Conv2D(64, (3, 3),activation = 'relu',padding = 'same', name='block1_conv1',input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3),activation = 'relu',padding = 'same', name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block1_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block2_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block3_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block3_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block3_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block4_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block4_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block4_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block5_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block5_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block5_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block6_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block6_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block6_max_pooling1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block7_conv1'))
    model.add(Conv2D(128, (3, 3),activation = 'relu',padding = 'same', name='block7_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='block7_max_pooling1'))                  
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


m=cnn_oc()
m.summary()










