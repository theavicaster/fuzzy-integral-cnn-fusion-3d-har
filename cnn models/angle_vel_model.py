import numpy as np
import os
from glob import glob
import math
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, concatenate
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras import layers, models
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image


features, _ = Image.open('/content/vel_merged/trainingangvel/trainangvel/a1/a1_s1_t1_skeleton.jpg').size

traindirangvel = '/content/vel_merged/trainingangvel/trainangvel'
valdirangvel = '/content/vel_merged/valangvel'


data_format = K.image_data_format()
K.set_image_data_format(data_format)
np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_of_classes = 27


angvelinput = Input(shape=( 65, features, 1))


angvelmodel = Conv2D(32, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal' , activation = 'relu')(angvelinput)
angvelmodel = Conv2D(32, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)

angvelmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angvelmodel)
angvelmodel = BatchNormalization()(angvelmodel)


angvelmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)
angvelmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)

angvelmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angvelmodel)
angvelmodel = BatchNormalization()(angvelmodel)


angvelmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)
angvelmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)

angvelmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angvelmodel)
angvelmodel = BatchNormalization()(angvelmodel)


angvelmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)
angvelmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angvelmodel)

angvelmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angvelmodel)
angvelmodel = BatchNormalization()(angvelmodel)

angvelmodel = Flatten()(angvelmodel)
angvelmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(angvelmodel)
angvelmodel = BatchNormalization()(angvelmodel)
angvelmodel = Dropout(0.5)(angvelmodel)
angvelmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(angvelmodel)
#temperature
angvelmodel = Lambda(lambda x: x / 2)(angvelmodel)
angvelmodel = Dense(num_of_classes, activation = 'softmax')(angvelmodel)


angvelmodel = Model(inputs = angvelinput, outputs = angvelmodel)



train_datagen_angvel = ImageDataGenerator()
val_datagen_angvel = ImageDataGenerator()

training_set_angvel= train_datagen_angvel.flow_from_directory(
    traindirangvel,
    target_size=(65, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_angvel= val_datagen_angvel.flow_from_directory(
    valdirangvel,
    target_size=(65, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

sgd = optimizers.SGD(lr = 0.05, momentum = 0.9, clipnorm = 1.0)
angvelmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint1 = ModelCheckpoint('modelangvel.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, min_delta = 0.0005,
                              patience=20, min_lr=0.0001, verbose = 1)
callbacks_list = [checkpoint1,reduce_lr]

H = angvelmodel.fit_generator(
    training_set_angvel,
    steps_per_epoch=87, 
    epochs=1000,
    validation_data = val_set_angvel,
    validation_steps = 87,
    callbacks=callbacks_list)


Y_pred = angvelmodel.predict_generator(val_set_angvel)
y_pred = np.argmax(Y_pred, axis=1)

mylabels = []

actions = ['a1', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19'\
           ,'a2', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a3', 'a4', 'a5', 
           'a6', 'a7', 'a8', 'a9']

mapping = {
    'a1': 'Swipe\n Left',
    'a2': 'Swipe\n Right',
    'a3': 'Wave',
    'a4': 'Clap',
    'a5': 'Throw',
    'a6': 'Arm\n Cross',
    'a7': 'Basketball\n Shoot',
    'a8': 'Draw X',
    'a9': 'Draw Circle\n CW',
    'a10': 'Draw Circle\n ACW',
    'a11': 'Draw triangvelle',
    'a12': 'Bowling',
    'a13': 'Boxing',
    'a14': 'Baseball\n Swing',
    'a15': 'Tennis Swing',
    'a16': 'Arm Curl',
    'a17': 'Tennis Serve',
    'a18': 'Push',
    'a19': 'Knock',
    'a20': 'Catch',
    'a21': 'Pickup &\n Throw',
    'a22': 'Jog',
    'a23': 'Walk',
    'a24': 'Sit to\n Stand',
    'a25': 'Stand to\n Sit',
    'a26': 'Lunge',
    'a27': 'Squat'
}


for l in actions:
  mylabels.append( mapping[l])


matrix = confusion_matrix(val_set_angvel.classes, y_pred,labels=None)

fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                cmap = 'viridis',
                                show_absolute=True,
                                show_normed=False,
                                figsize = (12,12),
                                class_names = mylabels
                                )
plt.savefig('modelangvelcm_utdmhad.png')
