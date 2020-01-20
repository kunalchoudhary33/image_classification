from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import urllib
import shutil
import os
import glob
from sklearn.metrics import confusion_matrix
#import data_science_tool as ds_tool
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model, Model
from keras import optimizers
from keras.applications.nasnet import NASNetLarge
from keras import backend as K 
import keras
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random
from PIL import ImageFile
from keras.callbacks import Callback
#from data_science_tool import Preprocess as utils
import neptune
from keras.callbacks import Callback
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set Parameters 
PARAMS = {'Architecture':'VGG16 Repair Replace Model with 1024 dense layer size on modified data',
          'description': 'This is vgg16 classification model for repair/replace decision on bumper specific images with modified data',
          'img_width':224,
          'img_height':224,
          'num_channels':3,
          'classes': ["Ok", "Repair", "Replace"],
          'num_classes':3,
          'lr': 0.0001,
          'batch_size':32,
          'class_mode':'categorical',
          'optimizer': 'sgd',
          'loss': 'categorical_crossentropy',
          'metrics': 'accuracy',
          'dense_layer_neurons' : 1024,
          'optimizer':'custom optimizer',
          'kernel_size':(3, 3),
          'pool_size':(2,2),
          'n_epochs': 100,
          'train_count':17743,
          'val_count':3343
          'train_path':'/home/paperspace/Kunal/CNN/RR_CNN/ds-dvc-data/Training/',
          'test_path':'/home/paperspace/Kunal/CNN/RR_CNN/ds-dvc-data/Validation/',
          'model_path':'/home/paperspace/Kunal/CNN/RR_CNN/model/nov_16_baseline_repair_replace_model.hdf5',
          'weight_path':'/home/paperspace/Kunal/CNN/RR_CNN/model/saved_models/nov_16_baseline_repair_replace_weight.hdf5',
          'log_file':'/home/paperspace/Kunal/CNN/RR_CNN/model/saved_models/nov_16_baseline_log_file.csv',
          'api_token':'*****Get api token*****'}

#Data Generator
train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
)

test_datagen = ImageDataGenerator(
        rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
        PARAMS['train_path'],  # this is the target directory
        target_size=(PARAMS['img_width'], PARAMS['img_height']),  
        batch_size=PARAMS['batch_size'],
        class_mode = PARAMS['class_mode'])  # since we use binary_crossentropy loss, we need binary labels

test_generator = test_datagen.flow_from_directory(
        PARAMS['test_path'],  # this is the target directory
        target_size=(PARAMS['img_width'], PARAMS['img_height']), 
        batch_size=PARAMS['batch_size'],
        class_mode = PARAMS['class_mode'])  # since we use binary_crossentropy loss, we need binary labels


#Model Training
def create_VGG16_model():
    base_model = VGG16(weights = 'imagenet',include_top = False, input_shape = (PARAMS['img_width'], PARAMS['img_height'],PARAMS['num_channels']))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense((PARAMS['num_classes']), activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)
    for layers in model.layers[:14]:
        layers.trainable = False

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss = 'categorical_crossentropy',
                 optimizer = sgd,
                 metrics = ['accuracy']
                 )
    model.summary()
    return model
model = create_VGG16_model()


# Neptune Monitor
class NeptuneMonitor(Callback):
    def __init__(self, neptune_experiment, n_batch):
        super().__init__()
        self.exp = neptune_experiment
        self.n = n_batch
        self.current_epoch = 0

    def on_batch_end(self, batch, logs=None):
        x = (self.current_epoch * self.n) + batch
        self.exp.send_metric(channel_name='batch end accuracy', x=x, y=logs['accuracy'])
        self.exp.send_metric(channel_name='batch end loss', x=x, y=logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        self.exp.send_metric('epoch end train accuracy', logs['accuracy'])
        self.exp.send_metric('epoch end val accuracy', logs['val_accuracy'])
        self.exp.send_metric('epoch end train loss', logs['loss'])
        self.exp.send_metric('epoch end val loss', logs['val_loss'])

        msg_acc = 'End of epoch {}, accuracy is {:.4f}'.format(epoch, logs['accuracy'])
        self.exp.send_text(channel_name='accuracy information', x=epoch, y=msg_acc)

        msg_loss = 'End of epoch {}, categorical crossentropy loss is {:.4f}'.format(epoch, logs['loss'])
        self.exp.send_text(channel_name='loss information', x=epoch, y=msg_loss)

        self.current_epoch += 1


neptune.init(api_token=PARAMS['api_token'],
             project_qualified_name='kunalcgi/sandbox')

# retrieve project
project = neptune.Session(PARAMS['api_token'])\
    .get_project('kunalcgi/sandbox')


# CallBAck
with project.create_experiment(name='repair-replace-classification-exp-v2',
                           params=PARAMS,
                           description=PARAMS['description'],
                           upload_source_files=[]) as npt_exp:
    np_callback = NeptuneMonitor(npt_exp,999999999999)
model_checkpointer = ModelCheckpoint(PARAMS['model_path'], \
        verbose=1, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False)
weight_checkpointer = ModelCheckpoint(PARAMS['weight_path'], \
        verbose=1, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.2,min_lr=0.0000032, cooldown=20)
earlystopper = EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=1, mode='min')
callbacks_list = [CSVLogger(PARAMS['log_file']), model_checkpointer, weight_checkpointer, np_callback, learning_rate_reduction, earlystopper]

# Balance dataset
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_generator.classes),train_generator.classes)

print ('---------------------Model training started--------------')
model.fit_generator(
        train_generator,
        steps_per_epoch=PARAMS['train_count'] // PARAMS['batch_size'],
        epochs=PARAMS['n_epochs'],
        validation_data=test_generator,
        validation_steps=PARAMS['val_count'] // PARAMS['batch_size'],
        callbacks=callbacks_list,
        class_weight=class_weights)

print ('---------------------Model training ended--------------')
