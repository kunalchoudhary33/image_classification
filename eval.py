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
import random
from PIL import ImageFile
import neptune
ImageFile.LOAD_TRUNCATED_IMAGES = True
import itertools
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score

# Set Neptune Parameters 
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
          'train_path':'/home/paperspace/Kunal/CNN/RR_CNN/ds-dvc-data/Sample_Data/Training/',
          'test_path':'/home/paperspace/Kunal/CNN/RR_CNN/ds-dvc-data/Sample_Data/Training/',
          'model_path':'/home/paperspace/Kunal/CNN/RR_CNN/model/nov_16_baseline_repair_replace_model.hdf5',
          'weight_path':'/home/paperspace/Kunal/CNN/RR_CNN/model/saved_models/nov_16_baseline_repair_replace_weight.hdf5',
          'log_file':'/home/paperspace/Kunal/CNN/RR_CNN/model/saved_models/nov_16_baseline_log_file.csv'}



model = load_model(PARAMS['model_path'])
csv_log = pd.read_csv(PARAMS['log_file'])

# summarize history for accuracy
plt.plot(csv_log['acc'])
plt.plot(csv_log['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('Model_Accuracy_graph.jpg')
plt.show()

# summarize history for loss
plt.plot(csv_log['loss'])
plt.plot(csv_log['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('Model_loss_graph.jpg')
plt.show() 


def load_images(path):
    img = plt.imread(path)
    resized = cv2.resize(img, (PARAMS['img_width'], PARAMS['img_height']), cv2.INTER_LINEAR)
    return resized

def load_train(train_path):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Loading training images...')
    folders = PARAMS['classes']
    for fld in folders:
        index = folders.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*')
        print(path)
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = load_images(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Training data load time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def normalize_train_data(train_path):
    train_data, train_target, train_id = load_train(train_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, PARAMS['num_classes'])

    print('Shape of training data:', train_data.shape)
    return train_data, train_target, train_id

test_data, test_target, test_id = normalize_train_data(PARAMS['test_path'])

data = []
for i,j in zip(test_data,test_target):
    prediction = model.predict([np.expand_dims(i, axis=0)])
    if(np.argmax(prediction) == 0):
        pred = 0
    if(np.argmax(prediction) == 1):
        pred = 1
    if(np.argmax(prediction) == 2):
        pred = 2
    if(np.argmax(j) == 0):
        act = 0
    if(np.argmax(j) == 1):
        act = 1
    if(np.argmax(j) == 2):
        act = 2
    data.append([i, act, pred])
    print('Actual: ',act,' Prediction: ',pred)
    
RR_prediction = pd.DataFrame(data, columns=['claim_number','Actual', 'Prediction'])


# Plot confusion matrix
accuracy_score(RR_prediction['Actual'].tolist(), RR_prediction['Prediction'].tolist())
classes = ["Ok", "Repair", "Replace"]
cm = confusion_matrix(RR_prediction['Actual'].tolist(), RR_prediction['Prediction'].tolist(), labels=[0, 1])
fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
plt.show()
plt.savefig('plot_confusion_matrix.jpg')