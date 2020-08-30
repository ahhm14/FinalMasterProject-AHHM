import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time
import pandas as pd


from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Concatenate
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, roc_auc_score, recall_score, roc_curve
from collections import Counter
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from DLR_extraction import concat_DLR, concat_DLR_LM
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import seaborn as sns

path= r'C:\Users\alejandro\Documents\files_master\HYPERTENSION\XL'

# X_ed = np.load(path+'/DL_matrices/X_ed.npy')
# X_es = np.load(path+'/DL_matrices/X_es.npy')
# X_set = np.load(path+'/DL_matrices/Extended/X_set.npy')
# print('pass')
# y_ = np.load(path+'/DL_matrices/Extended/y_classes.npy')

# X_es = np.load(path+'/DL_matrices/ANGINA/X_es.npy')
# #
# y_ = pd.read_csv(path+'/DL_matrices/ANGINA/y_classes(ang)2.csv', index_col=0)
# y_ = np.array(y_)

X_es = np.float32(np.load(path+'\X_es.npy'))
X_ed = np.float32(np.load(path+'\X_ed.npy'))

X = np.concatenate([X_ed, X_es])
print(X.shape)

#
y_ = pd.read_csv(path+'\y_classes.csv', index_col=0)
class_list = y_.iloc[:,0].unique()
y_ = np.array(y_)

y = np.concatenate([y_, y_])
print(y.shape)

print('pass')
encoder = LabelEncoder()


y_t = encoder.fit_transform(y)
print('pass')
y_t2 = to_categorical(y_t, num_classes=2)

print(y_t2.shape)

def metricas(y_pred, y_test_tr):
    cm = confusion_matrix(y_test_tr, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm_normalized)

    plt.figure(figsize=(10, 4))
    plt.title("Confusion Matrix")
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.tight_layout()

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks+0.5, class_list, rotation=45)
    plt.yticks(tick_marks+0.5, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test_tr,y_pred,target_names=class_list))

    test_acc = accuracy_score(y_test_tr, y_pred)
    precison_scor = precision_score(y_test_tr, y_pred)
    recall_scor = recall_score(y_test_tr, y_pred)

    print("\n Test Accuracy with best estimator: ", test_acc)
    print("\n Precision with best estimator: ", precison_scor)
    print("\n Recall with best estimator: ", recall_scor)

def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y_t2, test_size=0.30, random_state=42, shuffle=True, stratify=y_t2)

X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=True, stratify=y_train)

#Data Augmentation Set Up
train_gen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='constant')

train_gen.fit(X_train1)

def three_inp_gen(X1, X2, X3, Y):
    genX1 = train_gen.flow(X1,Y, seed=7)
    genX2 = train_gen.flow(X2, seed=7)
    genX3 = train_gen.flow(X3, seed=7)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            yield [X1i[0], X2i, X3i], X1i[1]

three_inp= three_inp_gen(X_train1[:,:,:,[0]],
                X_train1[:,:,:,[1]],
                X_train1[:,:,:,[2]],
                y_train1)


inp_shape= (150, 150,1)

#Optimzers
sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

#First branch

input1 = Input(inp_shape)
conv1 = layers.Conv2D(32, kernel_size=4, activation='relu')(input1)
conv2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1)
conv3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2)
pool = layers.MaxPooling2D()(conv3)  #Pooling

conv4 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool)
conv5 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3)
pool2 = layers.MaxPooling2D()(conv5)  #Pooling

conv6 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2)
pool3 = layers.MaxPooling2D()(conv6)  #Pooling

# conv7 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3)
flat1 = layers.Flatten()(pool3)

#Second Branch
input2 = Input(inp_shape)
conv1_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(input2)
conv2_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1_2)
conv3_2 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2_2)
pool_2 = layers.MaxPooling2D()(conv3_2)  #Pooling

conv4_2 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool_2)
conv5_2 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3_2)
pool2_2 = layers.MaxPooling2D()(conv5_2)  #Pooling

conv6_2 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2_2)
pool3_2 = layers.MaxPooling2D()(conv6_2)  #Pooling

# conv7_2 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3_2)
flat2= layers.Flatten()(pool3_2)

#Third Branch
input3 = Input(inp_shape)
conv1_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(input3)
conv2_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv1_3)
conv3_3 = layers.Conv2D(32, kernel_size=4, activation='relu')(conv2_3)
pool_3 = layers.MaxPooling2D()(conv3_3)  #Pooling

conv4_3 = layers.Conv2D(16, kernel_size=4, activation='relu')(pool_3)
conv5_3 = layers.Conv2D(16, kernel_size=4, activation='relu')(conv3_3)
pool2_3 = layers.MaxPooling2D()(conv5_3)  #Pooling

conv6_3 = layers.Conv2D(8, kernel_size=4, activation='relu')(pool2_3)
pool3_3 = layers.MaxPooling2D()(conv6_3)  #Pooling

# conv7_3 = layers.Conv2D(4, kernel_size=4, activation='relu')(pool3_3)
flat3 = layers.Flatten()(pool3_3)

# merge feature extractors
merge = Concatenate()([flat1, flat2, flat3])

# interpretation layer
hidden1 = Dense(512, activation='relu')(merge)

# prediction output
hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(2, activation='softmax')(hidden2)

model = Model(inputs=[input1,input2, input3], outputs=output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

NAME = 'Late Merging'

# filepathdest_incep = "/content/gdrive/My Drive/Colab Notebooks/Results/BaseCNN.hdf5"

# callback_setting = [ModelCheckpoint(filepath=filepathdest_incep, verbose=1, monitor='val_loss', mode='min', save_best_only=True)]

# log_dir = "logs/fit/" + NAME +' ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=False)

history = model.fit_generator(three_inp,
                    steps_per_epoch=len(X_train1)/50,
                    epochs=650,
                    verbose=1,
                    validation_data=([X_val[:,:,:,[0]],X_val[:,:,:,[1]],X_val[:,:,:,[2]]],y_val))

model.save(path+'\LM_model.hdf5')


print('----- Testing Model -------')

model.evaluate([X_test[:,:,:,[0]],
                    X_test[:,:,:,[1]],
                    X_test[:,:,:,[2]]],
                    y_test)

Y_pred = model.predict([X_test[:, :, :, [0]],
                        X_test[:, :, :, [1]],
                        X_test[:, :, :, [2]]])
y_pred = np.argmax(Y_pred, axis=1)

y_test_tr = np.argmax(y_test, axis=1)

metricas(y_pred, y_test_tr)
plt.show()

print('ROC_AUC Score', roc_auc_score(y_test.ravel(), Y_pred.ravel()))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.ravel(), Y_pred.ravel())

plot_roc_curve(fpr_keras, tpr_keras)



y_1 = to_categorical(encoder.fit_transform(y_))

print('----- Testing Model X_ED -------')

Y_pred = model.predict([X_ed[:, :, :, [0]],
                        X_ed[:, :, :, [1]],
                        X_es[:, :, :, [2]]])
y_pred = np.argmax(Y_pred, axis=1)

y_test_tr = np.argmax(y_1, axis=1)

metricas(y_pred, y_test_tr)

print('----- Testing Model X_ES -------')

Y_pred = model.predict([X_es[:, :, :, [0]],
                        X_es[:, :, :, [1]],
                        X_es[:, :, :, [2]]])
y_pred = np.argmax(Y_pred, axis=1)

y_test_tr = np.argmax(y_1, axis=1)

metricas(y_pred, y_test_tr)

id_df = pd.read_csv(path+'\ids.csv', index_col=0)
#
LM_features = concat_DLR_LM(model, X_ed, X_es, id_df)

LM_features.to_csv(path+'\iLM_features.csv')


