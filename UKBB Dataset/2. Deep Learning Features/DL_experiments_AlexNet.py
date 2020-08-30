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
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.applications import resnet, inception_resnet_v2, inception_v3, densenet, vgg16, vgg19
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, roc_auc_score, recall_score, roc_curve
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from DLR_extraction import concat_DLR
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

# X_set = np.load(path+'/DL_matrices/Extended/X_set.npy')
# print('pass')
# y_ = np.load(path+'/DL_matrices/Extended/y_classes.npy')

encoder = LabelEncoder()

y_t = encoder.fit_transform(y)
y_t2 = to_categorical(y_t, num_classes=2)

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

inp_shape = X_train1.shape[1:]


#Optimzers
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

###AlexNet Model

model = Sequential()

model.add(Conv2D(96, (11, 11), strides = (4,4), padding='valid', activation='relu', input_shape = inp_shape))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model.add(Conv2D(256, (5, 5), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model.add(Conv2D(384, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
model.add(Conv2D(384, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
model.add(Conv2D(256, (3, 3), strides=(1,1), padding="same", activation='relu', dilation_rate=2))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

# model.add(Dropout(0.20))

model.add(Flatten())

model.add(Dense(9216, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit_generator(train_gen.flow(X_train, y_train, batch_size=20),
                    steps_per_epoch = X_train1.shape[0]/20,
                    epochs=650,
                    verbose=1,
                    validation_data=(X_val, y_val)
                   )

model.save(path+'\Alex_net.hdf5')

print('----- Testing Model -------')

model.evaluate(X_test,y_test)

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis =1)

y_test_tr = np.argmax(y_test, axis = 1)
metricas(y_pred, y_test_tr)
plt.show()

Y_pred2 = model.predict(X_test).ravel()

print('ROC_AUC Score', roc_auc_score(y_test.ravel(), Y_pred2))

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.ravel(), Y_pred2)

plot_roc_curve(fpr_keras, tpr_keras)


y_1 = to_categorical(encoder.fit_transform(y_))

print('----- Testing Model X_ED -------')

model.evaluate(X_ed, y_1)

Y_pred = model.predict(X_ed)
y_pred = np.argmax(Y_pred, axis =1)

y_test_tr = np.argmax(y_1, axis = 1)
metricas(y_pred, y_test_tr)
plt.show()



print('----- Testing Model X_ES -------')

model.evaluate(X_es, y_1)

Y_pred = model.predict(X_es)
y_pred = np.argmax(Y_pred, axis =1)

metricas(y_pred, y_test_tr)



id_df = pd.read_csv(path+'\ids.csv', index_col=0)

model_feats = concat_DLR(model, X_ed, X_es, id_df)

model_feats.to_csv(path+'\_alex_features.csv')

