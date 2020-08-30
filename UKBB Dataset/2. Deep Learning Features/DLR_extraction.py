import os
# import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import time
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model

path='/home/alejandro/Documentos/files_master/DL_matrices/ANGINA/'

# model =load_model(path+'alexnet_model.hdf5')

def model_converter(model):
  model_FE = Model(inputs=model.inputs, outputs= model.layers[-2].output)
  return model_FE


def feature_extractor(model, X):
  extract_feats = list()
  for patient in range(X.shape[0]):
    img1 = X[patient]
    img_ = img1.reshape((1,150,150,3))
    FE = model.predict(img_)

    extract_feats.append(FE[0,:])

    output_vector = np.asanyarray(extract_feats)

    output_df = pd.DataFrame(output_vector)

  return output_df


def feature_extractor_LM(model, X):
    extract_feats = list()
    for patient in range(X.shape[0]):
        img1 = X[patient]
        img_ = img1.reshape((1, 150, 150, 3))
        FE = model.predict([img_[:, :, :, [0]],
                            img_[:, :, :, [1]],
                            img_[:, :, :, [2]]])

        extract_feats.append(FE[0, :])

    output_vector = np.asanyarray(extract_feats)

    output_df = pd.DataFrame(output_vector)

    return output_df


def col_names(df, cycle):
    col_names = ['{}_dlf_{}'.format(cycle, x) for x in range(len(df.columns))]
    df.columns = col_names

    return df


def concat_DLR(model, X_ed, X_es, id_df):
    '''concatenate features'''
    model_FE = model_converter(model)
    DLR_ED = feature_extractor(model_FE, X_ed)
    DLR_ED = col_names(DLR_ED, 'ED')
    DLR_ES = feature_extractor(model_FE, X_es)
    DLR_ES = col_names(DLR_ES, 'ES')
    joined_DLR = pd.concat([id_df, DLR_ED, DLR_ES], axis = 1)

    return joined_DLR

def concat_DLR_LM(model, X_ed, X_es, id_df):
    '''concatenate features'''
    model_FE = model_converter(model)
    DLR_ED = feature_extractor_LM(model_FE, X_ed)
    DLR_ED = col_names(DLR_ED, 'ED')
    DLR_ES = feature_extractor_LM(model_FE, X_es)
    DLR_ES = col_names(DLR_ES, 'ES')
    joined_DLR = pd.concat([id_df, DLR_ED, DLR_ES], axis = 1)

    return joined_DLR