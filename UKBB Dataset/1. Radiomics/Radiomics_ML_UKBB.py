#!/usr/bin/env python
# coding: utf-8

# # ACDC Challenge 
#
# In this notebook we will try to obatain the best model for classiying CVD with the use of radiomics data.
#
# We will create a pipeline to work through the data and compare the experiments

# In[1]:


### import the packages and libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV, RFE
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
# from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import seaborn as sns


# ## Machine Learning Pipeline
#
# 1. Dataset
# 2. Feature Engineering/Selection
# 4. ML Algorithms
# 5. Testing on the Data

# ### 1. Dataset
#
# We will load the training and test dataset, which contain the extracted radiomics of 100 and 50 patients, respectively. It will also contain the clinical data (only height and weight available) and the class of the patient (each of the 4 diseases or normal state). We will create a function to prepare the datasets.

# In[2]:


train_df = pd.read_csv('Extracted_radiomics\ACDC_(Radiomics+Clinical)_Training.csv')
print(train_df.shape)
test_df = pd.read_csv('Extracted_radiomics\ACDC_(Radiomics+Clinical)_Testing.csv')
print(test_df.shape)


# We know from prior knowledge that both the train and test dataset are equally balanced.

# We also know that there are features that are not numeric and are just the product of the of the radiomics library that will not be needed for our model. We show them and we are going to eliminate them.

# In[3]:


train_df = train_df.loc[:,~ train_df.columns.str.startswith('diagnostics')]
test_df = test_df.loc[:,~ test_df.columns.str.startswith('diagnostics')]
print(train_df.shape)
print(test_df.shape)


# In[67]:


y_train = train_df['class']
y_test = test_df['class']


# In[70]:


ED_train_df = train_df.filter(regex='ED')
ES_train_df = train_df.filter(regex='ES')

ED_test_df = test_df.filter(regex='ED')
ES_test_df = test_df.filter(regex='ES')


# In[50]:


# ED_train_df = pd.concat([ED_train_df, class_df_train], axis=1)
# ES_train_df = pd.concat([ES_train_df, class_df_train], axis=1)

# ED_test_df = pd.concat([ED_test_df, class_df_test], axis=1)
# ES_test_df = pd.concat([ES_test_df, class_df_test], axis=1)


# In[72]:


rad_train = pd.concat([ED_train_df, ES_train_df], axis = 1)
rad_test = pd.concat([ED_test_df, ES_test_df], axis = 1)


# In[73]:


med_info_train = train_df.iloc[:,-3:-1]
med_info_test = test_df.iloc[:,-3:-1]


# In[74]:


all_data_train = pd.concat([rad_train, med_info_train], axis=1)
all_data_test = pd.concat([rad_test, med_info_test], axis=1)


# ### 2. Feature Engineering/Selection

# ##### Feature Engineering
# * MinMaxScaler for all the features 
# * Encoding Class values 

# In[11]:


# def prepare_traindata(train_df, test_df, clinical=True):
#     clin = -3
    
#     if clinical == True:
#         clin = -1
        
#     X_train,y_train = train_df.iloc[:,:clin], train_df['class'] #Training/Val dataset
#     X_test, y_test = test_df.iloc[:,:clin], test_df['class']  #Test dataset
    
#     #tools scaling and labelling
#     scaler = MinMaxScaler()
#     encoder = LabelEncoder()
    
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     y_train= encoder.fit_transform(y_train)
#     y_test = encoder.fit_transform(y_test)
    
#     return X_train, X_test, y_train,y_test


# In[12]:


#X_train, X_test, y_train,y_test = prepare_traindata(train_df, test_df)


# In[13]:


#X_train.shape


# ##### Feature Selection
# 
# For this section we aim to apply several techiniques for feature selection and compare the performance with each proposed model/learning algorithm of the section 3

# Set of Techniques we are going to use:
# 1. <u> K-Best </u>: This method is a **filter method**, which select features according the K-Highest Score of an statistical Test. For this one we are going to use both **Anova-Test** and **Chi-Suared-Test**.
# 
# 2. <u> Sequential Forward Feature Elimination </u>: automatically select a subset of features that is most relevant to the problem. The goal of feature selection is two-fold: We want to improve the computational efficiency and reduce the generalization error of the model by removing irrelevant features or noise. http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-1-a-simple-sequential-forward-selection-example

# In[60]:


def GridSearch_fun(clf, grid, X_train, y_train, X_test):
    gscv = GridSearchCV(clf, grid, scoring='accuracy', cv= StratifiedKFold(4), return_train_score=True)
    
#     print(gscv.get_params().keys())
    
    gscv.fit(X_train, y_train)
    
    #print(gscv.best_parameters)
    #print(gscv.best_score_)
    
    y_pred = gscv.predict(X_test)
    
    return y_pred, gscv.best_params_


# #### Models

# This fucniton works, will need to be checked later becuase for the other methods we are using pipeline so is nested, so we need to adapt the param grids, we might want to do the same for this technique to work on the models for every technique

# In[61]:


## Creating Functions to test these techniques

#K-Best

def K_Best_testing(model, X_train,y_train, X_test, y_test, parameter_grid, score_function= f_classif):
    '''Function to test the number of features using the Method K-Best'''
    '''score_function defautl Anova Test'''
    
    trial_set = np.arange(1,X_train.shape[1], step=1)
    test_accuracy= []
    parameters = []
    y_pred = []
    
    for x in trial_set:
        selector = SelectKBest(score_function, k=x)
        
        selector.fit(X_train, y_train)
        
        # Get columns to keep and create new dataframe with those only
        
        X_train_tr = selector.transform(X_train)
        X_test_tr = selector.transform(X_test)
        
        #Splitting the data with the corresponding features
        
        y_test_pred, best_parameters = GridSearch_fun(model, parameter_grid, X_train_tr, y_train, X_test_tr)
    
        test_acc = accuracy_score(y_test, y_test_pred)
        
        y_pred.append(y_test_pred)
        test_accuracy.append(test_acc)
        parameters.append(best_parameters)
    
    best_nfeatures=test_accuracy.index(max(test_accuracy))+1
    best_model = parameters[best_nfeatures]
    
    print("Nº of Features Recommended :",  test_accuracy.index(max(test_accuracy))+1, "with a Test acc of :", max(test_accuracy))
    print("Best Model: \n", best_model )
    
    plt.figure(figsize=(8,6))
    #plt.plot(trial_set, train_accuracy, label='Train accuracy', c='blue')
    plt.plot(trial_set, test_accuracy, label='test accuracy', c='yellow')
    plt.legend()
    plt.title('Trial Set - Accuracy')
    plt.xlabel('Nº of Features')
    plt.ylabel('Accuracy')
    
    plt.show()
    
    selector = SelectKBest(score_function, k=best_nfeatures)
    selector.fit(X_train, y_train)
        
    # Get columns to keep and create new dataframe with those only
        
    cols = selector.get_support(indices=True)
    features_df_new = train_df.iloc[:,cols]
    K_best = list(features_df_new.columns)  ### Saving the name of the features
    
    y_test_pred = y_pred[best_nfeatures]
    

    cm = confusion_matrix(y_test, y_test_pred)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,4))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    sns.heatmap()

    class_list = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # f1_value = f1_score(y_val, y_val_pred, average='weighted')
    # f1_value

    print(classification_report(y_test, y_test_pred,target_names=class_list))
    
    return K_best


def KBest_GS(X_train, y_train, X_test, y_test, model, param_grid, df ):
    
    selector = SelectKBest()
    
    ### Pipeline
    
    ### we would need to adapt the "NUMBER OF FEATURES PARAMETER OF THE GRID"
    
    pipe = Pipeline([('selector', selector), 
                 ('model', model)])

    featss = np.array(range(3,X_train.shape[1]))

    dict_1 = {'selector__score_func': [f_classif, chi2],
              'selector__k':featss}   #### para pruebas
    
    dict_1.update(param_grid)
    
    gs = GridSearchCV(estimator=pipe, 
                  param_grid=dict_1, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=StratifiedKFold(4, shuffle=True, random_state=42),
                  iid=True,
                  refit=True,
                     verbose=3)
    
    print(pipe.get_params().keys())
    
    gs = gs.fit(X_train, y_train)
    
    print("Best Model", gs.best_params_)
    
    print('Best score:', gs.best_score_)
    
    y_test_pred = gs.predict(X_test)
    
    test_acc = accuracy_score(y_test,y_test_pred)
    
    print("\n Test Accuracy with best estimator: ", test_acc)
    
    cm = confusion_matrix(y_test, y_test_pred)
        
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8,4))
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    class_list = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test, y_test_pred,target_names=class_list))
    
    cols = gs.best_estimator_.steps[0][1].get_support(indices=True)
    features_df_new = df.iloc[:,cols]
    K_best = list(features_df_new.columns)
    
    print(K_best)
    
    return gs
    


# ---
# 
# #### Sequential Foward Selection

# In[64]:


def SFS_GS(X_train, y_train, X_test, y_test, model, param_grid):
    
    #Setting up the SFS
    sfs1 = SFS(estimator=model, 
           k_features=X_train.shape[1],
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=StratifiedKFold(4, shuffle=True, random_state=42))
    
    ### Pipeline
    
    ### we would need to adapt the "NUMBER OF FEATURES PARAMETER OF THE GRID"
    
    pipe = Pipeline([('sfs', sfs1), 
                 ('model', model)])
    
    #dict_1 = {'sfs__k_features':list(range(1,X_train.shape[1]))}   #### para pruebas
    
    dict_1 = {'sfs__k_features':[5,10,15]}  #Testing
    
    dict_1.update(param_grid)
    
    gs = GridSearchCV(estimator=pipe, 
                  param_grid=dict_1, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=StratifiedKFold(4, shuffle=True, random_state=42),
                  verbose=3,
                  refit=True)
    
    print(pipe.get_params().keys())
    
    gs = gs.fit(X_train, y_train)
    
#     print(gs.best_estimator_.steps)
    
    print("Best Model", gs.best_params_)
    
#     print('Best score:', gs.best_score_)
    
    y_test_pred = gs.predict(X_test)
    
    test_acc = accuracy_score(y_test,y_test_pred)
    
    print("\n Test Accuracy with best estimator: ", test_acc)
    
    cm = confusion_matrix(y_test, y_test_pred)
        
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8,4))
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    class_list = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']

    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print(classification_report(y_test, y_test_pred,target_names=class_list))
    
    feats = gs.best_estimator_.steps[0][1].k_feature_idx_
    
    feats_2= np.asanyarray(feats)
    
    print(train_df.iloc[:,feats_2].columns)
    
    return gs, pipe
    
    


# In[65]:


#Features from Irm
#train_prueba = train_df.iloc[:,[288,421,411,26,416,585,674,162,774, -1]]
#test_prueba = test_df.iloc[:,[288,421,411,26,416,585,674,162,774, -1]]


# In[24]:


#X_train1, X_train2,y_train1, y_train2 = train_test_split(X_train, y_train, shuffle=True, stratify=y_train, random_state=42)


# In[66]:


def processing(X_train,y_train, X_test, y_test):
    #tools scaling and labelling
    scaler = MinMaxScaler()
    encoder = LabelEncoder()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    
    return X_train, y_train, X_test, y_test



#ED Alone
X_train_ED, y_train, X_test_ED, y_test = processing(ED_train_df, y_train, ED_test_df, y_test)

#ES Alon

X_train_ES, y_train, X_test_ES, y_test = processing(ES_train_df, y_train, ES_test_df, y_test)

#Radiomics 

X_train_rad, y_train, X_test_rad, y_test = processing(rad_train, y_train, rad_test, y_test)

#All data

X_train_all, y_train, X_test_all, y_test = processing(all_data_train, y_train, all_data_test, y_test)


# ### 3. ML Algorithms
# 
# Let's test some algorithms with the feature selection techniques

# In[77]:


#Support Vector Classifier

model_SVC = SVC(gamma = 'scale', max_iter= 5000, random_state=42)

param_grid_SVC =  {'model__kernel':('linear', 'rbf'), 
                   'model__C':[5, 10]}

# param_grid_SVC_nested =  { 'sfs__estimator__kernel': ['linear', 'rbf', 'poly']},
#                           {'degree':[1,5,10]},{
#                    'sfs__estimator__C':[5]}

param_grid_SVC_nested_2 =  { 'selector__estimator__kernel': ['linear', 'rbf'],
                   'selector__estimator__C':[15]}

param_grid_SVC_test_2 =  { 
                   'estimator__model__C':[0.5, 1,5,10]}


#-------------------------------------------------------

#Random Forest 

# Number of trees in random forest
n_estimators = [10, 100,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4,6,8,10]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

# param_grid_RF = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

param_grid_RF = {'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}

param_grid_RF_2 = {'estimator__n_estimators': n_estimators,
               'estimator__bootstrap': bootstrap}

model_RF = RandomForestClassifier(random_state=42)

#--------------------------------------------------------------

# Logistic Regression

param_grid_LR_nested = {'model__penalty': ['l1','l2'], 
               'model__C': [0.1,1,10,100, 200]}

param_grid_LR = {'penalty': ['l1','l2'], 
               'C': [0.1,1,10,100, 200]}

model_LR = LogisticRegression( multi_class='auto', random_state=42)

#-------------------------------------------------------------------

### Gradient Boosting

# parameters_GRB = {
#     "loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3,5,8],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     }

# param_grid_GRB_nested = {
#     "model__loss":["deviance"],
#     "model__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "model__min_samples_split": np.linspace(0.1, 0.5, 12),
#     "model__min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "model__max_depth":[3,5,8],
#     "model__max_features":["log2","sqrt"],
#     "model__criterion": ["friedman_mse",  "mae"],
#     "model__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "model__n_estimators":[10]
#     }

# model_GRB = GradientBoostingClassifier(random_state=42)

#--------------------------------------------------------------------

### Adaboost

#...


# In[37]:





# In[39]:


#gs = KBest_GS(X_train_ED, y_train, X_test_ED, y_test, model_SVC, param_grid_SVC)
#gs = KBest_GS(X_train_ED, y_train, X_test_ED, y_test, model_RF, param_grid_RF)


# In[148]:


#gs, pipe = SFS_GS(X_train, y_train, X_test, y_test, model_SVC, param_grid_SVC_nested)

