### import the packages and libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectKBest, f_classif, chi2, RFECV, RFE
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle


# ### 1. Data Loading

###Radiomics

RF_df_train = pd.read_csv(r'C:\Users\alex1\Desktop\ACDC\Extracted_radiomics\ACDC_(Radiomics+Clinical)_Training.csv')
print(RF_df_train.shape)
RF_df_test = pd.read_csv(r'C:\Users\alex1\Desktop\ACDC\Extracted_radiomics\ACDC_(Radiomics+Clinical)_Testing.csv')
print(RF_df_test.shape)


RF_df_train = RF_df_train.loc[:,~ RF_df_train.columns.str.startswith('diagnostics')]
RF_df_test = RF_df_test.loc[:,~ RF_df_test.columns.str.startswith('diagnostics')]
print(RF_df_train.shape)
print(RF_df_test.shape)


# Also, we separate the medical inputs and separate the independent variable (disease)

# In[9]:


radiomics_train = RF_df_train.iloc[:,:-3]
med_info_train = RF_df_train.iloc[:,-3:-1]
y_train = RF_df_train.iloc[:,-1]




radiomics_test = RF_df_test.iloc[:,:-3]
med_info_test = RF_df_test.iloc[:,-3:-1]
y_test = RF_df_test.iloc[:,-1]


#### Deeply Learned Features
### Inception

path = r'Extracted_Features'

incep_DF_train_ED = pd.read_csv(path+'\Extracted_Features_IncepModel_train_ED.csv', header=None)
incep_DF_train_ES = pd.read_csv(path+'\Extracted_Features_IncepModel_train_ES.csv', header=None)
incep_DF_test_ED = pd.read_csv(path+'\Extracted_Features_IncepModel_test_ED.csv', header=None)
incep_DF_test_ES = pd.read_csv(path+'\Extracted_Features_IncepModel_test_ES.csv', header=None)




AlexNet_DF_train_ED = pd.read_csv(path+'\Extracted_Features_AlexNet_train_ED.csv', header=None)
AlexNet_DF_train_ES = pd.read_csv(path+'\Extracted_Features_AlexNet_train_ES.csv', header=None)
AlexNet_DF_test_ED = pd.read_csv(path+'\Extracted_Features_AlexNet_test_ED.csv', header=None)
AlexNet_DF_test_ES = pd.read_csv(path+'\Extracted_Features_AlexNet_test_ES.csv', header=None)



AlexNetdil_DF_train_ED = pd.read_csv(path+'\Extracted_Features_AlexNetDil_train_ED.csv', header=None)
AlexNetdil_DF_train_ES = pd.read_csv(path+'\Extracted_Features_AlexNetDil_train_ES.csv', header=None)
AlexNetdil_DF_test_ED = pd.read_csv(path+'\Extracted_Features_AlexNetDil_test_ED.csv', header=None)
AlexNetdil_DF_test_ES = pd.read_csv(path+'\Extracted_Features_AlexNetDil_test_ES.csv', header=None)


# In[14]:


###################### We leave this out for the moments ############################
#VGG_DF_train = pd.read_csv(path+'\VGG_Extracted_Features_train.csv', header=None) ##
#VGG_DF_test = pd.read_csv(path+'\VGG_Extracted_Features_test.csv', header=None)   ##
####################################################################################


# In[76]:


#Late Merging Model
LM_DF_train = pd.read_csv(path+'\LM_features_train.csv', header=None)
LM_DF_test_ES = pd.read_csv(path+'\LM_features_test_ES.csv', header=None)
LM_DF_test_ED = pd.read_csv(path+'\LM_features_test_ES.csv', header=None)


# In[77]:


LM_DF_train.shape


# In[78]:


np_LM_train=np.asanyarray(LM_DF_train)
np_LM_train.shape


# In[79]:


def divide_LM_train():
    ED_list = list()
    ES_list = list()
    
    for i in np.arange(1,600,3):
        if (i % 2) == 0:
            ES_list.append(np_LM_train[i,:])
        else:
            ED_list.append(np_LM_train[i,:])
         
    ES = np.asanyarray(ES_list)
    ED = np.asanyarray(ED_list)

    return ES, ED 


# In[80]:


ES, ED = divide_LM_train()


# In[81]:


LM_train_ES = pd.DataFrame(ES)
LM_train_ED = pd.DataFrame(ED)


# In[82]:


LM_DF_test_ES.shape


# In[83]:


def col_names(df, cycle):
    col_names= ['{}_dlf_{}'.format(cycle, x) for x in range(len(df.columns))]
    df.columns = col_names
    
    return df    


# In[105]:


#### Creation of datasets
incep_DF_train_ED = col_names(incep_DF_train_ED, cycle='ED')
incep_DF_train_ES = col_names(incep_DF_train_ES, cycle='ES')

incep_DF_train = pd.concat([incep_DF_train_ED, incep_DF_train_ES], axis= 1)
incep_train_tot = pd.concat([incep_DF_train, med_info_train], axis=1)

incep_DF_test_ED = col_names(incep_DF_test_ED , cycle='ED')
incep_DF_test_ES = col_names(incep_DF_test_ES, cycle='ES')

incep_DF_test = pd.concat([incep_DF_test_ED, incep_DF_test_ES], axis= 1)
incep_test_tot = pd.concat([incep_DF_test, med_info_test], axis=1)


# In[104]:


AlexNet_DF_train_ED = col_names(AlexNet_DF_train_ED, cycle='ED')
AlexNet_DF_train_ES = col_names(AlexNet_DF_train_ES, cycle= 'ES')

AlexNet_DF_train = pd.concat([AlexNet_DF_train_ED, AlexNet_DF_train_ES], axis= 1)
AlexNet_train_tot = pd.concat([AlexNet_DF_train, med_info_train], axis = 1)

AlexNet_DF_test_ED = col_names(AlexNet_DF_test_ED, cycle='ED')
AlexNet_DF_test_ES = col_names(AlexNet_DF_test_ES, cycle='ES')

AlexNet_DF_test = pd.concat([AlexNet_DF_test_ED, AlexNet_DF_test_ES], axis= 1)
AlexNet_test_tot = pd.concat([AlexNet_DF_test, med_info_test], axis = 1)


# In[173]:


AlexNetdil_DF_train_ED = col_names(AlexNetdil_DF_train_ED, cycle='ED')
AlexNetdil_DF_train_ES = col_names(AlexNetdil_DF_train_ES, cycle= 'ES')

AlexNetdil_DF_train = pd.concat([AlexNetdil_DF_train_ED, AlexNetdil_DF_train_ES], axis= 1)
AlexNetdil_DF_train_tot = pd.concat([AlexNet_DF_train, med_info_train], axis= 1)

AlexNetdil_DF_test_ED = col_names(AlexNetdil_DF_test_ED, cycle='ED')
AlexNetdil_DF_test_ES = col_names(AlexNetdil_DF_test_ES, cycle='ES')

AlexNetdil_DF_test = pd.concat([AlexNetdil_DF_test_ED, AlexNetdil_DF_test_ES], axis= 1)
AlexNetdil_DF_test_tot = pd.concat([AlexNet_DF_test, med_info_test], axis= 1)


# In[180]:


LM_train_ED = col_names(LM_train_ED, cycle='ED')
LM_train_ES = col_names(LM_train_ES, cycle='ES')

LM_train = pd.concat([LM_train_ED, LM_train_ES], axis=1)
LM_train_tot = pd.concat([LM_train, med_info_train], axis=1)

LM_DF_test_ED = col_names(LM_DF_test_ED, cycle='ED')
LM_DF_test_ES = col_names(LM_DF_test_ES, cycle='ES')

LM_test = pd.concat([LM_DF_test_ED, LM_DF_test_ES], axis=1)
LM_test_tot = pd.concat([LM_test, med_info_test], axis=1)


# In[47]:


# VGG_DF_train = col_names(VGG_DF_train)
# VGG_DF_test = col_names(VGG_DF_test)


# ### 2. Fusion of Features and data processing

# In this section we combine both data modalities in a single data frame. Also, we will be performing the necessary scale for the dataset

# In[109]:


class testing_model:
    def __init__(self, data1, data2, data3, data4):
        self.train1 = data1
        self.train2 = data2
        self.test1 = data3
        self.test2 = data4
        
    def concatenate(self):
        X_train = pd.concat([self.train1, self.train2], axis=1)
        X_test = pd.concat([self.test1, self.test2], axis = 1)
        
        return X_train, X_test


# In[111]:


def processing(X_train, y_train, X_test, y_test):
    # tools scaling and labelling
    scaler = MinMaxScaler()


    encoder = LabelEncoder()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    return X_train, y_train, X_test, y_test


# ### 3. Machine Learning

# - prueba 1 = Radiomics + ED
# - prueba 2 = Radiomics + ES
# - prueba 3 = Radiomics + ES + ED
# - prubea 4 = Radiomics + ES + ED + Med

# In[112]:


# In[113]:




# #### Grid Search K-Best

# In[156]:


def KBest_GS(X_train, y_train, X_test, y_test, model, param_grid, df):
    
    featss =np.arange(3,X_train.shape[1],1)
    
    selector = SelectKBest()
    
    ### Pipeline
    
    ### we would need to adapt the "NUMBER OF FEATURES PARAMETER OF THE GRID"
    
    pipe = Pipeline([('selector', selector), 
                 ('model', model)])
    
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
                verbose=2)
    
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
    print(df.iloc[:,cols].columns)
    
    
    return gs
    


# --- 
# #### Grid Search Sequential Forward Elimination

# In[267]:


def SFS_GS(X_train, y_train, X_test, y_test, model, param_grid, df):
    
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
    
    dict_1 = {'sfs__k_features':[5, 10]}  #pruebas
    
    dict_1.update(param_grid)
    
    gs = GridSearchCV(estimator=pipe, 
                  param_grid=dict_1, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=StratifiedKFold(4, shuffle=True, random_state=42),
                  verbose=3,
                  refit=True)
    
#     print(pipe.get_params().keys())
    
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
    
    print(df.iloc[:,feats_2].columns)
    
    return gs, pipe
    


# In[255]:


#Support Vector Classifier

model_SVC = SVC(kernel = 'linear', gamma = 'scale', max_iter= 5000, random_state=42)
model_SVC_2 = SVC(kernel = 'linear', gamma = 'scale', max_iter= 5000, random_state=42)

param_grid_SVC =  {'model__C':[5, 10],
                   'model__kernel':('linear', 'rbf')
                   }

param_grid_SVC_nested_2 =  { 'selector__estimator__kernel': ['linear', 'rbf'],
                   'selector__estimator__C':[15]}

param_grid_SVC_test_2 =  { 
                   'estimator__model__C':[0.5, 1,5,10]}


# In[82]:





# In[83]:





# In[19]:



# --- 
# #### Using the features extracted from the Inception

# In[226]:

#
# X_train1, X_test1 = testing_model(radiomics_train, incep_DF_train_ED, radiomics_test, incep_DF_test_ED).concatenate()
# X_train2, X_test2 = testing_model(radiomics_train, incep_DF_train_ES, radiomics_test, incep_DF_test_ES).concatenate()
# X_train3, X_test3 = testing_model(radiomics_train, incep_DF_train, radiomics_test, incep_DF_test).concatenate()
# X_train4, X_test4 = testing_model(radiomics_train, incep_train_tot, radiomics_test, incep_test_tot).concatenate()





# --- 
# #### Using the features extracted from the AlexNet





X_train1, X_test1 = testing_model(radiomics_train, AlexNet_DF_train_ED, radiomics_test, AlexNet_DF_test_ED).concatenate()
X_train2, X_test2 = testing_model(radiomics_train, AlexNet_DF_train_ES, radiomics_test, AlexNet_DF_test_ES).concatenate()
X_train3, X_test3 = testing_model(radiomics_train, AlexNet_DF_train, radiomics_test, AlexNet_DF_test).concatenate()
X_train4, X_test4 = testing_model(radiomics_train, AlexNet_train_tot, radiomics_test, AlexNet_test_tot).concatenate()


X_train_1, y_train, X_test_1, y_test= processing(X_train1, y_train, X_test1, y_test)

X_train_2, y_train, X_test_2, y_test= processing(X_train2, y_train, X_test2, y_test)

X_train_3, y_train, X_test_3, y_test= processing(X_train3, y_train, X_test3, y_test)

X_train_4, y_train, X_test_4, y_test= processing(X_train4, y_train, X_test4, y_test)


# In[249]:





# In[260]:


#gs, pipe = SFS_GS(X_train_2, y_train, X_test_2, y_test, model_SVC, param_grid_SVC, X_train2)


# In[261]:


#gs, pipe = SFS_GS(X_train_4, y_train, X_test_4, y_test, model_SVC, param_grid_SVC, X_train4)


# In[165]:


#gs = KBest_GS(X_train_2, y_train, X_test_2, y_test, model_SVC, param_grid_SVC, X_train2)


# # ---
# # #### Using the features extracted from the VGG
#
# # In[324]:
#
#
# X_train_df_3 = pd.concat([radiomics_train, VGG_DF_train], axis=1)
# X_test_df_3 = pd.concat([radiomics_test, VGG_DF_test], axis=1)


# In[325]:




# In[ ]:





# --- 
# #### Using the features extracted from the AlexNetDil

# In[175]:


# X_train1, X_test1 = testing_model(radiomics_train, AlexNetdil_DF_train_ED, radiomics_test, AlexNetdil_DF_test_ED).concatenate()
# X_train2, X_test2 = testing_model(radiomics_train, AlexNetdil_DF_train_ES, radiomics_test, AlexNetdil_DF_test_ES).concatenate()
# X_train3, X_test3 = testing_model(radiomics_train, AlexNetdil_DF_train, radiomics_test, AlexNetdil_DF_test).concatenate()
# X_train4, X_test4 = testing_model(radiomics_train, AlexNetdil_DF_train_tot, radiomics_test, AlexNetdil_DF_test_tot).concatenate()


# In[215]:




# In[ ]:





# In[ ]:





# --- 
# #### Using the features extracted from the Late Merging Model

# In[262]:


# X_train1, X_test1 = testing_model(radiomics_train, LM_train_ED, radiomics_test, LM_DF_test_ED).concatenate()
# X_train2, X_test2 = testing_model(radiomics_train, LM_train_ES, radiomics_test, LM_DF_test_ES).concatenate()
# X_train3, X_test3 = testing_model(radiomics_train, LM_train, radiomics_test, LM_test).concatenate()
# X_train4, X_test4 = testing_model(radiomics_train, LM_train_tot, radiomics_test, LM_test_tot).concatenate()


# In[263]:






# In[ ]:


#gs = KBest_GS(X_train_2, y_train, X_test_2, y_test, model_SVC, param_grid_SVC, X_train2)


# In[268]:


#gs, pipe = SFS_GS(X_train_4, y_train, X_test_4, y_test, model_SVC, param_grid_SVC, X_train4)

