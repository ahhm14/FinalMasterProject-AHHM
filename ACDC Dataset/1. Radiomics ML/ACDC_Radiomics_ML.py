
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


train_df = pd.read_csv('Extracted_radiomics\ACDC_(Radiomics+Clinical)_Training.csv')
print(train_df.shape)
test_df = pd.read_csv('Extracted_radiomics\ACDC_(Radiomics+Clinical)_Testing.csv')
print(test_df.shape)

train_df = train_df.loc[:,~ train_df.columns.str.startswith('diagnostics')]
test_df = test_df.loc[:,~ test_df.columns.str.startswith('diagnostics')]
print(train_df.shape)
print(test_df.shape)

y_train = train_df['class']
y_test = test_df['class']

ED_train_df = train_df.filter(regex='ED')
ES_train_df = train_df.filter(regex='ES')

ED_test_df = test_df.filter(regex='ED')
ES_test_df = test_df.filter(regex='ES')


rad_train = pd.concat([ED_train_df, ES_train_df], axis = 1)
rad_test = pd.concat([ED_test_df, ES_test_df], axis = 1)

med_info_train = train_df.iloc[:,-3:-1]
med_info_test = test_df.iloc[:,-3:-1]


all_data_train = pd.concat([rad_train, med_info_train], axis=1)
all_data_test = pd.concat([rad_test, med_info_test], axis=1)

def GridSearch_fun(clf, grid, X_train, y_train, X_test):
    gscv = GridSearchCV(clf, grid, scoring='accuracy', cv= StratifiedKFold(4), return_train_score=True)
    
#     print(gscv.get_params().keys())
    
    gscv.fit(X_train, y_train)
    
    #print(gscv.best_parameters)
    #print(gscv.best_score_)
    
    y_pred = gscv.predict(X_test)
    
    return y_pred, gscv.best_params_


# #### Models

#K-Best

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

#Support Vector Classifier

model_SVC = SVC(gamma = 'scale', max_iter= 5000, random_state=42)

param_grid_SVC =  {'model__kernel':('linear', 'rbf'), 
                   'model__C':[5, 10]}

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

#gs = KBest_GS(X_train_ED, y_train, X_test_ED, y_test, model_SVC, param_grid_SVC)
#gs = KBest_GS(X_train_ED, y_train, X_test_ED, y_test, model_RF, param_grid_RF)


