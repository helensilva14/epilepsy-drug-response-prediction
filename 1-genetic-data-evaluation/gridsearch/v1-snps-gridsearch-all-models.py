
# coding: utf-8

# ### Import the necessary libraries

# In[1]:

import pickle 
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import os
import warnings
import itertools
import time

from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut, GridSearchCV, train_test_split
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

np.random.seed(10)

import multiprocessing

# check number of processors on current machine
print("Number of CPUs on current machine: %d" % multiprocessing.cpu_count())

# select the processor to be used (comment if processors >= 4)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


# ### Get preprocessed data (241 samples)

# In[2]:

X = pickle.load(open( "../../data/preprocessed/article-genetic-data-features.p", "rb"))
y = pickle.load(open( "../../data/preprocessed/article-genetic-data-labels.p", "rb"))

print(type(X), " | ", X.shape)
print(type(y), " | ", len(y))


# ### Get data splits: train: 70% | validation: 20% | test: 10%

# In[3]:

# intermediate/test split (gives the test set)
X_intermediate, X_test, y_intermediate, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

# train/validation split (gives the train and validation sets)
X_train, X_validation, y_train, y_validation = train_test_split(X_intermediate, y_intermediate, 
                                                                test_size=0.22, random_state=1)
# delete intermediate variables
del X_intermediate, y_intermediate

# print proportions
print('train: {} | validation: {} | test: {}\n'.format(len(y_train), len(y_validation), len(y_test)))
print('train: {}% | validation: {}% | test: {}%'.format(round(len(y_train)/len(y),2),
                                                       round(len(y_validation)/len(y),2),
                                                       round(len(y_test)/len(y),2)))


# ### Create function to plot Confusion Matrix without Normalization

# In[4]:

def plot_confusion_matrix(cm, model_name):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Drug Response Prediction Confusion Matrix without Normalization')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    classNames = ['Refractory','Responsive']
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    # set the layout approach without normalization
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
    # save plot as image 
    plt.savefig('../figures/default-confusion-matrixes/%s-gridsearch-genetic-default-cm' % model_name.lower())
    plt.show()    


# ### Create function to plot a Normalized Confusion Matrix

# In[5]:

def plot_normalized_confusion_matrix(cm, model_name, use_portuguese=False):
    # apply normalization
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot matrix
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # check language to be used
    if(use_portuguese):
        plt.title('Dados genéticos - Matriz de Confusão Normalizada')
        plt.colorbar()
        plt.ylabel('Rótulo verdadeiro') 
        plt.xlabel('Rótulo previsto') 
        classNames = ['Refratário','Responsivo'] 
    else:
        plt.title('Normalized Drug Response Prediction Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        classNames = ['Refractory','Responsive']
    
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    # set the normalized layout approach
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    # save plot as image 
    plt.savefig('../figures/normalized-confusion-matrixes/%s-gridsearch-genetic-normalized-cm.pdf' % model_name.lower(), dpi=300, 
                pad_inches=0, bbox_inches='tight')
    plt.show()


# ### Create function to plot ROC curve

# In[6]:

def plot_roc_curve(fpr, tpr, auc_score, model_name, use_portuguese=False):
    plt.figure(1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') 
    plt.plot(fpr, tpr, color='darkorange', label='AUC = %f)' % auc_score)
    
    # check language to be used
    if(use_portuguese):    
        plt.xlabel('Taxa de falsos positivos')
        plt.ylabel('Taxa de verdadeiros positivos') 
        plt.title('Dados genéticos - Curva ROC') 
    else:
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Drug Response Prediction - ROC Curve')
    
    plt.legend(loc='best')
    # save plot as image 
    plt.savefig('../figures/roc-curves/%s-gridsearch-genetic-roc-curve.pdf' % model_name.lower(), dpi=300)
    plt.show()


# ### Create function to be used to perform model fitting using LOOCV 

# In[7]:

def fit_model(model, X, y):
    
    # prepare a LOOCV object (number of folds equals the number of samples)
    loocv = LeaveOneOut()
    loocv.get_n_splits(X)
    
    # perform cross-validation and get the accuracies
    cv_score = cross_val_score(model, X, y, cv=loocv, scoring='accuracy')
    
    # perform cross-validation and get the predictions and predictions probabilities
    preds = cross_val_predict(model, X, y, cv=loocv)
    predprobs = cross_val_predict(model, X, y, cv=loocv, method='predict_proba')[:,1]
    
    # calculate fpr and tpr values using the y_true and predictions probabilities
    fpr, tpr, _ = metrics.roc_curve(y, predprobs)
    
    # calculate the auc score based on fpr and tpr values
    auc_score = metrics.auc(fpr, tpr)

    # generate the confusion matrix for the model results and slice it into four pieces
    cm = metrics.confusion_matrix(y, preds)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    # print model informqtion
    print("\nModel Report\n")
    print(model) # print the used params for the model

    # print classification report
    print("\nAccuracy (CV Score) : Mean - %.7g | Std - %.7g" % (np.mean(cv_score), np.std(cv_score)))
    print("\nAUC Score : %f" % auc_score)
    
    # calculate sensitivity score
    # specificity: When the actual value is negative, how often is the prediction correct?
    # how "specific" (or "selective") is the classifier in predicting positive instances?
    specificity = TN / float(TN + FP)
    print("\nSpecificity Score : %f" % specificity)
    
    # calculate sensitivity score
    # sensitivity: When the actual value is positive, how often is the prediction correct?
    # how "sensitive" is the classifier to detecting positive instances? Also known as "True Positive Rate" or "Recall"
    sensitivity = TP / float(TP + FN)
    print("\nSensitivity Score : %f" % sensitivity)
    
    # print a complete classification metrics report
    print("\n" + metrics.classification_report(y, preds)) 
    
    # get current model name
    model_name = str(model).split('(')[0]
    
    # plot confusion matrix
    plot_confusion_matrix(cm, model_name)
    
    # plot normalized confusion matrix
    plot_normalized_confusion_matrix(cm, model_name) 

    # plot the roc curve
    plot_roc_curve(fpr, tpr, auc_score, model_name)
    
    return predprobs, fpr, tpr, auc_score 


# ### Create function to perform model fitting using GridSearchCV

# In[8]:

def gridsearch_fit(model, param_grid, scoring='accuracy'):
    
    # DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and 
    # will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
    grid = GridSearchCV(model, param_grid, cv=5, scoring=scoring, iid=True)
    
    start_time = time.time()
    
    # fit the grid with data
    grid.fit(X_train, y_train)
    
    print("Execution time: " + str((time.time() - start_time)) + ' ms')

    # show single best score achieved across all params
    print("\nAccuracy (GridSearchCV Score) : %f" % (grid.best_score_))

    # show dictionary containing the parameters used to generate that score
    print("\nBest parameters : %s \n" % grid.best_params_)

    # get actual model object fit with those best parameters 
    estimator = grid.best_estimator_
    
    # print all model params
    print(estimator) 
    
    # get the predictions and predictions probabilities
    preds = estimator.predict(X_validation)
    predprobs = estimator.predict_proba(X_validation)[:, 1]
    
    # calculate fpr and tpr values using the y_true and predictions probabilities
    fpr, tpr, _ = metrics.roc_curve(y_validation, predprobs)
    
    # calculate the auc score based on fpr and tpr values
    auc_score = metrics.auc(fpr, tpr)

    print("\nAUC Score : %f" % auc_score)
    
    # generate the confusion matrix for the model results and slice it into four pieces
    cm = metrics.confusion_matrix(y_validation, preds)
    TP, TN, FP, FN = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    
    # calculate sensitivity score
    # specificity: When the actual value is negative, how often is the prediction correct?
    # how "specific" (or "selective") is the classifier in predicting positive instances?
    specificity = TN / float(TN + FP)
    print("\nSpecificity Score : %f" % specificity)
    
    # calculate sensitivity score
    # sensitivity: When the actual value is positive, how often is the prediction correct?
    # how "sensitive" is the classifier to detecting positive instances? Also known as "True Positive Rate" or "Recall"
    sensitivity = TP / float(TP + FN)
    print("\nSensitivity Score : %f" % sensitivity)
    
    # print a complete classification metrics report
    print("\n" + metrics.classification_report(y_validation, preds)) 
    
    # get current model name
    model_name = str(estimator).split('(')[0]
    
    # plot confusion matrix
    plot_confusion_matrix(cm, model_name)
    
    # plot normalized confusion matrix
    plot_normalized_confusion_matrix(cm, model_name) 

    # plot the roc curve
    plot_roc_curve(fpr, tpr, auc_score, model_name)
    
    return predprobs, fpr, tpr, auc_score, preds


# ### Create a DecisionTreeClassifier model with parameter tunning

# In[9]:

dt = DecisionTreeClassifier(random_state=10)

"""
ACCURACY
Execution time: 613.2550058364868 ms
Accuracy (GridSearchCV Score) : 0.779762
Best parameters : {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 16} 
AUC Score : 0.884766
Specificity Score : 0.781250
Sensitivity Score : 1.000000
=================
ROC_AUC
Execution time: 695.0215799808502 ms
Accuracy (GridSearchCV Score) : 0.764823
Best parameters : {'max_depth': 5, 'min_samples_leaf': 9, 'min_samples_split': 27} 
AUC Score : 0.827148
Specificity Score : 0.812500
Sensitivity Score : 0.937500
"""
param_grid_dt = {'max_depth': np.arange(1, 21), 
                 'min_samples_split': np.arange(2, 51), 
                 'min_samples_leaf': np.arange(1, 16)}

# perform model fitting
dt_predprobs, fpr_dt, tpr_dt, auc_dt, dt_preds = gridsearch_fit(dt, param_grid_dt) #, 'roc_auc')

# export predictions
pickle.dump(dt_predprobs, open("../predictions/dt-gridsearch-genetic-predprobs.p", "wb"))
pickle.dump(dt_preds, open("../predictions/dt-gridsearch-genetic-preds.p", "wb"))


# ### Create a RandomForestClassifier model with parameter tunning

# In[10]:

rf = RandomForestClassifier(random_state=10)

"""
ACCURACY
Execution time: 23.998207569122314 ms
Accuracy (GridSearchCV Score) : 0.755952
Best params : {'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 30} 
AUC Score : 0.816406
Specificity Score : 0.937500
Sensitivity Score : 0.312500
==============
ROC_AUC
Execution time: 25.675724029541016 ms
Accuracy (GridSearchCV Score) : 0.811384
Best params : {'criterion': 'gini', 'max_depth': 30, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 30} 
AUC Score : 0.816406
Specificity Score : 0.937500
Sensitivity Score : 0.312500
"""
param_grid_rf1 = {'criterion':['gini','entropy'],
                 'max_depth': [30, 50], 
                 'n_estimators':[10,15,20,25,30],
                 'min_samples_split': [3, 4, 5], 
                 'min_samples_leaf': [3, 4, 5]} # [8, 10, 12]} # [5, 10, 15]} 

"""
Execution time: 63.010621070861816 ms
Accuracy (GridSearchCV Score) : 0.660714
Best params : {'criterion': 'gini', 'max_depth': 30, 'max_features': 2, 
'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 30} 
AUC Score : 0.722656
Specificity Score : 1.000000
Sensitivity Score : 0.000000
"""
param_grid_rf2 = {'criterion':['gini','entropy'],
                 'max_depth': [30, 50], 
                 'max_features': [2, 3],
                 'n_estimators':[30, 50, 100], 
                 'min_samples_split': [8, 10, 12],
                 'min_samples_leaf': [3, 4, 5]} # 

# perform model fitting
rf_predprobs, fpr_rf, tpr_rf, auc_rf, rf_preds = gridsearch_fit(rf, param_grid_rf1)

# export prediction probabilities 
pickle.dump(rf_predprobs, open("../predictions/rf-gridsearch-genetic-predprobs.p", "wb"))


# ### Create a SVM baseline model using a reduced set of tuned parameters

# In[11]:

param_grid_svc1 = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 1, 10, 100]}, 
                    {'kernel': ['linear'], 'C': [0.01, 1, 10, 100]}]

"""
Accuracy (CV Score) : Mean - 0.7966805 | Std - 0.4024682
AUC Score : 0.834271
Specificity Score : 0.870370
Sensitivity Score : 0.645570
"""
# clf = GridSearchCV(SVC(probability=True, random_state=10), param_grid_svc1)

"""
Execution time: 27.8266761302948 ms
Accuracy (GridSearchCV Score) : 0.755952
Best parameters : {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'} 
AUC Score : 0.832031
Specificity Score : 0.906250
Sensitivity Score : 0.687500
"""
param_grid_svc2 = {'kernel': ['rbf', 'linear'],
                   'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 
                   'C': [0.01, 1, 10, 100]}

param_grid_svc3 = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10]}, 
                   {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1]},
                   {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10]}]

"""
Execution time: 72.63676905632019 ms
Accuracy (GridSearchCV Score) : 0.812231
Best parameters : {'C': 0.1, 'decision_function_shape': 'ovo', 'gamma': 0.01, 'kernel': 'linear'} 
AUC Score : 0.792969
Specificity Score : 0.843750
Sensitivity Score : 0.625000
"""
# https://www.kaggle.com/jsnsaji/titanic-classifier
param_grid_svc4 = {'kernel':('linear', 'rbf'), 
              'C':(0.01, 0.1, 1, 0.25, 0.5),
              'gamma': (0.01, 0.1, 1, 2,'auto'),
              'decision_function_shape':('ovo','ovr')}

"""
Execution time: 163.8002290725708 ms
Accuracy (GridSearchCV Score) : 0.767857
Best parameters : {'C': 0.1, 'coef0': 1000.0, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'} 
AUC Score : 0.810547
Specificity Score : 0.812500
Sensitivity Score : 0.625000
"""
param_grid_svc5 = [{'kernel': ['rbf'],
                   'gamma': np.logspace(-4, 3, 10),
                   'C': [1e-3, 1e-2, 1e-1, 1, 10]},
                  {'kernel': ['poly'],
                   'gamma': ['auto'],
                   'degree': [1, 2, 3, 4],
                   'C': [1e-3, 1e-2, 1e-1, 1, 10],
                   'coef0': np.logspace(-4, 3, 10)},
                  {'kernel': ['linear'],
                   'gamma': ['auto'],
                   'C': [1e-3, 1e-2, 1e-1, 1, 10]}]

"""
Execution time: 1524.797747373581 ms
Accuracy (GridSearchCV Score) : 0.773810
Best parameters : {'C': 0.1, 'coef0': 78.47599703514607, 'degree': 2, 'gamma': 0.01, 'kernel': 'poly'} 
AUC Score : 0.812500
Specificity Score : 0.812500
Sensitivity Score : 0.625000
"""
param_grid_svc6 = [{'kernel': ['sigmoid'], #poly
                   'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 
                   'degree': [1, 2, 3, 4],
                   'C': [1e-3, 1e-2, 1e-1, 1, 10],
                   'coef0': np.logspace(-4, 3, 10)}]

svc = SVC(probability=True, random_state=10)

# perform model fitting
svc_predprobs, fpr_svc, tpr_svc, auc_svc, svc_preds = gridsearch_fit(svc, param_grid_svc4)

# export prediction probabilities 
pickle.dump(svc_predprobs, open("../predictions/svc-gridsearch-genetic-predprobs.p", "wb"))


# ### Create a GBM baseline model with parameter tunning

# In[12]:

gbm = GradientBoostingClassifier(random_state=10)

"""
ACCURACY
Execution time: 32.73474621772766 ms
Accuracy (GridSearchCV Score) : 0.785714
Best parameters : {'learning_rate': 0.1, 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 250} 
AUC Score : 0.884766
Specificity Score : 0.875000
Sensitivity Score : 0.562500
"""
param_grid_gbm1 = {'learning_rate': [0.1,0.05,0.01,0.005,0.001], 
                   'n_estimators': [100,250,500,1000],
                   'max_depth': [4], 
                   'max_features': ['sqrt']}

"""
Execution time: 29.629493713378906 ms
Accuracy (GridSearchCV Score) : 0.815476
Best parameters : {'learning_rate': 0.1, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 9, 'n_estimators': 250} 
AUC Score : 0.873047
Specificity Score : 0.875000
Sensitivity Score : 0.437500
"""
param_grid_gbm2 = {'learning_rate': [0.1], 
                   'n_estimators': [250],
                   'max_depth': [4], 
                   'max_features': ['sqrt'],
                   'max_depth':[2,3,4,5,6,7], 
                   'min_samples_leaf':[1,3,5,7,9]}

# perform model fitting
gbm_predprobs, fpr_gbm, tpr_gbm, auc_gbm = gridsearch_fit(gbm, param_grid_gbm1)

# export prediction probabilities 
pickle.dump(gbm_predprobs, open("../predictions/gbm-gridsearch-genetic-predprobs.p", "wb"))


# ### Create a XGB baseline model using default parameters

# In[14]:

# ignore deprecation warnings
# warnings.filterwarnings(action='ignore', category=DeprecationWarning)

xgb = XGBClassifier(random_state=10)

"""
Execution time: 155.3580596446991 ms
Accuracy (GridSearchCV Score) : 0.803571
Best parameters : {'learning_rate': 0.05, 'n_estimators': 250} 
AUC Score : 0.855469
Specificity Score : 0.812500
Sensitivity Score : 0.687500
"""
param_grid_xgb1 = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 
                   'n_estimators':[100,250,500]}

"""
Execution time: 907.6884641647339 ms
Accuracy (GridSearchCV Score) : 0.803571
Best parameters : {'learning_rate': 0.005, 'n_estimators': 1500, 'subsample': 0.8} 
AUC Score : 0.882812
Specificity Score : 0.875000
Sensitivity Score : 0.687500
"""
param_grid_xgb2 = {'learning_rate':[0.1,0.05,0.01,0.005,0.001], 
                   'n_estimators':[500,1000,1500],
                   'subsample': [0.8, 1]}

"""
Execution time: 1175.7173562049866 ms
Accuracy (GridSearchCV Score) : 0.803571
Best parameters : {'learning_rate': 0.005, 'max_depth': 3, 'n_estimators': 1500, 'subsample': 0.8} 
AUC Score : 0.882812
Specificity Score : 0.875000
Sensitivity Score : 0.687500
"""
param_grid_xgb3 = {'learning_rate':[0.01,0.005,0.001], 
                   'n_estimators':[1500, 2000],
                   'subsample': [0.8],
                   'max_depth': [3, 4, 5]}

# perform model fitting
xgb_predprobs, fpr_xgb, tpr_xgb, auc_xgb = gridsearch_fit(xgb, param_grid_xgb3)

# export prediction probabilities 
pickle.dump(xgb_predprobs, open("../predictions/xgb-gridsearch-genetic-predprobs.p", "wb"))


# ### Compare all generated ROC curves

# In[12]:

# get prediction probabilities from individual classifiers
dt_predprobs = pickle.load( open( "./predictions/dt-genetic-predprobs.p", "rb" ) )
rf_predprobs = pickle.load( open( "./predictions/rf-genetic-predprobs.p", "rb" ) )
svc_predprobs = pickle.load( open( "./predictions/svc-genetic-predprobs.p", "rb" ) )
gbm_predprobs = pickle.load( open( "./predictions/gbm-genetic-predprobs.p", "rb" ) )
xgb_predprobs = pickle.load( open( "./predictions/xgb-genetic-predprobs.p", "rb" ) )

# calculate fpr, tpr and auc score for all models using the y_true and its predictions probabilities
fpr_dt, tpr_dt, _ = metrics.roc_curve(y, dt_predprobs)
auc_dt = metrics.auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = metrics.roc_curve(y, rf_predprobs)
auc_rf = metrics.auc(fpr_rf, tpr_rf)

fpr_svc, tpr_svc, _ = metrics.roc_curve(y, svc_predprobs)
auc_svc = metrics.auc(fpr_svc, tpr_svc)

fpr_gbm, tpr_gbm, _ = metrics.roc_curve(y, gbm_predprobs)
auc_gbm = metrics.auc(fpr_gbm, tpr_gbm)

fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y, xgb_predprobs)
auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)

# plot all roc curves into the same image
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot([0, 1], [0, 1], color='black', linestyle='--')  , 
plt.plot(fpr_dt, tpr_dt, color='aqua', label='DT (AUC = %f)' % auc_dt)
plt.plot(fpr_rf, tpr_rf, color='cornflowerblue', label='RF (AUC = %f)' % auc_rf)
plt.plot(fpr_svc, tpr_svc, color='darkorange', label='SVM (AUC = %f)' % auc_svc)
plt.plot(fpr_gbm, tpr_gbm, color='deeppink', label='GB (AUC = %f)' % auc_gbm)
plt.plot(fpr_xgb, tpr_xgb, color='navy', label='XG (AUC = %f)' % auc_xgb)
plt.xlabel('Taxa de falsos positivos') # False positive rate
plt.ylabel('Taxa de verdadeiros positivos') # True positive rate
plt.title('Comparação das curvas ROC dos classificadores') # Drug Response Prediction - ROC Curve
plt.legend(loc='lower right')
# save plot as image 
plt.savefig('./figures/roc-curves/genetic-models-comparison-roc-curves.pdf', dpi=300)
plt.show()


# In[ ]:



