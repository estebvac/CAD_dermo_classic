import pandas as pd
import numpy as np
import re
import seaborn as sns # for intractve graphs
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from os import system

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt # to plot graph
import datetime # to dela with date and time
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings

#Function for computing the sensitivity
def get_sensitivity(model, test_features, test_labels):
    predicted = model.predict(test_features)
    cm = confusion_matrix(test_labels,predicted)
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
    return sensitivity

def trainCascadeRandomForestClassifier(d_ntree, d_mtry, NP, parameters, st, training_features_tp, training_features_ntp, fullX, fullY, fullValidX, fullValidY, max_num_layers):
  #Implement RF
  k = 0
  current_sensitivity = 0
  dropped = 1 #value 1 to ensure at least one iteration
  layers = [] #List to put every layer of the cascade
  layers_default = []
  while(dropped>0 and training_features_ntp.shape[0]>=NP and k<=max_num_layers):
    k=k+1
    #First, create RF with default parameters (as described in the paper)
    d_rf = RandomForestClassifier(n_estimators=d_ntree, max_features=d_mtry, random_state=0)
    d_rf.fit(fullX, fullY)
    d_probs = d_rf.predict_proba(training_features_ntp) #Get probabilities for each negative sample
    prob_sort_indexes = np.argsort(d_probs[:,1]) #Sort probabilities of being positive in ascending order
    neg_next_training = training_features_ntp[prob_sort_indexes[:NP], :] #These are the negative samples for the balanced set
    neg_next_training_remaining = training_features_ntp[prob_sort_indexes[NP:], :] #Remaining negatives (most mass-like candidates)
    current_sensitivity = 0
    #Now start grid search
    while(current_sensitivity < st):
      xx = np.concatenate((training_features_tp, neg_next_training))
      yy = np.concatenate((np.ones((training_features_tp.shape[0]), dtype=np.uint32), np.zeros((neg_next_training.shape[0]), dtype=np.uint32)))
      for ntree_sub in parameters['n_estimators']:
        for mtry_sub in parameters['max_features']:
          _ = system('cls')
          print("********************************************")
          print("Current layer index: ", str(k))
          print("Amount of FP in current layer: ", str(training_features_ntp.shape[0]))
          print("********************************************")
          print("Current parameters of the search grid: ")
          print("n_estimators: ", ntree_sub)
          print("max_features: ", mtry_sub)
          print("Num TP: ", training_features_tp.shape[0])
          print("Num NTP: ", neg_next_training.shape[0])
          rf = RandomForestClassifier(n_estimators = ntree_sub, max_features=mtry_sub, random_state =0) #Create base model
          rf.fit(xx, yy)
          current_sensitivity = get_sensitivity(rf, fullValidX, fullValidY)
          print("Sensitivity of current grid: ", str(current_sensitivity))
          if(current_sensitivity>=st): 
            break
        if(current_sensitivity>=st):
          break
      if(current_sensitivity < st): #If sensitivity could not be reached and number of negatives still higher than NP
        #10% of negatives goes to remaining and loop again
        print("***Removing 10 percent of the negative samples***")
        num_samples_neg_next_training = neg_next_training.shape[0]
        pivot = int(num_samples_neg_next_training*0.9)
        temp = neg_next_training[pivot:, :]
        neg_next_training = neg_next_training[:pivot,:]
        neg_next_training_remaining = np.concatenate((neg_next_training_remaining, temp))
    print("Sensitivity threshold reached!!!!!!!!!!!!!!!!!!!!!")
    #After sensitivity threshold is reached, classify remaining and discard some of them   
    predicted_after_st = rf.predict(neg_next_training_remaining)
    dropped = len(np.where(predicted_after_st==0)[0])
    print(str(dropped), " samples are dropped in the current layer")
    to_keep = np.where(predicted_after_st==1)[0]
    training_features_ntp = neg_next_training_remaining[to_keep]
    layers.append(rf)
    layers_default.append(d_rf)
    fullX = np.concatenate((training_features_tp,training_features_ntp))
    fullY = np.concatenate((np.ones((training_features_tp.shape[0]), dtype=np.uint32), np.zeros((training_features_ntp.shape[0]), dtype=np.uint32)))
  return layers



def trainCascadeRandomForestClassifierFaster(d_ntree, d_mtry, NP, parameters, st, training_features_tp, training_features_ntp, fullX, fullY, fullValidX, fullValidY, max_num_layers, current_fold):
  #Implement RF
  k = 0
  current_sensitivity = 0
  dropped = 1 #value 1 to ensure at least one iteration
  layers = [] #List to put every layer of the cascade
  layers_default = []
  while(dropped>0 and training_features_ntp.shape[0]>=NP and k<=max_num_layers):
    k=k+1
    #First, create RF with default parameters (as described in the paper)
    d_rf = RandomForestClassifier(n_estimators=d_ntree, max_features=d_mtry, random_state=0)
    d_rf.fit(fullX, fullY)
    d_probs = d_rf.predict_proba(training_features_ntp) #Get probabilities for each negative sample
    prob_sort_indexes = np.argsort(d_probs[:,1]) #Sort probabilities of being positive in ascending order
    neg_next_training = training_features_ntp[prob_sort_indexes[:NP], :] #These are the negative samples for the balanced set
    neg_next_training_remaining = training_features_ntp[prob_sort_indexes[NP:], :] #Remaining negatives (most mass-like candidates)
    current_sensitivity = 0
    #Now start grid search
    speedup = False
    while(current_sensitivity < st):
      xx = np.concatenate((training_features_tp, neg_next_training))
      yy = np.concatenate((np.ones((training_features_tp.shape[0]), dtype=np.uint32), np.zeros((neg_next_training.shape[0]), dtype=np.uint32)))
      for ntree_sub in parameters['n_estimators']:
        for mtry_sub in parameters['max_features']:
          _ = system('cls')
          print("********************************************")
          print("Current fold:", str(current_fold))
          print("Current layer index: ", str(k))
          print("Amount of FP in current layer: ", str(training_features_ntp.shape[0]))
          print("********************************************")
          print("Current parameters of the search grid: ")
          print("n_estimators: ", ntree_sub)
          print("max_features: ", mtry_sub)
          print("Num TP: ", training_features_tp.shape[0])
          print("Num NTP: ", neg_next_training.shape[0])
          rf = RandomForestClassifier(n_estimators = ntree_sub, max_features=mtry_sub, random_state =0) #Create base model
          rf.fit(xx, yy)
          current_sensitivity = get_sensitivity(rf, fullValidX, fullValidY)
          if(np.abs(st - current_sensitivity) > 0.2):
            speedup = True
          else:
            speedup = False
          print("Sensitivity of current grid: ", str(current_sensitivity))
          if(current_sensitivity>=st or speedup): 
            break
        if(current_sensitivity>=st or speedup):
          break
      if(current_sensitivity < st or speedup): #If sensitivity could not be reached and number of negatives still higher than NP
        #10% of negatives goes to remaining and loop again
        print("***Removing 10 percent of the negative samples***")
        num_samples_neg_next_training = neg_next_training.shape[0]
        pivot = int(num_samples_neg_next_training*0.9)
        temp = neg_next_training[pivot:, :]
        neg_next_training = neg_next_training[:pivot,:]
        neg_next_training_remaining = np.concatenate((neg_next_training_remaining, temp))
        speedup = False
        current_sensitivity = 0
    print("Sensitivity threshold reached!!!!!!!!!!!!!!!!!!!!!")
    #After sensitivity threshold is reached, classify remaining and discard some of them   
    predicted_after_st = rf.predict(neg_next_training_remaining)
    dropped = len(np.where(predicted_after_st==0)[0])
    print(str(dropped), " samples are dropped in the current layer")
    to_keep = np.where(predicted_after_st==1)[0]
    training_features_ntp = neg_next_training_remaining[to_keep]
    layers.append(rf)
    layers_default.append(d_rf)
    fullX = np.concatenate((training_features_tp,training_features_ntp))
    fullY = np.concatenate((np.ones((training_features_tp.shape[0]), dtype=np.uint32), np.zeros((training_features_ntp.shape[0]), dtype=np.uint32)))
  return layers



def applyCascade(layers, test_features):
  origninal_rows = test_features.shape[0]
  to_keep = []
  #From first layer:
  first_pred = layers[0].predict(test_features)
  indexes_first = np.where(first_pred==1)[0]
  test_features = test_features[indexes_first]
  to_keep.append(indexes_first)
  for currLayer in layers[1:]:
    predictions = currLayer.predict(test_features)
    to_discard_indexes = np.where(predictions==0)[0]
    to_keep_indexes = np.where(predictions==1)[0]
    indexes_first = np.delete(indexes_first, tuple(to_discard_indexes)) 
    to_keep.append(indexes_first)
    test_features = test_features[to_keep_indexes] 
  preds = np.zeros(origninal_rows, dtype=int)
  preds.shape
  preds[to_keep[-1]] = 1
  #len(np.where(preictions>0)[0])
  return preds


def get_probs(layers, test_features):
  num_test_rows = test_features.shape[0]
  p = [] #List for storing probabilities of passing samples to next layer
  rejected_indexes =[]
  rejected_probs = []
  to_keep = []
  #From first layer:
  first_pred = layers[0].predict(test_features)
  first_probs = layers[0].predict_proba(test_features)
  indexes_first = np.where(first_pred==1)[0]
  p.append(len(indexes_first)/test_features.shape[0])
  rejected_indexes.append(np.where(first_pred==0)[0])
  rejected_probs.append(first_probs[np.where(first_pred==0)[0]])
  test_features = test_features[indexes_first]
  to_keep.append(indexes_first)
  for currLayer in layers[1:]:
    predictions = currLayer.predict(test_features)
    probs = currLayer.predict_proba(test_features)
    to_discard_indexes = np.where(predictions==0)[0]
    rejected_indexes.append(indexes_first[to_discard_indexes]) #*******************************
    rejected_probs.append(probs[to_discard_indexes]) #*******************************
    to_keep_indexes = np.where(predictions==1)[0]
    p.append(len(to_keep_indexes)/test_features.shape[0]) #*******************************
    indexes_first = np.delete(indexes_first, tuple(to_discard_indexes)) 
    to_keep.append(indexes_first)
    test_features = test_features[to_keep_indexes] 
  preds = np.zeros(num_test_rows, dtype=int)
  preds.shape
  preds[to_keep[-1]] = 1
  
  #Temporal variables
  indexes_tp = to_keep[-1]
  probs_tp = probs[to_keep_indexes,1]
  
  all_probs = np.zeros((num_test_rows,))
  #First, recover the probabilities of the last layer
  all_probs[indexes_tp] = probs_tp
  all_probs[rejected_indexes[-1]] = rejected_probs[-1][:,1]

  #Now recover the remaining probs
  for i in range(len(rejected_indexes)-1):
    for j in range(i,len(rejected_indexes)):
      rejected_probs[i][:,1] = np.multiply(rejected_probs[i][:,1],p[j])
  for i in range(len(rejected_indexes)-1):
    all_probs[rejected_indexes[i]] = rejected_probs[i][:,1]
    
  return all_probs


def subdivide_dataset_k(training_features, training_labels, perc, negative_multiplier):
    indexes_true_positive = np.where(training_labels>0)[0]
    indexes_negatives = np.where(training_labels==0)[0]
    training_features_tp = training_features[indexes_true_positive,:]
    training_features_ntp = training_features[indexes_negatives,:]
    samples_for_validation = int(training_features_tp.shape[0]*perc) #20% of the positives for internal validation of the classifier
    negative_multiplier = 1
    validation_features_tp = training_features_tp[-samples_for_validation:,:]
    validation_features_ntp = training_features_ntp[-negative_multiplier*samples_for_validation:,:]

    training_features_tp = training_features_tp[:-samples_for_validation,:]
    training_features_ntp = training_features_ntp[:-negative_multiplier*samples_for_validation,:]
    return training_features_tp, training_features_ntp, validation_features_tp, validation_features_ntp