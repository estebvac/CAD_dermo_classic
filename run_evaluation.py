import pandas as pd
import numpy as np
import re
import seaborn as sns # for intractve graphs
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from evaluation.model_evaluation import *
from sklearn.preprocessing import StandardScaler # for preprocessing the data

path = 'C:/MyData/MAIA/Semester2/AdvancedImageAnalysis/Project/dataset-20190406T200657Z-001/dataset/'

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


#test_data = pd.read_csv(r'../dataset_hog/testing.csv')
#test_metadata = pd.read_csv(r'../dataset_hog/testing_metadata.csv')
#test_metadata = recover_filename(test_metadata)
#train_data = pd.read_csv(r'../dataset_hog/training.csv')
#train_metadata = pd.read_csv(r'../dataset_hog/training_metadata.csv')
#train_metadata = recover_filename(train_metadata)

total_data = pd.read_csv(r'../dataset_hog_with_fp/training.csv')
total_metadata = pd.read_csv(r'../dataset_hog_with_fp/training_metadata.csv')
total_metadata = recover_filename(total_metadata)
total_metadata.shape

# Define the parameters for the Cross Validation
folds = 10   #@param {type:"number"}
FROC_samples = 15 #@param {type:"number"}
#train_dataframe = train_data
#train_metadata = train_metadata
index_max_layer = 4 # Means 5 layers maximum

k_froc_vals = Kfold_FROC_curve_cascadeRF(folds, FROC_samples, total_data, total_metadata, path, index_max_layer)
np.save('froc_k10_5layers.npy', k_froc_vals)
plot_FROC(k_froc_vals)

