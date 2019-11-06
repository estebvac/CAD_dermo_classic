from feature_extraction.training_data import *

path_train = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO\train"
path_val = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO\val"

segmentation_alg = "ws"  # "ws, ls  or None"

prepare_dataset(path_train, "train_no_segm", segm_alg= None)
