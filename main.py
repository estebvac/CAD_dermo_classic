from feature_extraction.training_data import prepate_datasets

path_train = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO_TEST\train"
path_val = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO_TEST\val"

segmentation_alg = "ls" #"ws"

prepate_datasets(path_train, "train.csv", segm_alg = segmentation_alg)
prepate_datasets(path_val, "val.csv", segm_alg = segmentation_alg)