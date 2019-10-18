from feature_extraction.training_data import prepate_datasets

path_train = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO_TEST\train"
path_val = r"C:\Users\esteb\Documents\CAD_PROJECT\DERMO_TEST\val"

prepate_datasets(path_train, "train.csv")
prepate_datasets(path_val, "val.csv")