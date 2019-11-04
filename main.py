from feature_extraction.training_data import prepate_datasets

path_train = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\train"
path_val = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\val"

segmentation_alg = "ws" #"ws"

prepate_datasets(path_train, "train_with_HoG.csv", segm_alg = segmentation_alg)
#prepate_datasets(path_val, "val_with_HoG.csv", segm_alg = segmentation_alg)