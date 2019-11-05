from feature_extraction.training_data import prepate_datasets_no_segm

path_train = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\train"
path_val = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\val"

segmentation_alg = "ws" #"ws"

prepate_datasets_no_segm(path_train, "train_no_segm.csv")
#prepate_datasets(path_val, "val_no_segm.csv", segm_alg = segmentation_alg)