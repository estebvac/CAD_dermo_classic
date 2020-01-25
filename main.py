"""
This is the main function of the program in case you want to run it offline. 
The program has to be run twice: once for segmentation case and once for no segmentation, generating
a total of 4 files (two csv and two npy)

After the first run please move the generated files from ~/train and ~/val to another folder so that 
the new files can be generated. For the second run, comment out lines 23 and 24 and imcomment last
two lines and run again. At the end put all 4 generated files together and run the classifiers
that are available in the google colab notebook.


"""


from feature_extraction.training_data import *

path_train = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\train"
path_val = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\val"
path_test = r"C:\MyData\MAIA\Semester3\CAD\Projects\Dermo\Data\test"

segmentation_alg = "ws"  # "ws, ls  or None"

prepare_dataset(path_train, "train_segm", segm_alg= "ws", debug=False)
prepare_dataset(path_val, "val_segm", segm_alg= "ws")

#For generating features without segmentation, please uncomment following lines
#prepare_dataset(path_train, "train_no_segm", segm_alg= None)
#prepare_dataset(path_val, "val_no_segm", segm_alg= None)
