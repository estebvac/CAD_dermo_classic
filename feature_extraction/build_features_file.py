import pandas as pd
import os


def read_images(general_path):
    """
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    """

    #  Read the dataset
    data_tuple = []
    for folder in sorted(os.listdir(general_path)):
        for file in sorted(os.listdir(general_path + '/' + folder)):
            full_path = general_path + '/' + folder + '/' + file
            data_tuple.append((full_path, folder))

    images_df = pd.DataFrame(data_tuple, columns=['File', 'Class'])
    return images_df
