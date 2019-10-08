from main_flow import flow
from feature_extraction.training_data import prepate_datasets

path = r"C:\AIA_PROJECT\tmp\\" #r"C:\AIA_PROJECT\dataset\\"
path_img = r"C:\AIA_PROJECT\dataset\images\\"
image_name = "24065584_d8205a09c8173f44_MG_L_CC_ANON.tif"

#[raw_im_Path, gt_im_path, raw_images, gt_images, false_positive_path, true_positive_path] = read_images(path)


#flow.get_rois_from_image(path, image)

prepate_datasets(path)


#flow.segment_single_image(path_img, image_name)