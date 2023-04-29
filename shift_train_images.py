import shutil
import os
from tqdm import tqdm
image_root = '/data/mrigankr/mml/images/train/'
copy_path = '/data/mrigankr/mml/train'
# get list of all files in directory
file_list = os.listdir(image_root)
# iterate over each file
for file_name in tqdm(file_list):
    # create the full input path and read the file
    full_file_name = os.path.join(image_root, file_name)
    list_images = os.listdir(full_file_name)
    for image in list_images:
        shutil.copy(os.path.join(full_file_name, image), copy_path)
        # print(os.path.join(full_file_name, image))