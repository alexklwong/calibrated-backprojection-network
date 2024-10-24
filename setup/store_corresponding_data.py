import sys
import pathlib
import numpy as np
from numpy.core.fromnumeric import argmin
import shutil
from natsort import natsorted
from tqdm import tqdm

outer_folder = sys.argv[1]
outer_folder_path = pathlib.Path(outer_folder)

# list(pathlib.Path('your_directory').glob('*.txt'))
subFolders = list( outer_folder_path.glob('*') ) 

subFolder_images = []
subFolder_times = []
subFolder_images_lens = []

for subF in subFolders:
    corres_folder = pathlib.Path(str(subF)+'/corresponding_images')
    try:
        corres_folder.mkdir(parents=True,exist_ok=True)
    except:
        corres_folder.mkdir(parents=True)

    images_paths_list = natsorted(list( subF.glob('*jpg') ))
    pngs = natsorted([image for image in subF.glob('*png')])
    for png in pngs:
        images_paths_list.append(png)
    
    times = natsorted([int(image_path.stem) for image_path in images_paths_list])
    subFolder_images.append(images_paths_list)
    subFolder_images_lens.append(len(images_paths_list))
    subFolder_times.append(times)

min_folder = np.argmin(subFolder_images_lens)

print('Minimum images in folder: {}'.format(subFolders[min_folder].stem ))
count = 1
for iter,image_path in  tqdm (enumerate(subFolder_images[min_folder]), total=len(subFolder_images[min_folder])):
    image_time = subFolder_times[min_folder][iter]
    subFolder = image_path.parent

    new_filename = '/corresponding_images/' + str(count) + image_path.suffix
    new_file_str = str(subFolder)+new_filename
    new_file_path = pathlib.Path(new_file_str)

    if not new_file_path.exists():
        shutil.copy(image_path, new_file_path)

    for i in range(len(subFolders)):
        if i == min_folder:
            continue
        diff = [abs(subF_time - image_time) for subF_time in subFolder_times[i]] 
        min_arg = np.argmin(diff)
        subF_image_path = subFolder_images[i][min_arg]
        
        new_filename = '/corresponding_images/'+str(count) + subF_image_path.suffix
        new_file_str = str(subFolders[i])+new_filename
        new_file_path = pathlib.Path(new_file_str)
        if not new_file_path.exists():
            shutil.copy(subF_image_path, new_file_path)
    count+=1

