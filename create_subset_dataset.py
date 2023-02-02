import numpy as np
import os
from util import get_distance
from glob import glob
from tqdm import tqdm
import faiss


OG_dataset_path = 'datasets/pitts30k/images/train/database/'
#OG_dataset_path = 'datasets/st_lucia/images/test/database/'

def max_distance_check(OG_dataset_list, new_dataset_list):

    new_coords_list = []

    for filename in new_dataset_list:

        new_coords = np.array([(filename.split("@")[1], filename.split("@")[2])]).astype(np.float32)
        #print(new_coords.shape)
        #print('coords: {}'.format(new_coords))

        new_coords_list.append(new_coords[0])


    faiss_index = faiss.IndexFlatL2(2)
    res = faiss.StandardGpuResources()
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    min_distances = []
    old_coords_list = []
    print('comparing distances')
    for filename in tqdm(OG_dataset_list, ncols=100):

        old_coords = np.array([(filename.split("@")[1], filename.split("@")[2])]).astype(np.float32)
        #print(old_coords)
        old_coords_list.append(old_coords[0])



    old_coords_list = np.asarray(old_coords_list)
    new_coords_list = np.asarray(new_coords_list)
    print(old_coords_list.shape)
    print(np.amin(old_coords_list))
    print(np.amin(new_coords_list))
    min_value = min(np.amin(old_coords_list), np.amin(new_coords_list))
    # if np.amin(old_coords_list) < np.amin(new_coords_list):
    #     min_value = min(old_coords_list)
    #
    # if np.amin(new_coords_list) < np.amin(old_coords_list):
    #     min_value = min(new_coords_list)

    print('min value is: {}'.format(min_value))

    #old_coords_list = old_coords_list - min_value
    #new_coords_list = new_coords_list - min_value

    print(new_coords_list.shape)
    print(old_coords_list[100:200])
    faiss_index.add(new_coords_list)
    D, I = faiss_index.search(old_coords_list, 1)

    print(D.shape)
    print(I.shape)
    print(D[0])
    print(I[0])
        # distances = []
        # for i in range(len(new_coords_list)):
        #     #print(coords[0][0])
        #     #print(new_coords_list[i][0][0])
        #     distance = get_distance(coords[0], new_coords_list[i][0])
        #     distances.append(distance)

        # min_distances.append(min(distances))
    #min_distances.sort(reverse=True)
    #D[::-1].sort(reverse=True)
    D = np.sort(D)[::-1]
    print(D)
    print(np.count_nonzero(I==3))
    print(np.count_nonzero(D==0))
    new_D = D[np.nonzero(D)]

    return new_D


def get_filename_lists(OG_dataset_path, resize_factor):
    print('getting filenames')
    OG_dataset_list = sorted(glob(os.path.join(OG_dataset_path, '*.jpg')))
    new_dataset_list = []

    for i in range(len(OG_dataset_list)):
        if i % resize_factor == 0:
            new_dataset_list.append(OG_dataset_list[i])
    return OG_dataset_list, new_dataset_list


OG_dataset_list, new_dataset_list = get_filename_lists(OG_dataset_path, 10000)

min_distances = max_distance_check(OG_dataset_list, new_dataset_list)


print('Maximum minimal distances: {}'.format(min_distances[:]))

print('Maximum minimal distances: {}'.format(np.amax(min_distances)))
