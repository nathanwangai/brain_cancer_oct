import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix

# log transform and median filtering
def process_bframe(file):
    B_frame = pd.read_csv(file, header=None).to_numpy() + 1
    B_frame = np.log(B_frame) # all values > 0
    B_frame = cv2.medianBlur(B_frame.astype("float32"), 3)

    return B_frame

# returns list of B-frame slices evenly sampled from left to right
def slice_bframe(B_frame, slice_width, step):
    slice_list = []
    num_possible_slices = B_frame.shape[1] // slice_width

    for i in range(0, num_possible_slices, step):
        slice = B_frame[:, i*slice_width:(i+1)*slice_width]
        slice_list.append(slice)

    return slice_list

# returns surface depth in given B-frame slice
def extract_edge(slice):
    profile = np.sum(slice[100:2000, :], axis=1)
    avg_profile = np.mean(np.reshape(profile, (95, 20)), axis=1)
    frac_max = 0.95 * np.amax(avg_profile)

    for i in range(len(avg_profile)-1):
        prior = avg_profile[i]
        next = avg_profile[i+1]
        
        if((prior<frac_max) and (next>frac_max)):
            return i*20 + 100

def convert_to_texture(trunc_slice):
    min_val, max_val = 4.0141471, 13.925418 
    rescale = np.interp(trunc_slice, (min_val, max_val), (0, 99)).astype(int)
    texture = graycomatrix(rescale, [15], [0], levels=100)
    texture = np.reshape(texture, (100, 100, 1))
    texture = np.average(texture, axis=2)

    return texture.astype("float32")

# min-max normalization
def normalize(array):
    max = np.amax(array)
    min = np.amin(array)

    return (array-min) / (max-min)

if __name__ == "__main__":
	pass # do debugging here