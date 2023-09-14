import cv2
import numpy as np
from skimage.feature import graycomatrix

'''

'''

# ====================================================================================================

'''
log transform and median 
'''
def process_bframe(file):
    B_frame = np.log(np.load(file) + 1)
    B_frame = cv2.medianBlur(B_frame.astype('float32'), 3)

    return B_frame

'''
returns list of B-frame slices sampled uniformly from left to right
'''
def slice_bframe(B_frame, slice_width, step):
    slice_list = []
    num_possible_slices = B_frame.shape[1] // slice_width

    for i in range(0, num_possible_slices, step):
        slice = B_frame[:, i*slice_width:(i+1)*slice_width]
        slice_list.append(slice)

    return slice_list

# ====================================================================================================

'''
min-max normalization
array:
'''
def normalize(array):
    max = np.max(array)
    min = np.min(array)

    return (array-min) / (max-min)

# ====================================================================================================

def extract_edge(slice):
    profile = np.sum(slice[100:2000, :], axis=1)  # 1900 pixels deep
    avg_profile = np.mean(np.reshape(profile, (95, 20)), axis=1)  # 95 pixels deep
    frac_max = 0.95 * np.max(avg_profile)
    transition_indices = np.where((avg_profile[:-1] < frac_max) & (avg_profile[1:] > frac_max))[0]
    
    return (transition_indices[0] * 20) + 100

# ====================================================================================================

'''
trunc_slice:
min_val:
max_val:
'''
def convert_to_texture(trunc_slice, min_val, max_val): 
    rescale = np.interp(trunc_slice, (min_val, max_val), (0, 99)).astype(int)
    texture = graycomatrix(rescale, [15], [0], levels=100)
    texture = np.reshape(texture, (100, 100, 1))

    return texture.astype("float32")