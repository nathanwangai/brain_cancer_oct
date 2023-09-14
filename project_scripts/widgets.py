import os
import ipywidgets as widgets
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from project_scripts.pre_processing import process_bframe, slice_bframe, normalize, extract_edge, convert_to_texture

'''
Contains ipython widgets to help visualize data
'''

# ====================================================================================================

'''
Enables full exploration of dataset
'''
def explore_dataset_widget(plotting_func):
    # create widgets
    class_dir = widgets.Dropdown(options = ['testing_data\\cancer', 'testing_data\\non_cancer'])
    sample_dir = widgets.Dropdown(options = os.listdir(class_dir.value))
    max_frame_idx = len(os.listdir(os.path.join(class_dir.value, sample_dir.value)))-1
    slice_idx = widgets.IntSlider(value=0, min=0, max=6)
    frame_idx = widgets.IntSlider(value=0, min=0, max=max_frame_idx)

    def update_sample_dir(change):
        sample_dir.options = os.listdir(change['new'])
        sample_dir.value = sample_dir.options[0]

    def update_max_frame_idx(change):
        dir1, dir2 = (class_dir.value, change['new'])
        if change['owner'] == class_dir: dir1, dir2 = (change['new'], sample_dir.value)

        frame_idx.max = len(os.listdir(os.path.join(dir1, dir2))) - 1
    
    # run callback function when widgets change
    class_dir.observe(update_sample_dir, names='value')
    class_dir.observe(update_max_frame_idx, names='value')
    sample_dir.observe(update_max_frame_idx, names='value')

    return widgets.interactive(plotting_func, 
                               slice_idx=slice_idx, 
                               frame_idx=frame_idx, 
                               class_dir=class_dir, 
                               sample_dir=sample_dir)

# ====================================================================================================

'''
Plots bframe, slice, and texture side-by-side

slice_idx: 
frame_idx: 
class_dir: 
sample_dir: 
'''
def plot_data(slice_idx, frame_idx, class_dir, sample_dir):
    dir_path = os.path.join(class_dir, sample_dir)
    fname = os.listdir(dir_path)[frame_idx]

    # pre-proecessing
    B_frame = process_bframe(os.path.join(dir_path, fname)) # step 1: log(raw_bframe + 1)
    slices = slice_bframe(B_frame, slice_width=100, step=3) # step 2: sample bframe into slices
    B_frame_slice = slices[slice_idx]
    edge_depth = extract_edge(B_frame_slice) 
    B_frame_slice = B_frame_slice[edge_depth:edge_depth+200] # step 3a: truncate slice to 200 pixels below edge
    slice_texture = normalize(convert_to_texture(B_frame_slice, 4.11, 13.92)) # step 3b: convert slice to texture

    # full B-frame
    plt.subplots(1,3, figsize=(5,15))
    plt.subplot(1,3,1)
    plt.title(f'Edge depth: {edge_depth}')
    plt.axis('off')
    plt.imshow(B_frame, cmap='gray')
    plt.axhline(y=edge_depth, color='r', linestyle='-', linewidth=0.5)

    # B-frame slice
    plt.subplot(1,3,2)
    plt.title('Slice #: ' + str(slice_idx+1))
    plt.axis('off')
    plt.imshow(B_frame_slice, cmap='gray')

    # Texture of B-frame slice
    plt.subplot(1,3,3)
    plt.title('Texture')
    plt.imshow(slice_texture, cmap='gray')
    plt.axis('off')
    plt.show()

# ====================================================================================================

'''
Plays a video of the dataset

X_data:
Y_data:
'''
def dataset_movie_widget(X_data, Y_data):
    X_data, Y_data = shuffle(X_data, Y_data)
    num_training_ex = X_data.shape[0] - 1
    play_idx = widgets.Play(max=num_training_ex, interval=500)
    slider_idx = widgets.IntSlider(max=num_training_ex)
    widgets.jslink((play_idx, 'value'), (slider_idx, 'value'))

    def my_func(idx, slider_idx):
        plt.figure(figsize=(3,3))
        plt.axis('off')
        plt.imshow(X_data[idx], cmap='gray')
        title = 'Cancer' if Y_data[idx][0] == 1 else 'Non-cancer'
        plt.title(title)
        plt.colorbar()
        plt.show()

    return widgets.interactive(my_func, idx=play_idx, slider_idx=slider_idx)