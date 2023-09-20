import gc
import os
import numpy as np
from project_scripts.neural_networks import B_frame_CNN, texture_CNN
from project_scripts.pre_processing import process_bframe, slice_bframe, normalize, extract_edge, convert_to_texture

'''

'''

# ====================================================================================================

'''

data_dir: 
step: 
'''
def load_dataset(data_dir, step=3):
    X_train = []
    Y_train = []
    max_val = -np.inf
    min_val = np.inf

    for subdir, _, files in os.walk(data_dir):
        label = [0, 1]
        if len(subdir.split('\\')) > 2 and (subdir.split('\\')[1] == 'cancer'): label = [1, 0]
        print(f'LOADING: directory=({subdir}) | label=({label}) | total_bframes=({len(files)})')

        for file in files:
            fpath = os.path.join(subdir, file)
            B_frame = process_bframe(fpath)
            slices = slice_bframe(B_frame, slice_width=100, step=step)

            for slice in slices:
                edge_depth = extract_edge(slice)
                slice = slice[edge_depth:edge_depth+200]
                X_train.append(slice)
                Y_train.append(label)

                slice_max, slice_min = np.max(slice), np.min(slice)
                max_val = max(max_val, slice_max)
                min_val = min(min_val, slice_min)
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    return X_train, Y_train, max_val, min_val

# ====================================================================================================

'''

X_slices: 
'''
def slices_to_textures(X_slices):
    X_texture = []
    for sample in X_slices:
        X_texture.append(normalize(convert_to_texture(sample, 4.11, 13.92)))
    return np.array(X_texture)

# ====================================================================================================

'''
X_slices:
X_textures:
'''
def dataset_to_embeddings(X_slices, X_textures):
    model1 = B_frame_CNN(3, 'relu', 'same')
    model1.load_weights('saved_models\\final_models\\bframe_cnn').expect_partial()
    bframe_embedding_model = model1.get_embedding()

    model2 = texture_CNN(3, 'relu', 'same')
    model2.load_weights('saved_models\\final_models\\texture_cnn').expect_partial()
    texture_embedding_model = model2.get_embedding()

    bframe_embeddings = bframe_embedding_model.predict(X_slices)
    texture_embeddings = texture_embedding_model.predict(X_textures)

    del bframe_embedding_model
    del texture_embedding_model
    gc.collect()

    return np.concatenate((bframe_embeddings, texture_embeddings), axis=1)