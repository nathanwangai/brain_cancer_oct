import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

'''
Plots the training and validation accuracy of the models over epochs

title: title of the plot
log_fpath: path to the csv file containing training history
'''
def plot_training_curves(log_fpath, title):
    logs = pd.read_csv(log_fpath).drop(['loss', 'val_loss'], axis=1).set_index('epoch')
    plt.plot(logs)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(logs.columns)

'''
model_fpath: file path to the saved model
X_data:
Y_data:
'''
def model_uncertainty(model_fpath, X_data, Y_data):
    accuracy_arr = []
    tpr_arr = []
    tnr_arr = []
    loaded_model = tf.keras.models.load_model(model_fpath)
    
    for i in range(20):
        predictions = loaded_model.predict(X_data, verbose=0)[:,0] > 0.5
        confusion_mtx = tf.math.confusion_matrix(Y_data[:,0], predictions, num_classes=2)
        accuracy_arr.append(float(np.trace(confusion_mtx) / np.sum(confusion_mtx)))
        tpr_arr.append(float(confusion_mtx[0,0] / np.sum(confusion_mtx[:,0])))
        tnr_arr.append(float(confusion_mtx[1,1] / np.sum(confusion_mtx[:,1])))

    del loaded_model
    gc.collect()

    mu_acc, std_acc = np.mean(accuracy_arr), np.std(accuracy_arr)
    mu_tpr, mu_tnr =  np.mean(tpr_arr), np.mean(tnr_arr) 

    print(f'mu_acc: {mu_acc*100:.3f}%, std_acc: {std_acc*100:.3f}%, mu_sens: {mu_tpr*100:.3f}%, std: {mu_tnr*100:.3f}%')
    
    return mu_acc, std_acc, mu_tpr, mu_tnr

'''
Displays predicted cancer map of a volumetric OCT scan by evaluating one slice/texture pair at a time

X_data: 
model_fpath:
'''
def cancer_segmentation(X_data, model_fpath):
    model = tf.keras.models.load_model(model_fpath)
    seg_map = model.predict(X_data) > 0.5
    seg_map = np.reshape(seg_map[:,0], (256,20))
    plt.imshow(resize(seg_map, (20,20)), cmap='gray')
    plt.colorbar()
    plt.show()