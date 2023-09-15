# Deep learning-based optical coherence tomography image analysis of human brain cancer
*Nathan Wang, Cheng-Yu Lee, Hyeon-Cheol Park, David W. Nauen, Kaisorn L. Chaichana, Alfredo Quinones-Hinojosa, Chetan Bettegowda, and Xingde Li, "Deep learning-based optical coherence tomography image analysis of human brain cancer," Biomed. Opt. Express 14, 81-88 (2023) [read here](https://opg.optica.org/boe/fulltext.cfm?uri=boe-14-1-81&id=522789)*

To request our dataset, please contact Prof. Xingde Li at xingde@jhu.edu. \
For other questions, please contact Nathan Wang at swang279@jhu.edu.

## Model Architecture

![](manuscript/figure_2.png)

**Fig. 2.** Ensemble learning architecture integrating B-frame slice pixels and texture features
for OCT cancer tissue analysis. The dimension of each layer is indicated, where the two
dimensions outside each layer are the feature map shape and the number inside each layer is
the number of maps/filters. The B-frame Slices CNN and Texture CNN are trained separately.
Once they have converged, they are used to create embeddings of their respective inputs,
which are linked to a fully connected network that makes the final classification.

## Quick Setup

```
conda create --name my_env pip python=3.9
conda activate my_env
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
cd path\to\brain_cancer_oct
pip install -r requirements.txt
```

Note: As of 09/13/23, only TensorFlow version <2.11, which requires Python <=3.9, is GPU-enabled and requires. Please refer to the [official instructions](https://www.tensorflow.org/install/pip) for up to date compatability details.

## Content Description

- **[manuscript/](manuscript):** original figures and iterations of the published manuscript
- **[margin_data/](margin_data):\*** OCT scan of tumor margin tissues (mixed cancer and non-cancer regions) 
- **[project_scripts/](project_scripts):** contains functions that are imported into the Jupyter notebooks
    - **data_loading.py:**
    - **neural_networks.py:**
    - pre-processing.py
    - widgets.py
- **[saved_models/](saved_models)**
    - **final_models/**
        - bframe_cnn/
        - texture_cnn/
        - ensemble_mlp/
    - **models_history/:**
- **[testing_data/](testing_data):\*** OCT scan data from _ patients
    - cancer/
    - non_cancer/
- **[training_data/](training_data):\*** OCT scan data from _ patients
    - cancer/
    - non_cancer/
- **analyze_neural_networks.ipynb**
- **train_neural_networks.ipynb**

**\*** Please request these data from Prof. Xingde Li at xingde@jhu.edu

## 