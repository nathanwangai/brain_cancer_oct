# Deep learning-based optical coherence tomography image analysis of human brain cancer
*Nathan Wang, Cheng-Yu Lee, Hyeon-Cheol Park, David W. Nauen, Kaisorn L. Chaichana, Alfredo Quinones-Hinojosa, Chetan Bettegowda, and Xingde Li, "Deep learning-based optical coherence tomography image analysis of human brain cancer," Biomed. Opt. Express 14, 81-88 (2023)*

https://opg.optica.org/boe/fulltext.cfm?uri=boe-14-1-81&id=522789

To request data, please contact Prof. Xingde Li at xingde@jhu.edu.
For other questions, please contact Nathan Wang at swang279@jhu.edu.

## Key Result
An ensemble deep neural network trained on B-frame "slices" is able to distinguish high grade cancer from healthy tissue with 93% sensitivity and 97% specificity in real-time.

![](https://user-images.githubusercontent.com/98730743/201268404-e86ce5d4-6a04-4aa2-b464-0d16b0a71cdb.png)

## Quick Setup
```
conda create --name my_env pip
conda activate my_env
cd path\to\brain_cancer_oct
pip install -r requirements.txt
```
FYI to install TensorFlow with GPU support:
```
conda activate my_env
pip install 
```

## Directory Structure

- [project_scripts/](./project_scripts/)
    - data_loading.py
    - neural_networks.py
    - pre-processing.py
    - widgets.py
- saved_models/
    - final_models/
        - bframe_cnn/
    - models_history/

Download "saved_models\" and "seg_app\" and navigate into the latter folder. Run the following in the command line: "Python seg_app.py"
