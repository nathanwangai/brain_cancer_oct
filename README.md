# Deep learning-based optical coherence tomography image analysis of human brain cancer
*Nathan Wang, Cheng-Yu Lee, Hyeon-Cheol Park, David W. Nauen, Kaisorn L. Chaichana, Alfredo Quinones-Hinojosa, Chetan Bettegowda, and Xingde Li, "Deep learning-based optical coherence tomography image analysis of human brain cancer," Biomed. Opt. Express 14, 81-88 (2023)*

https://opg.optica.org/boe/fulltext.cfm?uri=boe-14-1-81&id=522789

## Key result
An ensemble deep neural network trained on B-frame "slices" is able to distinguish high grade cancer from healthy tissue with 93% sensitivity and 97% specificity in real-time.

![](https://user-images.githubusercontent.com/98730743/201268404-e86ce5d4-6a04-4aa2-b464-0d16b0a71cdb.png)

## Contents

- saved_models\: all trained Tensorflow models
- seg_app\: Python app with interface for displaying segmentation result
- cancer_classifier.ipynb: Python notebook with all experiments (not thoroughly documented)

Download "saved_models\" and "seg_app\" and navigate into the latter folder. Run the following in the command line: "Python seg_app.py"
