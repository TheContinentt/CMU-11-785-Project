# CMU-11-785-Project
Final Project for 11-785 SP2021

### Speaker Recognition Model (Text Branch)
The speaker recognition model (text branch) folder contains code for running text-based speaker recognition model.
Generate data splits using ground truth texts and train model:
```
cd Speaker_recognition_model_text
python generate_trainvaltest.py
python train.py
```
Generate data splits using FairSeq predicted texts and train model:
```
cd Speaker_recognition_model_text
python generate_trainvaltest_fairseq.py
python train.py
```
### Citation
The code is based on this [Github repo](https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch)
