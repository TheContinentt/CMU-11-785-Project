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
### Speaker Recognition Model (Audio Branch)
The speaker recognition model (audio branch) folder contains code for running audio-based speaker recognition model.
Generate data splits using ground truth texts and train model:
1. Upload Preprocessing.ipynb in Speaker_recognition_model_audio folder to Google Colab
2. Connect running instance in Preprocessing.ipynb to Google Drive and run through this file.
3. Data splits are generated in Google Drive directory: /content/gdrive/MyDrive/knnw/data

Train and test data using audio-based model:
1. Upload Speaker_recognition_model_audio directory as a zip file to Google Colab with same Google account using in data splits and unzip it.
2. open terminal in Google Colab
3. 
```
python3 TIMIT_preparation.py /content/gdrive/MyDrive/knnw/data data_lists/TIMIT_all.scp
python3 speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
```
### Citation
The code is based on this [Github repo](https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch) and [Github repo](https://github.com/mravanelli/SincNet)
