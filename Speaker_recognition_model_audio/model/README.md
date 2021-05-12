### Speaker Recognition Model (Audio Branch)
The speaker recognition model (audio branch) folder contains code for running audio-based speaker recognition model.
Generate data splits using ground truth texts and train model:
1. go to Speaker_recognition_model_audio/Preprocessing directory
2. Upload Preprocessing.ipynb to Google Colab
3. Upload train.txt and test.txt to /content/gdrive/MyDrive/knnw folder of Google drive using same Google account
4. Upload knnw_en_sub.csv and knnw_en_mono.wav to /content/gdrive/MyDrive/knnw folder of Google drive using same Google account
5. Connect running instance in Preprocessing.ipynb to Google Drive and run through this file.
6. Data splits are generated in Google Drive directory: /content/gdrive/MyDrive/knnw/data
7. download TIMIT_train.scp, TIMIT_test.scp and TIMIT_all.scp to Speaker_recognition_model_audio/model/data_lists folder

Train and test data using audio-based model:
1. Upload Speaker_recognition_model_audio/model directory as a zip file to Google Colab with same Google account using in data splits and unzip it.
2. open terminal in Google Colab
3. 
```
python3 TIMIT_preparation.py /content/gdrive/MyDrive/knnw/data data_lists/TIMIT_all.scp
python3 speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
```
### Citation
The code is based on this [Github repo](https://github.com/mravanelli/SincNet)
