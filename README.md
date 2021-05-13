# CMU-11-785-Project
Final Project for 11-785 SP2021


### Speaker Recognition Model (Audio Branch)
The speaker recognition model (audio branch) folder contains code for running audio-based speaker recognition model.
Generate data splits using ground truth texts and train model:
1. go to Speaker_recognition_model_audio/Preprocessing directory
2. Upload Preprocessing.ipynb and TIMIT_labels.npy to Google Colab
3. Upload train.txt and test.txt to /content/gdrive/MyDrive/knnw folder of Google drive using same Google account
4. Upload id_speaker.txt, knnw_en_sub.csv and knnw_en_mono.wav to /content/gdrive/MyDrive/knnw folder of Google drive using same Google account
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

### Speech Recognition Models
The speech recognition model folder contains code for running two speech recognition models (DeepSpeech2 and Fairseq S2T).
Before training the model, we need to generate data splits:
1. First you need to create a folder for KNNW data, and move knnw_en_sub.csv and knnw_en_mono.wav into the folder.
```
cd Speech_recognition/data_preprocessing
mkdir knnw
```
2. Then you need to run data_preprocessing.py to generate wav files and tsv files for training.
```
cd Speech_recognition
python data_preprocessing.py
```
3. To train DeepSpeech2 model,
```
cd Speech_recognition/deepspeech2
pip3 install -r requirements.txt
python deep_speech.py
```
4. To train Fairseq-S2T model,
```
cd Speech_recognition/fairseq-s2t
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $MODEL_PATH --max-epoch 80 --task speech_recognition --arch vggtransformer_2 --optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 --clip-norm 10.0  --max-tokens 5000 --log-format json --log-interval 1 --criterion cross_entropy_acc --user-dir examples/speech_recognition/
```
5. To evaluate the result on averaged Levenshtein distance,
```
cd Speech_recognition
python evaluation.py
```
6. (Optional) To use Google API for speech recognition (as reference):
```
cd Speech_recognition
python GoogleAPI-speech_recognition
```

### Multimodal Fusion
Once we have speaker classification predictions from audio and text branches, we can run multimodal fusion by:
```
cd Speaker_recognition_model_text
python fusion.py
```
### Citation
The code is based on this [Github repo](https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch) and [Github repo](https://github.com/mravanelli/SincNet)
