from pydub import AudioSegment
import soundfile
import pandas as pd
import csv
import re

def read_txt(file_path):
	'''
	Helper function to read indexes of train, valid and test dataset
	'''
    res = []
    f = open(file_path, "r")
    lines = f.readlines()
    for line in lines:
      number = int(line.strip())-1
      res.append(number)
    return res

def write_tsv(file_name, index_list):
	'''
	Helper function to write tsv files containing file path
	'''
    with open(file_name, "w") as f:
        print('~/dl_project/', file=f)

        for index in index_list:
            wav_path = 'data/'+str(index)+'.wav'
            wav_full_path = '~/dl_project/data/'+str(index)+'.wav'
            frames = soundfile.info(wav_full_path).frames
            print("{}\t{}".format(wav_path, frames), file=f)
    return

# Separate the whole wav file into small clips based on given sub.csv
csv_file = '~/dl_project/knnw/knnw_en_sub.csv'
wav_file = '~/dl_project/knnw/knnw_en_mono.wav'
trans_file = '~/dl_project/trans.txt'

orig_wav = AudioSegment.from_wav(wav_file)
with open(csv_file, mode='r') as f, open(trans_file, mode='w') as trans_f:
    csv_reader = csv.reader(f, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        id, start_time, end_time, text = row
        wav_file = '~/dl_project/data/'+str(id)+'.wav'
        new_wav = orig_wav[int(start_time):int(end_time)]
        new_wav = new_wav.set_frame_rate(16000)
        new_wav.export(wav_file, format="wav")
        print("{} {}".format(str(id), text), file=trans_f)

train_file = '~/dl_project/train.txt'
val_file = '~/dl_project/val.txt'
test_file = '~/dl_project/test.txt'

train_idx = read_txt(train_file)
val_idx = read_txt(val_file)
test_idx = read_txt(test_file)

train_tsv = '~/dl_project/train.tsv'
val_tsv = '~/dl_project/valid.tsv'
test_tsv = '~/dl_project/test.tsv'

write_tsv(train_tsv, train_idx)
write_tsv(val_tsv, val_idx)
write_tsv(test_tsv, test_idx)
