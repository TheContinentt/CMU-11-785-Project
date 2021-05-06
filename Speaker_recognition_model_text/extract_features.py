import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from keras_bert import extract_embeddings
import numpy as np
model_path = '/content/uncased_L-12_H-768_A-12'

csv_file = '/content/knnw_en_sub.csv'
cnt = 0
with open(csv_file, 'r', encoding="utf8") as f:
    for line in f:
        cnt += 1
        print(cnt)
        line = line.strip().split(';')
        name = line[0]
        text = line[-1]
        new_embeddings = extract_embeddings(model_path, [text])
        np.save('./feature/{}.npy'.format(str(name)), new_embeddings)
