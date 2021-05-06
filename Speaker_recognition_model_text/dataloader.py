import torch
import numpy as np
import torch.utils.data
import torch
from torch import nn
from torch.nn.utils.rnn import *
from clean_text import Preprocessing



class DatasetMaper(torch.utils.data.Dataset):
    def __init__(self, file_path):
        num_words = 100
        seq_len = 100
        # file_path = '/content/val_full.txt'
        pr = Preprocessing(num_words, seq_len, file_path)
        pr.load_data()
        pr.clean_text()
        pr.text_tokenization()
        pr.build_vocabulary()
        pr.word_to_idx()
        X, Y = pr.padding_sentences()

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # x = self.X[idx, :, :]
        # y = self.Y[idx, :]
        # # print('y = ', y)
        # X_lens = self.X_lens[idx]
        # Y_lens = self.Y_lens[idx]
        return self.X[idx, :], self.Y[idx]
