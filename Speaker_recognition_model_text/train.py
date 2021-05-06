import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from dataloader import DatasetMaper
from model import TextClassifier

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import lr_scheduler

import numpy as np


def train(model, params):

    # Initialize dataset maper
    # train_file_path = '/content/train_full_fair.txt'
    # test_file_path = '/content/test_full_fair.txt'
    # val_file_path = '/content/val_full_fair.txt'
    train_file_path = '/content/train_full.txt'
    test_file_path = '/content/test_full.txt'
    val_file_path = '/content/val_full.txt'
    train = DatasetMaper(train_file_path)
    test = DatasetMaper(test_file_path)
    val = DatasetMaper(val_file_path)

    # Initialize loaders
    loader_train = DataLoader(train, batch_size=params['batch_size'])
    loader_test = DataLoader(test, batch_size=params['batch_size'])
    loader_val = DataLoader(val, batch_size=params['batch_size'])

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    lr_scheduler_v = ReduceLROnPlateau(optimizer, mode='max', factor=0.01, patience=3, verbose=True)
    # lr_scheduler_v = lr_scheduler.CosineAnnealingLR(optimizer,T_max =20, eta_min=0.01)

    # Starts training phase
    epoch_lst = []
    loss_lst = []
    accu_lst = []
    for epoch in range(params['epochs']):
        epoch_lst.append(int(epoch))
        # Set model in training model
        model.train()
        predictions = []
        # Starts batch training
        for x_batch, y_batch in loader_train:
            y_batch = y_batch.type(torch.FloatTensor)

            # Feed the model
            y_pred = model(x_batch) # B, 99

            # Loss calculation
            loss = torch.nn.CrossEntropyLoss()(y_pred, y_batch.long())

            # Clean gradientes
            optimizer.zero_grad()

            # Gradients calculation
            loss.backward()

            # Gradients update
            optimizer.step()

            # Save predictions
            predictions += list(y_pred.detach().numpy())
        loss_lst.append(loss.item())
        print('loss = ', loss.item())
        # Evaluation phase


        # Metrics calculation
        model.eval()
        correct = 0
        ttl = 0
        # test_predictions = Run.evaluation(model, loader_test)
        for x_batch, y_batch in loader_test:
            outputs = model(x_batch)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(y_batch, predictions):
                if label == prediction:
                    correct += 1
                ttl += 1
        accu_lst.append(correct/ttl)
        print('test_accu = ', correct / ttl)

        # test set
        correct = 0
        ttl = 0
        predictions_lst = []
        score_lst = []
        # test_predictions = Run.evaluation(model, loader_test)
        for x_batch, y_batch in loader_val:
            outputs = model(x_batch)
            scores, predictions = torch.max(outputs, 1)
            scores = [x.detach().numpy() for x in scores]
            score_lst.extend(scores)

            for label, prediction in zip(y_batch, predictions):
                predictions_lst.append(prediction)
                if label == prediction:
                    correct += 1
                ttl += 1
        with open('/content/pred_{}.txt'.format(str(epoch)), 'w') as out_f:
            for idx, pred in enumerate(predictions_lst):
                print(pred.numpy(), score_lst[idx], file=out_f)
        # accu_lst.append(correct / ttl)
        print('val_accu = ', correct / ttl)
        lr_scheduler_v.step(correct / ttl)

        # plt.plot(epoch_lst, loss_lst, 'o', color='black')
        plt.close()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epoch_lst, loss_lst, '-ok', color='black')
        plt.savefig('/content/epoch_loss.png')
        plt.close()

        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(epoch_lst, accu_lst, '-ok', color='black')
        plt.savefig('/content/epoch_accu.png')
        plt.close()




params = {
    'seq_len': 100,
    'num_words': 100,
    'embedding_size': 100,
    'out_size': 128,
    'stride': 1,
    'epochs': 20,
    'batch_size': 12,
    'learning_rate': 0.1
}
model = TextClassifier(params)
train(model, params)
