from __future__ import print_function
from models import *

from util import Dictionary, get_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
# import random
import os

from att_visual import createHTML
import numpy as np

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def visualize_attention(visual_list,filename):
    axis_max = max(list(map(lambda x:x[0].shape[2], visual_list)))
    data_concat = np.array([])
    wts_concat = np.array([])
    for wts, data in visual_list:
        data = pad_along_axis(data.detach().cpu().numpy().T,axis_max,axis=1)
        if data_concat.any():
            data_concat = np.concatenate((data_concat,data),axis=0)
        else:
            data_concat = data
        wts_add = torch.sum(wts,1)
        wts_add_np = pad_along_axis(wts_add.data.detach().cpu().numpy(),axis_max,axis=1)
        if wts_concat.any():
            wts_concat = np.concatenate((wts_concat,wts_add_np),axis=0)
        else:
            wts_concat = data

    createHTML(list(map(lambda x: ' '.join(list(map(lambda y:dictionary.idx2word[y], x))),data_concat)), wts_concat.tolist(), filename)
    print("Attention visualization created for {} samples".format(len(data_concat)))
    return

def package(data, requires_grad=False):
    """Package data for training / evaluation."""
    data = list(map(lambda x: json.loads(x), data))
    dat = list(map(lambda x: list(map(lambda y: dictionary.word2idx[y], x['text'])), data))
    maxlen = 0
    for item in dat:
        maxlen = max(maxlen, len(item))
    targets = list(map(lambda x: label_list.word2idx[x['label']], data))
    maxlen = min(maxlen, 3500)
    for i in range(len(data)):
        if maxlen < len(dat[i]):
            dat[i] = dat[i][:maxlen]
        else:
            for j in range(maxlen - len(dat[i])):
                dat[i].append(dictionary.word2idx['<pad>'])
    
    dat = Variable(torch.LongTensor(dat), requires_grad=requires_grad)
    targets = Variable(torch.LongTensor(targets), requires_grad=requires_grad)
    return dat.t(), targets


def predict():
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    total_loss = 0
    total_correct = 0
    visual_list = list()
    for batch, i in enumerate(range(0, len(raw_all_data), args.batch_size)):
        data, targets = package(raw_all_data[i:min(len(raw_all_data), i+args.batch_size)], requires_grad=False)
        hidden = model.init_hidden(data.size(1))
        output, attention = model.forward(data, hidden)
        output_flat = output.view(data.size(1), -1)
        prediction = torch.max(output_flat, 1)[1]
        # print(prediction)
        total_correct += torch.sum((prediction == targets).float())
        visual_list.append((attention.clone(), data.clone()))

    visualize_attention(visual_list,filename='attention_test.html')
    return total_loss / (len(raw_all_data) // args.batch_size), total_correct.data / len(raw_all_data)


'''
    Traning 소스를 가져와서 predict용으로 살짝 변경해주었습니다.
    visualize가 필요하므로, 해당 소스를 활용합니다.
'''
if __name__ == '__main__':
    # parse the arguments
    args = get_args()

    # Set the random seed manually for reproducibility.
    # torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #     else:
    #         torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.data_path)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    label_list = Dictionary(path=args.label_list)

    best_val_loss = None
    best_acc = None

    n_token = len(dictionary)
    '''
        모델명 변경 필요
    '''
    model = torch.load('./models/model-medium.pt')
    model = model.cpu()
    raw_all_data = open(args.data_path).readlines()

    data, _ = package(raw_all_data, requires_grad=False)
    predict()
