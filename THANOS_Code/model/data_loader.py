import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable


def create_toks(lis, vocab_to_index):
    #     val = []
    #     for v in li :
    #         val.append(v)
    toks = []
    i = 0
    for v in lis:
        tok_val = v.split(' ')
        li = [vocab_to_index[v] for v in tok_val]
        toks.append(li)

    return toks


def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype=np.int)
    ### Creating masks for the padded zeros in the sentence and words..
    word_mask = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype=np.long)
    sent_mask = np.zeros((mini_batch_size, max_sent_len), dtype=np.long)

    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i, j, k] = mini_batch[i][j][k]
                except IndexError:
                    pass
            try:
                word_mask[i, j, 0:len(mini_batch[i][j])] = 1
            except IndexError:
                pass

        sent_mask[i, 0:len(mini_batch[i])] = 1
    # print(mask)
    #     print('Printing Data matrix')
    #     print(main_matrix)
    return Variable(torch.from_numpy(main_matrix).transpose(0, 1)), Variable(
        torch.from_numpy(word_mask).transpose(0, 1)).float(), Variable(
        torch.from_numpy(sent_mask)).float()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def gen_minibatch(tokens, labels, mini_batch_size, params, vocab_to_index, shuffle= False):

    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        token, word_mask, sent_mask = pad_batch(token)
        yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).long().cuda(), word_mask.cuda(), sent_mask.cuda()