import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable
from torch.autograd import Variable
def convert_unary_binary_bracketed_data(tree_data):
    trans = []
    for data in tree_data:
        examples = []
        for line in data:
            example = {}
            line = line.strip()
            #print(line)
            if len(line) == 0:
                continue
            example["label"] = line[1]
            example["sentence"] = line
            example["tokens"] = []
            example["transitions"] = []

            words = example["sentence"].split(' ')
            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":
                        example["transitions"].append(2)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(3)
            examples.append(example['transitions'])
        trans.append(examples)

    return trans


def create_toks(li, vocab_to_index):
    tokens = []
    for v in li:
        toks = []
        ### For each sentence
        for val in v:
            tokend_val = val.split(' ')
            li = [vocab_to_index[v] for v in tokend_val]
            toks.append(li)

        tokens.append(toks)

    return tokens

def iterate_minibatches(inputs, sent, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], sent[excerpt], targets[excerpt]


def pad_batch(tokens, trans):
    mini_batch_size = len(tokens)
    max_sent_len = int(np.max([len(x) for x in tokens]))
    max_token_len = int(np.max([len(val) for sublist in tokens for val in sublist]))
    sent_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len + 1), dtype=np.float)
    l = 0
    sent_mask = np.zeros((mini_batch_size, max_sent_len), dtype = np.long)
    ## Getting the padded matrix for Sentence tokens
    for i in range(sent_matrix.shape[0]):
        for j in range(sent_matrix.shape[1]):
            l = 0
            for k in range(sent_matrix.shape[2] - 1, -1, -1):
                try:
                    sent_matrix[i, j, k] = tokens[i][j][l]
                    l = l + 1

                except IndexError:
                    pass
        sent_mask[i, 0:len(tokens[i])] = 1



                    ## Getting padded matrix for the Transition tokens
    max_trans_len = int(np.max([len(val) for sublist in trans for val in sublist]))
    trans_matrix = np.zeros((mini_batch_size, max_sent_len, max_trans_len), dtype=np.long)
    for i in range(trans_matrix.shape[0]):
        for j in range(trans_matrix.shape[1]):
            for k in range(trans_matrix.shape[2]):
                try:

                    trans_matrix[i, j, k] = trans[i][j][k]
                except IndexError:
                    pass

    return Variable(torch.from_numpy(sent_matrix).transpose(0, 1)).long(), Variable(
        torch.from_numpy(trans_matrix).transpose(0, 1)).long(), Variable(torch.from_numpy(sent_mask)).float()



def gen_minibatch(tree, sent, labels, mini_batch_size, params,vocab, shuffle= False):
    for tree, sent, label in iterate_minibatches(tree, sent, labels, mini_batch_size, shuffle= shuffle):
        transitions = convert_unary_binary_bracketed_data(tree)
        tokens = create_toks(sent, vocab)
        sent, trans, sent_mask = pad_batch(tokens, transitions)
        if params.cuda:
            yield sent.cuda(), trans.cuda(),  Variable(torch.from_numpy(label), requires_grad= False).cuda(), sent_mask.cuda()
        else:
            yield sent, trans, Variable(torch.from_numpy(label), requires_grad=False), sent_mask
