### THANOS IMPLEMENTATION WITH TREE LSTM ON DOCUMENTS

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import itertools


def get_predictions(val_tokens, word_mask, sent_mask, word_attn_model, sent_attn_model):
    max_sents, batch_size, max_tokens = val_tokens.size()
    state_word = word_attn_model.init_hidden().cuda()
    state_sent = sent_attn_model.init_hidden().cuda()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(val_tokens[i, :, :].transpose(0, 1), state_word, word_mask[i, :, :])
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    y_pred, state_sent, _ = sent_attn_model(s, state_sent, sent_mask)
    return y_pred



def test_accuracy_mini_batch(tokens, labels, word_mask, sent_mask, word_attn, sent_attn):
    y_pred = get_predictions(tokens, word_mask, sent_mask, word_attn, sent_attn)
    _, y_pred = torch.max(y_pred, 1)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = np.ndarray.flatten(labels.data.cpu().numpy())
    num_correct = sum(correct == labels)
    return float(num_correct) / len(correct)


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()



def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)




class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden,params, bidirectional):

        super(AttentionWordRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.word_dropout = nn.Dropout(0.15)
        self.attn_dropout = nn.Dropout(params.dropout)

        self.lookup = nn.Embedding(num_tokens, embed_size)
        # self.embedding, num_embeddings, embedding_dim = create_emb_layer(weight_matrix)
        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True, batch_first=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 2 * word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))

        self.softmax_word = nn.Softmax()
        # self.weight_W_word.data.uniform_(-0.1, 0.1)
        torch.nn.init.xavier_uniform(self.weight_W_word)
        # self.weight_proj_word.data.uniform_(-0.1,0.1)
        torch.nn.init.xavier_uniform(self.weight_proj_word)
        self.bias_word.data.fill_(0)

    def forward(self, embed, state_word, word_mask):
        # embeddings

        embedded = self.lookup(embed)
        embedded = self.word_dropout(embedded)
        # embedded = self.embedding(embed).float()
        # word level gru
        output_word, state_word = self.word_gru(embedded)
        #         print output_word.size()
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)

        word_attn_norm = self.softmax_word(word_attn.transpose(1, 0))

        ### Masking the padded zeros in the sentences...

        word_attn_norm = torch.mul(word_mask, word_attn_norm)
        word_attn_norm = F.normalize(word_attn_norm, p=1, dim=1)

        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))
        word_attn_vectors = self.attn_dropout(word_attn_vectors)

        return word_attn_vectors, state_word, word_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))


class AttentionSentRNN(nn.Module):
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, params, bidirectional=True):

        super(AttentionSentRNN, self).__init__()
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.word_dropout = nn.Dropout(params.dropout)

        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 2 * sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2 * sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=False)
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax()
        self.final_softmax = nn.Softmax()
        # self.weight_W_sent.data.uniform_(-0.1, 0.1)
        torch.nn.init.xavier_uniform(self.weight_W_sent)
        # self.weight_proj_sent.data.uniform_(-0.1,0.1)
        torch.nn.init.xavier_uniform(self.weight_proj_sent)
        self.bias_sent.data.fill_(0)

    def forward(self, word_attention_vectors, state_sent, sent_mask):
        
        sent_lengths = (sent_mask>0).sum(1)
        
        output_sent, state_sent = self.sent_gru(word_attention_vectors)
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1, 0))

        ### Masking the padded 0's ####

        sent_attn_norm = torch.mul(sent_mask, sent_attn_norm)
        sent_attn_norm = F.normalize(sent_attn_norm, p=1, dim=1)

        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1, 0))
        # final classifier
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        final_map = self.word_dropout(final_map)

        return F.log_softmax(final_map), state_sent, sent_attn_norm

    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))