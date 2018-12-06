### THANOS IMPLEMENTATION WITH TREE LSTM ON DOCUMENTS

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import itertools





def accuracy(SPINN_Model, sent_attn, sent, trans, label, mask_sent):
    # batch.label = batch.label-1
    print('#######Computing Accuracy for the batch ###############')
    max_sents, batch_size, max_tokens = sent.size()
    state_sent = sent_attn.init_hidden().cuda()
    s = None
    for i in range(max_sents):
        _s = SPINN_Model(sent[i, :, :].transpose(0, 1), trans[i, :, :].transpose(0, 1)).unsqueeze(0)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)

    y_pred, state_sent, _ = sent_attn(s, state_sent, mask_sent)
    prob, pred = torch.max(y_pred, 1)
    # print(pred)
    correct = np.ndarray.flatten(pred.data.cpu().numpy())
    labels = np.ndarray.flatten(label.data.cpu().numpy())
    num_correct = sum(correct == labels)
    accuracy = float(num_correct) / len(correct)

    return accuracy




### Helper functions for the SPINN
def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)

def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)

def compute_c_h(c1, c2, lstm):
    g, i, f1, f2, o = lstm.chunk(5, 1)
    c = f1.sigmoid() * c1 + f2.sigmoid() * c2 + i.sigmoid() * g.tanh()
    h = o.sigmoid() * c.tanh()
    return h, c


### Tracker_size is the size of output from LSTM
class Tracker(nn.Module):
    def __init__(self, size, tracker_size):
        super(Tracker, self).__init__()
        ## 3*size is beacuse buffer + stacks1+ stack2 (is  provides as input to the LSTMCell)
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        self.state_size = tracker_size
        self.transition = nn.Linear(tracker_size, 3)

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        ## Get elements from the end os the list since the sentence are stored in reverse order in buffer.
        ## pop top-2 elements from the stack and initially the stack is empty.
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        # print(stacks)
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [Variable(
                x.data.new(x.size(0), self.state_size).zero_())]

        self.state = self.rnn(x, self.state)
        return unbundle(self.state), None


### Composition function will compute the (c,h) pair values for a node, for all nodes in the batch.
### It takes (c,h) values for the left and right child in the bath + (c,h)
### values of the tracker output

class Composition(nn.Module):
    def __init__(self, size, tracker_size=None):
        super(Composition, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        ### bundling converts tensors of dimension (B, 2*H) to two tensors of dimension (B, H)
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        ### Computing the single output for the three different inputs ###
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        lstm_in += self.track(tracking[0])
        ### left[0] & right[0] are the top two values of stack & left[1], right[1] are their corresponding c values..
        states = unbundle(compute_c_h(left[1], right[1], lstm_in))
        return states


class SPINN(nn.Module):
    def __init__(self, d_hidden, d_tracker):
        super(SPINN, self).__init__()
        self.tracker = Tracker(d_hidden, d_tracker)
        self.reduce = Composition(d_hidden, d_tracker)

    def forward(self, buffers, transitions):

        buffers = [list(torch.split(b.squeeze(1), 1, 0))
                   for b in torch.split(buffers, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]

        self.tracker.reset_state()
        if transitions is not None:
            num_transitions = transitions.size(0)

        for i in range(num_transitions):
            ### Obtain the context vector from the TRacker
            ### In our case we already have the transitions defined in the inputs..
            #             print('Printinf Stack length')
            #             for stack in stacks :
            #                 print(len(stack))
            tracker_states, trans_hyp = self.tracker(buffers, stacks)
            if transitions is not None:
                trans = transitions[i]
            # print(trans)

            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)
            ## one iteartion of transition on each sentence in the batch...
            for transition, buf, stack, tracking in batch:

                if transition == 3:  # shift
                    stack.append(buf.pop())
                # print('In Shift')
                elif transition == 2:  # reduce
                    #                     print('In Reduce')
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)

            if rights:
                # print('In Rightss')
                ###If there is some item reduced in stacks, append the reduced value to the stack.
                reduced = iter(self.reduce(lefts, rights, trackings))
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))

        return bundle([stack.pop() for stack in stacks])[0]


########## Spinn Model class ###################
class SPINNClassifier(nn.Module):
    def __init__(self, vocab_size, params):
        super(SPINNClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, params.embed_dim)
        self.linear_trans1 = nn.Linear(params.embed_dim, params.project_dim)

        self.word_dropout = nn.Dropout(0.15)
        self.linear_trans2 = nn.Linear(params.hidden_dim, params.num_class, bias=True)
        self.LogSoftmax = nn.LogSoftmax()

        self.SPINN = SPINN(params.hidden_dim, 64)

        # self.embed_dropout = nn.Dropout(p=embed_dropout)

    def forward(self, token, trans):
        prem_embed = self.embed(token)
        prem_embed1 = self.linear_trans1(prem_embed)
        prem_embed1 = self.word_dropout(prem_embed1)
        Spinn_model = self.SPINN(prem_embed1, trans)
        #         out = self.linear_trans2(Spinn_model)
        #         out = self.LogSoftmax(out)
        # val, pred = torch.max(out, 1)

        return Spinn_model






### Implementing Sention Model
#### Helper functions for implementing Sention Attenstion Model
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



######### Setnece Attention Model ##################
class AttentionSentRNN(nn.Module):
    def __init__(self, params, bidirectional=True):

        super(AttentionSentRNN, self).__init__()

        self.batch_size = params.batch_size
        self.sent_gru_hidden = params.sent_gru
        self.n_classes = params.num_class
        self.word_gru_hidden = params.sent_gru
        self.bidirectional = bidirectional
        self.word_dropout = nn.Dropout(0.10)
        self.bn = nn.BatchNorm1d(self.n_classes)

        if bidirectional == True:
            self.sent_gru = nn.GRU(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 2 * self.sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2 * self.sent_gru_hidden, self.n_classes)
        else:
            self.sent_gru = nn.GRU(self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(self.sent_gru_hidden, self.sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(self.sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(self.sent_gru_hidden, 1))
            self.final_linear = nn.Linear(self.sent_gru_hidden, self.n_classes)
        self.softmax_sent = nn.Softmax()
        self.final_softmax = nn.Softmax()
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1, 0.1)

    def forward(self, word_attention_vectors, state_sent, sent_mask):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1, 0))
        ### Masking the padded 0's ####
        sent_attn_norm = torch.mul(sent_mask, sent_attn_norm)
        sent_attn_norm = F.normalize(sent_attn_norm, p=1, dim=1)
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1, 0))


        # final classifier
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        #final_map = self.bn(final_map)
        final_map = self.word_dropout(final_map)

        return F.log_softmax(final_map), state_sent, sent_attn_norm

    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))






