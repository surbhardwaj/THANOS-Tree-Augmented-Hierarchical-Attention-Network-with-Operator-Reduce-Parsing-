"""Train the model"""
import argparse
import os
from torch import nn
import numpy as np
import torch
import torch.optim as optim
import utils
import model.net as net
from evaluate import evaluate
import pandas as pd
from model.data_loader import gen_minibatch
import json
import time




#### Parsing the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'



def train_model(data_train, SPINNClass_Model, sent_attn, SPINN_optim, sent_optimizer, params, num_steps, vocab_to_index):

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    SPINNClass_Model.train()
    sent_attn.train()
    print('Data generation')
    sent_dropout = nn.Dropout(0.4)
    g = gen_minibatch(data_train['Tree'].values, data_train['Sent'].values, data_train['rating'].values, params.batch_size, params, vocab_to_index)
    # Use tqdm for progress bar
    for batch_idx in range(1, num_steps + 1):
        print(batch_idx)
        SPINN_optim.zero_grad()
        sent_optimizer.zero_grad()
        sent, trans, label, sent_mask = next(g)
        max_sents, batch_size, max_tokens = sent.size()
        if params.cuda:
            state_sent = sent_attn.init_hidden().cuda()
        else:
            state_sent = sent_attn.init_hidden()
        s = None
        for i in range(max_sents):
            _s = SPINNClass_Model(sent[i, :, :].transpose(0, 1), trans[i, :, :].transpose(0, 1)).unsqueeze(0)
            sent_dropout(_s)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

        y_pred, state_sent, _ = sent_attn(s, state_sent, sent_mask)
        if params.cuda:
            loss = params.criterion(y_pred.cuda(), label)
        else:
            loss = params.criterion(y_pred, label)

        loss.backward()
        SPINN_optim.step()
        sent_optimizer.step()
        if batch_idx % params.save_summary_steps == 0:
            # compute all metrics on this batch
            summary_batch = {}
            acc = net.accuracy(SPINNClass_Model, sent_attn, sent, trans, label, sent_mask)
            summary_batch['accuracy'] = acc
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            metrics_string = 'Train Metrics for Batch : {} : Loss: {}; Accuracy :{}'.format(batch_idx, loss.item(), acc)
            utils.log("- Train metrics: " + metrics_string, logger)
        torch.cuda.empty_cache()

        loss_avg.update(loss.item())
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    utils.log("- Train metrics: " + metrics_string, logger)
    
       




def train_evaluate_model(SPINNClass_Model, sent_attn, data_train, data_val, SPINN_optim, sent_optimizer, params, model_dir, restore_file, vocab_to_index):
    # reload weights from restore_file if specified

    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        utils.log("Restoring parameters from {}".format(restore_path), logger)
        utils.load_checkpoint(restore_path, SPINNClass_Model, SPINN_optim, spinn=True)
        utils.load_checkpoint(restore_path, sent_attn, sent_optimizer, spinn=False)

    best_val_acc = 0.0
    start_time = time.time()
 

    for epoch in range(params.num_epochs):
        # Run one epoch
        utils.log("Epoch {}/{}".format(epoch + 1, params.num_epochs), logger)
        # compute number of batches in one epoch (one full pass over the training set)
        
        num_steps = (params.train_size + 1) // params.batch_size
        train_model(data_train, SPINNClass_Model, sent_attn, SPINN_optim, sent_optimizer, params, num_steps, vocab_to_index)
        print("--- %s seconds in completing a epoch---" % (time.time() - start_time))
        start_time = time.time()
        torch.cuda.empty_cache()
        
        
        utils.log("-----Starting Evaluation-----", logger)
        num_steps = (params.val_size + 1) // params.batch_size
        val_metrics = evaluate(data_val, SPINNClass_Model, sent_attn, params, num_steps, vocab_to_index)
        
        torch.cuda.empty_cache()
        

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'SPINN_State_dict': SPINNClass_Model.state_dict(),
                               'SPINN_Optim_dict': SPINN_optim.state_dict(),
                                'Sent_State_dict' :  sent_attn.state_dict(),
                                'Sent_Optim_dict': sent_optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)

        if is_best:
            utils.log("- Found new best accuracy", logger)
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
            
            




if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    print('*** Fetching vocab indexing ***')
    with open("data/vocab_to_index.json") as f:
        vocab_to_index = json.load(f)
    vocab_size = len(vocab_to_index)


    # Set the random seed for reproducible experiments
    torch.manual_seed(560)
    if params.cuda: torch.cuda.manual_seed(560)
        
    # Set the logging
    #log = Logger_log(os.path.join(args.model_dir, 'train.log'))
    logger = utils.setup_file_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    utils.log("Loading the datasets...", logger )
    
    # load data
    data_train = pd.read_pickle(args.data_dir+"/train.pkl")
    #data_train = data_train.head(500)
    data_train['rating'] = pd.to_numeric(data_train['rating'])
    data_train['rating'] = data_train['rating'] - 1
    data_val = pd.read_pickle(args.data_dir+"/dev.pkl")
    #data_val = data_val.head(128)
    data_val['rating'] = pd.to_numeric(data_val['rating'])
    data_val['rating'] = data_val['rating'] - 1



    # specify the train and val dataset sizes
    params.train_size = len(data_train)
    params.val_size = len(data_val)
    #params.val_size = len(data_val)

    utils.log("- done.", logger)

    # Define the model and optimizer

    print(params.cuda)
    SPINNClass_Model = net.SPINNClassifier(vocab_size+1, params).cuda() if params.cuda else net.SPINNClassifier(vocab_size+1, params)
    criterion = nn.CrossEntropyLoss()
    SPINN_optim = optim.RMSprop(SPINNClass_Model.parameters(), lr=params.learning_rate, alpha=params.alpha, eps=params.eps)

    sent_attn = net.AttentionSentRNN(params, bidirectional=True).cuda() if params.cuda else net.AttentionSentRNN(params, bidirectional=True)
    sent_optimizer = optim.RMSprop(sent_attn.parameters(), lr=params.learning_rate, alpha=params.alpha, eps=params.eps)
    params.criterion = nn.NLLLoss()

    print("***Training the model ***")
    utils.log("Starting training for {} epoch(s)".format(params.num_epochs), logger)
    train_evaluate_model(SPINNClass_Model, sent_attn, data_train, data_val, SPINN_optim, sent_optimizer, params, args.model_dir, args.restore_file, vocab_to_index)

    print('## Done ###')
