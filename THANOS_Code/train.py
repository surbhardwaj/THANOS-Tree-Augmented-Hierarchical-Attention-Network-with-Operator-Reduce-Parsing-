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
from model.data_loader import create_toks



#### Parsing the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'



def train_model(data_train, word_attn, sent_attn, word_optimizer, sent_optimizer, params, num_steps, vocab_to_index):

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    word_attn.train()
    sent_attn.train()
    print('Data generation')
    g = gen_minibatch(data_train['toks'].values, data_train['rating'].values, params.batch_size, params, vocab_to_index)
    # Use tqdm for progress bar
    for batch_idx in range(1, num_steps + 1):
        word_optimizer.zero_grad()
        sent_optimizer.zero_grad()
        tokens, labels, word_mask, sent_mask = next(g)
        state_word = word_attn.init_hidden().cuda()
        state_sent = sent_attn.init_hidden().cuda()
        max_sents, batch_size, max_tokens = tokens.size()

        s = None
        
        for i in range(max_sents):
            _s, state_word, _ = word_attn(tokens[i, :, :].transpose(0, 1), state_word, word_mask[i, :, :])
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)



        y_pred, state_sent, _ = sent_attn(s, state_sent, sent_mask)
        loss = params.criterion(y_pred.cuda(), labels)
        loss.backward()
        word_optimizer.step()
        sent_optimizer.step()
        if batch_idx % params.save_summary_steps == 0:
            # compute all metrics on this batch
            summary_batch = {}
            acc = net.test_accuracy_mini_batch(tokens, labels, word_mask, sent_mask, word_attn, sent_attn)

            summary_batch['accuracy'] = acc
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            metrics_string = 'Train Metrics for Batch : {} : Loss: {}; Accuracy :{}'.format(batch_idx, loss.item(), acc)
            utils.log("- Train metrics: " + metrics_string, logger)
        

        loss_avg.update(loss.item())
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    utils.log("- Train metrics: " + metrics_string, logger)




def train_evaluate_model(word_attn, sent_attn, data_train, data_val, word_optmizer, sent_optimizer, params, model_dir, restore_file, vocab_to_index):
    # reload weights from restore_file if specified

    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        utils.log("Restoring parameters from {}".format(restore_path), logger)
        utils.load_checkpoint(restore_path, word_attn, word_optmizer, spinn=True)
        utils.load_checkpoint(restore_path, sent_attn, sent_optimizer, spinn=False)

    best_val_acc = 0.0
    i = 0

    for epoch in range(params.num_epochs):
        # Run one epoch
        utils.log("Epoch {}/{}".format(epoch + 1, params.num_epochs), logger)
        # compute number of batches in one epoch (one full pass over the training set)
        ### RUn the model over 1 epoch...

        num_steps = (params.train_size + 1) // params.batch_size
        print(num_steps)
        train_model(data_train, word_attn, sent_attn, word_optmizer, sent_optimizer, params, num_steps, vocab_to_index)

        utils.log("-----Starting Evaluation-----", logger)
        num_steps = (params.val_size + 1) // params.batch_size
        val_metrics = evaluate(data_val, word_attn, sent_attn, params, num_steps, vocab_to_index)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'Word_State_dict': word_attn.state_dict(),
                               'Word_Optim_dict': word_optmizer.state_dict(),
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

    #log = Logger_log(os.path.join(args.model_dir, 'train.log'))
    logger = utils.setup_file_logger(os.path.join(args.model_dir, 'train.log'))


    with open("/home/user1/Surbhi/HAN_new/Yelp_data/vocab_to_index.json") as f:
        vocab_to_index = json.load(f)
    vocab_size = len(vocab_to_index)+1


    # Set the random seed for reproducible experiments
    torch.manual_seed(560)
    if params.cuda: torch.cuda.manual_seed(560)
        
    # Set the logging
    
    # Create the input data pipeline
    utils.log("Loading the datasets...", logger )
    
    # load data
    data_train = pd.read_pickle(args.data_dir+"/train.pkl")
    #data_train =  data_train.head(800)
    data_train['rating'] = pd.to_numeric(data_train['rating'])
    data_train['rating'] = data_train['rating'] - 1
    data_val = pd.read_pickle(args.data_dir+"/dev.pkl")
    data_val['rating'] = pd.to_numeric(data_val['rating'])
    data_val['rating'] = data_val['rating'] - 1
    #data_val = data_val.head(200)
    
    data_train['toks'] = data_train.Sent.apply(lambda x: create_toks(x,vocab_to_index ))
    data_val['toks'] = data_val.Sent.apply(lambda x: create_toks(x, vocab_to_index))




    # specify the train and val dataset sizes
    params.train_size = len(data_train)
    params.val_size = len(data_val)
    #params.val_size = len(data_val)

    utils.log("- done.", logger)

    # Define the model and optimizer

    word_attn = net.AttentionWordRNN(batch_size=params.batch_size, num_tokens=vocab_size, embed_size=params.embed_size, word_gru_hidden=params.word_gru_hidden, params= params ,bidirectional=False)
    word_attn = word_attn.cuda()

    sent_attn = net.AttentionSentRNN(batch_size=params.batch_size, sent_gru_hidden=params.sent_gru_hidden, word_gru_hidden=params.word_gru_hidden,n_classes=params.num_classes, params= params, bidirectional=False)
    sent_attn =sent_attn.cuda()

    word_optmizer = torch.optim.RMSprop(word_attn.parameters(), lr=params.learning_rate, alpha=params.alpha, eps=params.eps)
    sent_optimizer = torch.optim.RMSprop(sent_attn.parameters(), lr=params.learning_rate, alpha=params.alpha, eps=params.eps)
    criterion = nn.NLLLoss()
    params.criterion = nn.NLLLoss()

    print("***Training the model ***")
    utils.log("Starting training for {} epoch(s)".format(params.num_epochs), logger)
    train_evaluate_model(word_attn, sent_attn, data_train, data_val, word_optmizer, sent_optimizer, params, args.model_dir, args.restore_file, vocab_to_index)

    print('## Done ###')
