import argparse
import logging
import os
from model.data_loader import gen_minibatch
import numpy as np
import torch
import utils
import model.net as net
import pandas as pd
import json
import torch.optim as optim
from torch import nn



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(data, word_attn, sent_attn, params, num_steps, vocab_to_index):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    word_attn.eval()
    sent_attn.eval()
    # summary for current eval loop
    summ = []
    summ = []
    g = gen_minibatch(data['toks'].values, data['rating'].values, params.batch_size,params,vocab_to_index, shuffle=False)
    # compute metrics over the dataset
    for batch_idx in range(1, num_steps + 1):
        with torch.no_grad():
            tokens, label, word_mask, sent_mask = next(g)

            max_sents, batch_size, max_tokens = tokens.size()
            state_word = word_attn.init_hidden().cuda()
            state_sent = sent_attn.init_hidden().cuda()

            s = None
            for i in range(max_sents):
                _s, state_word, _ = word_attn(tokens[i, :, :].transpose(0, 1), state_word, word_mask[i, :, :])
                if (s is None):
                    s = _s
                else:
                    s = torch.cat((s, _s), 0)

            y_pred, state_sent, _ = sent_attn(s, state_sent, sent_mask)
            loss = params.criterion(y_pred.cuda(), label)
            prob, pred = torch.max(y_pred, 1)
            # print(pred)
            correct = np.ndarray.flatten(pred.data.cpu().numpy())
            labels = np.ndarray.flatten(label.data.cpu().numpy())
            num_correct = sum(correct == labels)
            accuracy = float(num_correct) / len(correct)

            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['accuracy'] = accuracy
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    logger = utils.setup_file_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    utils.log("Creating the dataset...", logger)

    # load data
    data_test = pd.read_pickle(args.data_dir + "/test.pkl")
    data_test['rating'] = pd.to_numeric(data_test['rating'])
    data_test['rating'] = data_test['rating'] - 1
    # specify the test set size
    params.test_size = len(data_test)

    utils.log("- done.", logger)
    with open("/home/user1/Surbhi/cs230-code-examples-master/pytorch/nlp/data/vocab_to_index.json") as f:
        vocab_to_index = json.load(f)
    vocab_size = len(vocab_to_index)+1

    # Define the model
    SPINNClass_Model = net.SPINNClassifier(vocab_size + 1, params)
    criterion = nn.CrossEntropyLoss()
    SPINN_optim = optim.RMSprop(SPINNClass_Model.parameters(), lr=params.learning_rate, alpha=params.alpha,
                                eps=params.eps)

    sent_attn = net.AttentionSentRNN(params, bidirectional=True)
    sent_optimizer = optim.RMSprop(sent_attn.parameters(), lr=params.learning_rate, alpha=params.alpha, eps=params.eps)
    params.criterion = nn.NLLLoss()

    utils.log("Starting evaluation", logger)

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), SPINNClass_Model, spinn=True)
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), sent_attn, spinn=False)



    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size

    test_metrics = evaluate(data_test, SPINNClass_Model, sent_attn, params, num_steps, vocab_to_index)

    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
