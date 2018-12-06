import json
import logging
import os
import shutil

import torch
import datetime


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

# class Logger_log:
#
#     def __init__(self, log_path):
#         self.logger = logging.getLogger('THANOS')
#         self.hdlr = logging.FileHandler(log_path)
#         self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#
#
#
#     def set_logger(self):
#         """Set the logger to log info in terminal and file `log_path`.
#
#         In general, it is useful to have a logger so that every output to the terminal is saved
#         in a permanent file. Here we save it to `model_dir/train.log`.
#
#         Example:
#         ```
#         logging.info("Starting training...")
#         ```
#
#         Args:
#         log_path: (string) where to log
#         """
#         #logger = logging.getLogger('THANOS')
#         #hdlr = logging.FileHandler(log_path)
#
#         self.hdlr.setFormatter(self.formatter)
#         self.logger.addHandler(self.hdlr)
#         self.logger.setLevel(logging.INFO)
#         return self.logger
#
#
#
#     def reset_logger(self):
#         """Set the logger to log info in terminal and file `log_path`.
#
#         In general, it is useful to have a logger so that every output to the terminal is saved
#         in a permanent file. Here we save it to `model_dir/train.log`.
#
#         Example:
#         ```
#         logging.info("Starting training...")
#         ```
#
#         Args:
#             log_path: (string) where to log
#         """
#         self.logger.removeHandler(self.hdlr)
#         self.hdlr.close()






def setup_file_logger(log_file):
    logger = logging.getLogger()
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def log(message, logger):
    # outputs to Jupyter console
    #print('{} {}'.format(datetime.datetime.now(), message))
    # outputs to file
    logger.info(message)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, spinn=True):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    if spinn==True:
        print('Here')
        model.load_state_dict(checkpoint['SPINN_State_dict'])
    else:
        model.load_state_dict(checkpoint['Sent_State_dict'])

    if optimizer:
        if spinn == True:
            optimizer.load_state_dict(checkpoint['SPINN_Optim_dict'])
        else:
            optimizer.load_state_dict(checkpoint['Sent_Optim_dict'])

    return checkpoint