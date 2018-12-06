"""Build vocabularies of words and tags from datasets"""

import argparse
import json
import os
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP
import pickle




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kaggle', help="Directory containing the dataset")



def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def get_vocab(data):
    pass


# ### Craete the vocab for the data and assign index to each word
def create_vocab():
    vocab_to_index = {}
    with open(args.data_dir + '/vocab.pkl', 'r') as f:
        vocab = pickle.load(f)
        for i, word in enumerate(vocab):
            vocab_to_index[word.rstrip('\n')] = i + 1
    return vocab_to_index



if __name__ == '__main__':
    args = parser.parse_args()
    vocab_to_index = create_vocab()
    save_dict_to_json(vocab_to_index, 'data/vocab_to_index.json')


    print("### Vocab building and indexing done ###")









