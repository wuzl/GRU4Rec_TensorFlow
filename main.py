# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model

class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 50
    seed = 13
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.9
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_path = './data/checkpoint'
    summary_path = './data/summary'
    serving_path = './data/exported'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--batch', default=50, type=int)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--top', default=50, type=int)
    parser.add_argument('--train_path', default='data/rsc15_train_full.txt.14', type=str)
    parser.add_argument('--test_path', default='data/rsc15_test.txt.8', type=str)
    parser.add_argument('--checkpoint_path', default='data/checkpoint', type=str)
    parser.add_argument('--summary_path', default='data/summary', type=str)
    parser.add_argument('--serving_path', default='data/exported', type=str)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=0.9, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    top = command_line.top
    args = Args()
    data = pd.read_csv(command_line.train_path, sep='\t', dtype={args.item_key: np.int64})
    valid = pd.read_csv(command_line.test_path, sep='\t', dtype={args.item_key: np.int64})
    args.n_items = len(data[args.item_key].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.batch_size = min(command_line.batch, len(valid)) if args.is_training == 0 else min(command_line.batch, len(data))
    args.seed = command_line.seed
    args.checkpoint_path = command_line.checkpoint_path
    args.summary_path = command_line.summary_path
    args.serving_path = command_line.serving_path
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.decay = command_line.decay
    args.decay_steps = len(data) / args.batch_size
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print args.__dict__
    for path in [args.checkpoint_path, args.summary_path, args.summary_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data)
        else:
            data = None
            res = gru.evaluate(valid, cut_off=top)
            print('Precision@{}: {}\tMRR@{}: {}'.format(top, res[0], top, res[1]))
