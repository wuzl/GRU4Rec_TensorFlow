# GRU4Rec_TensorFlow
TensorFlow implementation of *GRu4Rec*, which was descibed in "Session-based Recommendations With Recurrent Neural Networks". See paper: http://arxiv.org/abs/1511.06939. 

# Requirements
Python: 2.7

Pandas < 0.17 

Numpy 1.12.1 or later

TensorFlow: 0.12.1

# Usage
Get dataset

    $ cd data
    $ curl -Lo yoochoose-data.7z https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z
    $ 7za x yoochoose-data.7z
    $ cd ..
    $ python3 preprocess.py

Train/Test file should consists of three columns:   

     First column: SessionId  
     Second column: ItemId  
     Third column: Timestamps

To train a model with default parameter settings:

    $ python main.py --size 3 --batch 2 --epoch 100 --train_path data/rsc15_train_full.txt.14 --test_path data/rsc15_test.txt.8 --lr 0.01 --dropout 1
    $ python main.py --layer 2 --size 200 --batch 5000 --lr 0.001 --dropout 0.8 --epoch 25 --train_path data/UT.train.seq.csv --test_path data/UT.test.seq.csv --checkpoint_path data/checkpoint

    Other optional parameters include:   
     --layer: Number of GRU layers. Default is 1.  
     --size: Number of hidden units in GRU model. Default is 100.   
     --epoch: Runing epochs. Default is 3.   
     --lr : Initial learning rate. Default is 0.001.   
     --train: Specify whether training(1) or evaluating(0). Default is 1.   
     --hidden_act: Activation function used in GRU units. Default is tanh.   
     --final_act: Final activation function. Default is softmax.    
     --loss: Loss functions, cross-entropy, bpr or top1 loss. Default is cross-entropy.      
     --dropout: Dropout rate. Default is 0.5.

To evaluate a trained model:

    $ python main.py --size 3 --batch 2 --top 3 --train 0 --test 99  --train_path data/rsc15_train_full.txt.14 --test_path data/rsc15_test.txt.8
    $ python main.py --layer 2 --size 200 --batch 100 --top 50 --train 0 --test 24 --train_path data/UT.train.seq.csv --test_path data/UT.test.seq.csv --checkpoint_path data/checkpoint
    
    One optional parameter is:    
     --test: Specify which saved model to evaluate(only used when --train is 0). Default is 2.

# Acknowledgement
This repository refers a lot to the original [Theano implementation](https://github.com/hidasib/GRU4Rec).
