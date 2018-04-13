# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import pandas as pd
import numpy as np

class GRU4Rec:
    def __init__(self, sess, args):
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        self.sess = sess
        self.is_training = args.is_training
        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items 
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.checkpoint_path = args.checkpoint_path
        if not os.path.isdir(self.checkpoint_path):
            raise Exception("[!] Checkpoint Dir not found")

        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self.is_training:
            return

        # use self.predict_state to hold hidden states during prediction. 
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_path, args.test_model))

        import time
        x_info = tf.saved_model.utils.build_tensor_info(self.X)
        state_info = []
        yhat_info = tf.saved_model.utils.build_tensor_info(self.yhat)
        final_state_info = []
        for i in range(self.layers):
            state_info.append(tf.saved_model.utils.build_tensor_info(self.state[i]))
            final_state_info.append(tf.saved_model.utils.build_tensor_info(self.final_state[i]))
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs = dict([('X', x_info)] + [('state%s' % i, state_info[i]) for i in range(self.layers)]),
                outputs=dict([('yhat', yhat_info)] + [('final_state%s' % i, final_state_info[i]) for i in range(self.layers)]),
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        )
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(args.serving_path, '%.0f' % time.time()))
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict': prediction_signature,
            },
        )
        builder.save()

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat)+1e-24))
    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat)-yhatT)))
    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat)+yhatT)+tf.nn.sigmoid(yhatT**2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat)**2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def build_model(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in xrange(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            self.embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            self.softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            self.softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)
            
            inputs = tf.nn.embedding_lookup(self.embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples. 每个样本会与这个batch里的所有y计算score,然后得到softmax
            '''
            sampled_W = tf.nn.embedding_lookup(self.softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(self.softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:
            logits = tf.matmul(output, self.softmax_W, transpose_b=True) + self.softmax_b
            self.yhat = self.final_activation(logits)

        if not self.is_training:
            return

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True)) 
        optimizer = tf.train.AdamOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs 
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, data):
        data.sort([self.session_key, self.time_key], inplace=True)
        offset_sessions = np.zeros(data[self.session_key].nunique()+1, dtype=np.int32)
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        return np.array(zip(offset_sessions[:-1], offset_sessions[1:]))
    
    def fit(self, data):
        offset_sessions = self.init(data)
        #print 'train_data:', data
        print('fitting model...')
        for epoch in xrange(self.n_epochs):
            np.random.shuffle(offset_sessions)
            #print 'offset_session:', offset_sessions
            epoch_cost = []
            step, lr = 0, self.lr
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in xrange(self.layers)]
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[iters][:,0]
            end = offset_sessions[iters][:,1]
            finished = False
            while not finished:
                minlen = (end-start).min()
                out_idx = data[self.item_key].values[start]
                #print 'start, end, minlen:', start, end, minlen
                for i in range(minlen-1):
                    in_idx = out_idx
                    out_idx = data[self.item_key].values[start+i+1]
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in xrange(self.layers): 
                        feed_dict[self.state[j]] = state[j]
                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    #print 'i_of_minlen, in, out, cost:', i, in_idx, out_idx, cost
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        return
                start = start+minlen-1
                mask = np.arange(len(iters))[(end-start)<=1]
                #print 'start, end, mask:', start, end, mask
                for idx in mask:
                    maxiter += 1
                    #print 'maxiter, offset_sessions:', maxiter, offset_sessions
                    if maxiter >= len(offset_sessions):
                        #当训练样本不足一个batch,结束
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[maxiter][0]
                    end[idx] = offset_sessions[maxiter][1]
                if len(mask) and self.reset_after_session:
                    for i in xrange(self.layers):
                        state[i][mask] = 0
            
            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                return
            print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_path), global_step=epoch)
            with open(os.path.join(self.checkpoint_path, 'embedding_%s' % epoch), 'w') as f:
                item_embedding = self.sess.run(self.embedding)
                for index, vec in enumerate(item_embedding.tolist()):
                    print >>f, str(index) + '\t' + '|'.join(map(lambda x:'%.4f' % x, vec))
            with open(os.path.join(self.checkpoint_path, 'w_%s' % epoch), 'w') as f:
                w = self.sess.run(self.softmax_W)
                for index, vec in enumerate(w.tolist()):
                    print >>f, str(index) + '\t' + '|'.join(map(lambda x:'%.4f' % x, vec))

    
    def predict_next_batch(self, session_ids, input_item_ids, batch=50):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1 
            self.predict = True
        
        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0: # change internal states with session changes
            for i in xrange(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session=session_ids.copy()

        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: input_item_ids}
        for i in xrange(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds)


    def evaluate(self, data, cut_off=20, filter_history=True):
        self.predict = False
        offset_sessions = self.init(data)
        session2items = {k: set(g['ItemId'].values) for k,g in data.groupby(self.session_key)}
        evalutation_point_count = 0
        mrr, precision = 0.0, 0.0
        self.batch_size = min(self.batch_size, len(offset_sessions))
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = offset_sessions[iters][:, 0]
        end = offset_sessions[iters][:, 1]
        in_idx = np.array([-1] * self.batch_size, dtype=np.int32)
        out_idx = np.array([-1] * self.batch_size, dtype=np.int32)
        #print 'data:', data
        #print 'offset_sessions:', offset_sessions
        while True:
            # 预测允许一个batch不全是可用的样本, 因为预测不进行采样，而是计算softmax，所以不会像训练那样loss依赖于整个batch
            # train的时候对样本进行shuffle,所以缺失最后的不够一个batch的样本对结果影响不大
            valid_mask = iters >= 0
            #print 'valid_mask, iters:', valid_mask, iters
            if valid_mask.sum() == 0:
                break
            start_valid = start[valid_mask]
            minlen = (end[valid_mask]-start_valid).min()
            in_idx[valid_mask] = data[self.item_key].values[start_valid]
            #print 'start, start_valid, end, minlen, in_idx:',start, start_valid, end, minlen, in_idx
            for i in xrange(minlen-1):
                out_idx[valid_mask] = data[self.item_key].values[start_valid+i+1]
                preds = self.predict_next_batch(iters, in_idx, self.batch_size)
                preds.fillna(0, inplace=True)
                #print 'i_of_minlen, in, out, preds:', i, in_idx, out_idx, preds
                in_idx[valid_mask] = out_idx[valid_mask]
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
                ranks_mask = (end[valid_mask] - start[valid_mask] - i - 1<=1)
                final_session = data.SessionId.values[start[valid_mask]][ranks_mask]
                out_value = out_idx[valid_mask][ranks_mask]
                final_pred = preds[preds.columns.values[valid_mask][ranks_mask]]
                for _i in range(len(final_session)):
                    item2score = sorted(zip(final_pred.index.values, final_pred[final_pred.columns[_i]].values), key=lambda x:x[1], reverse=True)
                    #print final_session[_i], item2score, session2items[final_session[_i]], out_value[_i], cut_off
                    top = 1
                    for k, v in item2score:
                        if top == cut_off:
                            break
                        if k == out_value[_i]:
                            precision += 1.0
                            mrr += 1.0 / top
                        elif k in session2items[final_session[_i]] and filter_history:
                            continue
                        top += 1
                    evalutation_point_count += 1
                    #print 'precision, mrr, evalutation_point_count, maxiter:', precision, mrr, evalutation_point_count, maxiter
            start = start+minlen-1
            mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
            #print 'start, mask:', start, mask
            for idx in mask:
                maxiter += 1
                if maxiter >= len(offset_sessions):
                    iters[idx] = -1
                else:
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[maxiter][0]
                    end[idx] = offset_sessions[maxiter][1]
        return precision/evalutation_point_count, mrr/evalutation_point_count
