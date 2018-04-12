# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import numpy as np
import pandas as pd


def evaluate_sessions_batch(model, train_data, test_data, cut_off=20, batch_size=50, session_key='SessionId', item_key='ItemId', time_key='Time', filter_history=True):
    
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by precision@N and MRR@N.

    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for precision@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Precision@N, MRR@N)
   
    '''
    model.predict = False
    
    test_data.sort([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    start_end = np.array(zip(offset_sessions[:-1], offset_sessions[1:]))
    session2items = {k: set(g['ItemId'].values) for k,g in test_data.groupby(session_key)}
    evalutation_point_count = 0
    mrr, precision = 0.0, 0.0
    if len(start_end) < batch_size:
        batch_size = len(start_end)
    iters = np.arange(batch_size)
    maxiter = iters.max()
    start = start_end[iters][:, 0]
    end = start_end[iters][:, 1]
    in_idx = np.array([-1] * batch_size, dtype=np.int32)
    out_idx = np.array([-1] * batch_size, dtype=np.int32)
    #print 'item_list:', itemids
    #print 'test_data:', test_data
    #print 'start_end:', start_end
    while True:
        # 预测允许一个batch不全是可用的样本, 因为预测不进行采样，而是计算softmax，所以不会像训练那样loss依赖于整个batch
        # train的时候对样本进行shuffle,所以缺失最后的不够一个batch的样本对结果影响不大
        valid_mask = iters >= 0
        #print 'valid_mask, iters:', valid_mask, iters
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        #print 'start, start_valid, end, minlen, in_idx:',start, start_valid, end, minlen, in_idx
        for i in xrange(minlen-1):
            out_idx[valid_mask] = test_data[item_key].values[start_valid+i+1]
            preds = model.predict_next_batch(iters, in_idx, batch_size)
            preds.fillna(0, inplace=True)
            #print 'i_of_minlen, in, out, preds:', i, in_idx, out_idx, preds
            in_idx[valid_mask] = out_idx[valid_mask]
            ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0) + 1
            ranks_mask = (end[valid_mask] - start[valid_mask] - i - 1<=1)
            final_session = test_data.SessionId.values[start[valid_mask]][ranks_mask]
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
            if maxiter >= len(start_end):
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = start_end[maxiter][0]
                end[idx] = start_end[maxiter][1]
    return precision/evalutation_point_count, mrr/evalutation_point_count
