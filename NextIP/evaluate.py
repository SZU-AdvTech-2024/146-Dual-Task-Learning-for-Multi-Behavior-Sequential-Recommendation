from __future__ import absolute_import, division, print_function

import numpy as np
import math
import multiprocessing
import time

def evaluate_rec_ndcg_mrr(ratings, top_k=10, row_target_position=0):
    ratings = np.array(ratings)
    ratings = ratings[~ np.any(np.isnan(ratings), -1)]

    num_rows = len(ratings)
    if num_rows == 0:
        return 0, 0, 0

    ranks = np.argsort(np.argsort(-np.array(ratings), axis=-1), axis=-1)[:, row_target_position] + 1

    ranks = ranks[ranks <= top_k]

    rec = len(ranks) / num_rows
    ndcg = np.sum(1 / np.log2(ranks + 1)) / num_rows
    mrr = np.sum(1 / ranks) / num_rows

    return rec, ndcg, mrr

def evaluate_rec_ndcg_mrr_batch(ratings,results, top_k=10, row_target_position=0):
    ratings = np.array(ratings)
    ratings = ratings[~ np.any(np.isnan(ratings), -1)]

    num_rows = len(ratings)
    if num_rows == 0:
        return 0, 0, 0

    ranks = np.argsort(np.argsort(-np.array(ratings), axis=-1), axis=-1)[:, row_target_position] + 1

    ranks = ranks[ranks <= top_k]

    results[0] += len(ranks)
    results[1] += np.sum(1 / np.log2(ranks + 1))
    results[2] += np.sum(1 / ranks)

def evaluate_nop_error_batch(preditions, labels, error):
    # preditions = np.array(preditions)
    # error[0] += np.sum(np.abs(labels-preditions))
    preditions = np.array(preditions)
    error[0] += np.sum((np.abs(labels - preditions) < 0.5) * (preditions > 0.5))  # TP
    error[1] += np.sum((np.abs(labels - preditions) < 0.5) * (preditions < 0.5))  # TN
    error[2] += np.sum((np.abs(labels - preditions) >= 0.5) * (preditions < 0.5))  # FN
    error[3] += np.sum((np.abs(labels - preditions) >= 0.5) * (preditions > 0.5))  # FP

def evaluate_nop_error_batch2(preditions, labels, error):
    preditions = np.array(preditions)
    error[0] += np.sum(np.abs(labels-preditions))


def evaluate_rec_ndcg_mrr_grouped(ratings, seq_length, 
                                  top_k=10, row_target_position=0, group_segment=[5,10,20,50]):
    indexs_grouped = []
    len_bound = 0
    for group_max_len in group_segment:
        indexs_gi = np.where((seq_length > len_bound) & (seq_length <= group_max_len))[0]
        indexs_grouped.append(indexs_gi)
        len_bound = group_max_len
    indexs_gl = np.where(seq_length > len_bound)[0]
    indexs_grouped.append(indexs_gl)

    recs = []
    ndcgs = []
    mrrs = []

    print("---- grouped results, segment=%s" % group_segment)

    for i in range(len(indexs_grouped)):
        _ratings = np.array(ratings)[np.array(indexs_grouped[i])]

        num_rows = len(_ratings)
        if num_rows > 0:
            ranks = np.argsort(np.argsort(-np.array(_ratings), axis=-1), axis=-1)[:, row_target_position] + 1
            ranks = ranks[ranks <= top_k]

            rec = len(ranks) / num_rows
            ndcg = np.sum(1 / np.log2(ranks + 1)) / num_rows
            mrr = np.sum(1 / ranks) / num_rows
        else:
            rec = -1
            ndcg = -1
            mrr = -1

        print("            HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f" % (rec, ndcg, mrr))
        recs.append(rec)
        ndcgs.append(ndcg)
        mrrs.append(mrr)

    return recs, ndcgs, mrrs
