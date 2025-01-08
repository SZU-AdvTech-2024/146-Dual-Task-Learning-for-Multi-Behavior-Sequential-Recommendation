from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import datetime
import random
from dataset import Dataset
from evaluate import *
from nextip import Model
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Configurations.")
    parser.add_argument('--path', type=str, default='../data_process/UB/processed_data/', help='Path of data files.')
    parser.add_argument('--dataset', type=str, default='UB', help='Name of the dataset (e.g. Tmall).')
    # parser.add_argument(
    #     '--path', type=str, default='../data_process/Tmall/processed_data/', help='Path of data files.')
    # parser.add_argument('--dataset', type=str, default='Tmall',
    #                     help='Name of the dataset (e.g. Tmall).')
    parser.add_argument('--bw', type=str, default='', help='behavior weight.')
    parser.add_argument('--valid', type=int, default=0,
                        help='Whether to evaluate on the validation set.')
    parser.add_argument('--random_seed', type=int,
                        default=1, help='Random seed.')
    parser.add_argument('--num_epochs', type=int,
                        default=5000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--maxlen', type=int, default=50,
                        help='Maximum length of sequences.')
    parser.add_argument('--num_behavior', type=int,
                        default=4, help='num_behavior.')
    parser.add_argument('--hidden_units', type=int, default=50,
                        help='i.e. latent vector dimensionality.')
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='Number of intra-sequence self-attention blocks.')
    parser.add_argument('--num_blocks_behavior', type=int, default=1,
                        help='Number of behavior self-attention blocks.')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of heads for intra-sequence attention.')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--ext_modules', type=str, default='l',
                        help='Extension. ')
    parser.add_argument('--wo', type=str, default='',
                        help='E. ')  # g:
    parser.add_argument('--eva_interval', type=int, default=1,
                        help='Number of epoch interval for evaluation.')
    parser.add_argument('--L', type=int, default=10, help='L.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('\n'.join([str(k) + ',' + str(v) for k, v in vars(args).items()]))

    # Loading data
    dataset = Dataset(args.path + args.dataset, args.valid)
    dataset.fix_length(args.maxlen)

    # Build model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    model = Model(dataset.user_maxid, dataset.item_maxid, args)

    sess.run(tf.global_variables_initializer())

    # ---- Train model ----
    num_full_batch = len(dataset.user_set) // args.batch_size
    size_last_batch = len(dataset.user_set) % args.batch_size

    print("----")
    t_start = time.time()

    best_rec10 = 0.0
    best_ndcg10 = 0.0
    best_mrr10 = 0.0
    best_rec5 = 0.0
    best_ndcg5 = 0.0
    best_mrr5 = 0.0
    best_rec1 = 0.0

    best_epoch = 0
    for ep in range(1, args.num_epochs + 1):
        start_time_i = datetime.datetime.now()  # epoch start
        for b in range(num_full_batch):
            users, inps, behs, poss, negs, target_feedback, mask_pos = dataset.sample_batch(
                args.batch_size)
            _, loss = sess.run([model.train_op, model.loss],
                               {model.input_id: inps,
                                model.user_id: users,
                                model.pos_id: poss,
                                model.neg_id: negs,
                                model.is_training: True,
                                model.target_behavior: target_feedback,
                                model.behavior_id: behs
                                })
        users, inps, behs, poss, negs, target_feedback, mask_pos = dataset.sample_batch(
            args.batch_size)

        _, loss = sess.run(
            [model.train_op, model.loss],
            {model.input_id: inps,
             model.user_id: users,
             model.pos_id: poss, model.neg_id: negs,
             model.is_training: True,
             model.target_behavior: target_feedback,
             model.behavior_id: behs
             })

        over_time_i = datetime.datetime.now()  # epoch end
        total_time_i = (over_time_i - start_time_i).total_seconds()
        if ep % 1 == 0:
            print('epoch %s done ,total times: %s' % (ep, total_time_i))
            print('epoch %s done ,loss: %s' % (ep, loss))

        # Evaluate model
        if ep % args.eva_interval == 0:
            u_list = list(dataset.valid_seq.keys())
            input_list = np.array(list(dataset.valid_seq.values()))[:, :-1]

            tar = np.array(list(dataset.valid_seq.values()))[
                :, -1][:, np.newaxis]
            behavior_list = np.array(
                list(dataset.valid_behaiovr_seq.values()))[:, :-1]

            neg_cand = np.array(list(dataset.valid_neg_cand.values()))
            cand_list = np.hstack((tar, neg_cand)).tolist()

            user_num = len(u_list)
            pred_ratings = []
            results = [0., 0., 0.]
            results_1 = [0., 0., 0.]
            results_5 = [0., 0., 0.]

            while len(u_list) > 0:
                pred_ratings = list(sess.run(model.cand_rating,
                                             {model.input_id: input_list[-args.batch_size:],
                                              model.user_id: u_list[-args.batch_size:],
                                              model.candidate_id: cand_list[-args.batch_size:],
                                              model.is_training: False,
                                              model.behavior_id: behavior_list[-args.batch_size:]
                                              }))

                u_list = u_list[:-args.batch_size]
                input_list = input_list[:-args.batch_size]

                tar = tar[:-args.batch_size]
                behavior_list = behavior_list[:-args.batch_size]

                cand_list = cand_list[:-args.batch_size]

                evaluate_rec_ndcg_mrr_batch(
                    pred_ratings, results_1, top_k=1, row_target_position=0)
                evaluate_rec_ndcg_mrr_batch(
                    pred_ratings, results_5, top_k=5, row_target_position=0)
                evaluate_rec_ndcg_mrr_batch(
                    pred_ratings, results, top_k=10, row_target_position=0)

            rec_1, ndcg_1, mrr_1 = results_1[0] / \
                user_num, results_1[1] / user_num, results_1[2] / user_num
            print("epoch: %4d, HR@1: %.6f, NDCG@1: %.6f, MRR: %.6f [loss: %.6f, time: %ds]" % (
                ep, rec_1, ndcg_1, mrr_1, loss, (time.time() - t_start)), end='\n')
            rec_5, ndcg_5, mrr_5 = results_5[0] / \
                user_num, results_5[1] / user_num, results_5[2] / user_num
            print("epoch: %4d, HR@5: %.6f, NDCG@5: %.6f, MRR: %.6f [loss: %.6f, time: %ds]" % (
                ep, rec_5, ndcg_5, mrr_5, loss, (time.time() - t_start)), end='\n')
            rec, ndcg, mrr = results[0] / user_num, results[1] / \
                user_num, results[2] / user_num
            print("epoch: %4d, HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f [loss: %.6f, time: %ds]" % (
                ep, rec, ndcg, mrr, loss, (time.time() - t_start)), end='\n')

            if rec > best_rec10:
                best_rec10 = rec
                best_ndcg10 = ndcg
                best_mrr10 = mrr
                best_rec5 = rec_5
                best_ndcg5 = ndcg_5
                best_mrr5 = mrr_5
                best_rec1 = rec_1
                best_epoch = ep
            print("")

            if ep >= best_epoch + 30:
                break

    print("epoch: %4d, HR@1: %.6f, NDCG@1: %.6f, MRR: %.6f " % (
        best_epoch, best_rec1, best_rec1, best_rec1), end='\n')
    print("epoch: %4d, HR@5: %.6f, NDCG@5: %.6f, MRR: %.6f " % (
        best_epoch, best_rec5, best_ndcg5, best_mrr5), end='\n')
    print("epoch: %4d, HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f " % (
        best_epoch, best_rec10, best_ndcg10, best_mrr10), end='\n')
