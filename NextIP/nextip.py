import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.array_ops import boolean_mask
from tensorflow.python.ops.array_ops import sequence_mask
from tensorflow.python.ops.array_ops import _all_dimensions


class Model(object):
    def __init__(self, num_user, num_item, args):
        # ==== some configurations ====
        self.num_user = num_user
        self.num_item = num_item
        self.maxlen = args.maxlen
        self.num_units = args.hidden_units
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.l2_reg = args.l2_reg
        # ==== input batch data ====
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.input_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.pos_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.neg_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.behavior_id = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.target_behavior = tf.placeholder(tf.int32, shape=(None, self.maxlen))
        self.user_id = tf.placeholder(tf.int32, shape=(None))
        padding_mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_id, 0)), -1)
        self.purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 0)), -1)  # mask purchase
        self.cart_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 1)), -1)  # mask cart
        self.favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 2)), -1)  # mask favorite
        self.view_mask = tf.expand_dims(tf.to_float(tf.equal(self.behavior_id, 3)), -1)  # mask view
        # ==== user embedding matrix ====
        with tf.variable_scope("user_embeddings"):
            self.user_emb_mat_1 = tf.get_variable('lookup_table1',
                                                  dtype=tf.float32,
                                                  shape=[self.num_user + 1, self.num_units],
                                                  regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            user_embs_1 = tf.nn.embedding_lookup(self.user_emb_mat_1, self.user_id)  # [bs, 1, d]

        # ==== item embedding matrix ====
        with tf.variable_scope("item_embeddings"):
            # items are indexed from 1 so that the max row is num_item+1
            self.item_emb_mat = tf.get_variable('lookup_table',
                                                dtype=tf.float32,
                                                shape=[self.num_item, self.num_units],
                                                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.item_emb_mat = tf.concat((tf.zeros(shape=[1, self.num_units]), self.item_emb_mat), 0) * (
                        self.num_units ** 0.5)
            item_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.input_id)

        # ==== position embedding matrix ====
        with tf.variable_scope("position_embeddings"):
            pos_emb_mat = tf.get_variable('lookup_table',
                                          dtype=tf.float32,
                                          shape=[self.maxlen, self.num_units],
                                          regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            pos_embs = tf.nn.embedding_lookup(pos_emb_mat, tf.range(self.maxlen))

        # ==== Add positional embeddings, then layer normalize and perform dropout. ====
        with tf.variable_scope("input_matrix"):
            inputs = item_embs + pos_embs
            inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            inputs = inputs * padding_mask
            inputs = self.layer_normalization(inputs)

        # ==== (behavior-aware self attention) ====
        loc_embs_ub = inputs
        attention_mask_ub = self.create_attention_mask(padding_mask=padding_mask)
        if args.L > 0:
            attention_mask_ub = tf.where(tf.greater(tf.cumsum(attention_mask_ub, axis=-1, reverse=True), args.L),
                                         x=tf.zeros_like(attention_mask_ub), y=attention_mask_ub)
        loc_embs_ub, _ = self.multihead_attention_ub(loc_embs_ub,
                                                     attention_mask=attention_mask_ub,
                                                     num_heads=args.num_heads,
                                                     dropout_rate=args.dropout_rate,
                                                     score='scaled_dot',
                                                     causality=True,
                                                     get_att='lastq_ave',
                                                     residual=True)
        loc_embs_ub *= padding_mask
        memory0, memory1, memory2, memory3 = self.get_ubp_memory(loc_embs_ub)  # [bs, L, d]

        # ==== item sequence output initialization ====
        loc_embs = tf.zeros([tf.shape(inputs)[0], self.maxlen, self.num_units])
        # ==== local representation (behavior-agnostic item sequence) ====
        if 'l' in args.ext_modules:
            loc_embs = inputs
            for b in range(args.num_blocks):
                with tf.variable_scope("self-attention_blocks_%d" % b):
                    loc_embs, laststep_attention = self.multihead_attention(loc_embs,
                                                                            padding_mask=padding_mask,
                                                                            num_heads=args.num_heads,
                                                                            dropout_rate=args.dropout_rate,
                                                                            score='scaled_dot',
                                                                            causality=True,
                                                                            get_att='lastq_ave',
                                                                            residual=True)
                    loc_embs = self.layer_normalization(loc_embs)
                    loc_embs = self.feed_forward(loc_embs)
                    loc_embs *= padding_mask
                    loc_embs = self.layer_normalization(loc_embs)

        # ==== behavior embedding matrix ====
        with tf.variable_scope("behavior_embeddings"):
            self.beha_emb_mat = tf.get_variable('lookup_table',
                                                dtype=tf.float32,
                                                shape=[args.num_behavior + 1, self.num_units],
                                                regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            target_beha_emb = tf.nn.embedding_lookup(self.beha_emb_mat, self.target_behavior)  # [bs, L, d]

        # ==== behavior-specific context information (behavior-agnostic item sequence) ====
        attention_mask_b = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='b')
        loc_embs_b = inputs * self.purchase_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("buy-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_b, laststep_attention_b = self.multihead_attention_ub(loc_embs_b,
                                                                               attention_mask=attention_mask_b,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_b = self.layer_normalization(loc_embs_b)
                loc_embs_b = self.feed_forward(loc_embs_b)
                loc_embs_b *= self.purchase_mask
                loc_embs_b = self.layer_normalization(loc_embs_b)
                loc_embs_b *= self.purchase_mask

        attention_mask_c = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='c')
        loc_embs_c = inputs * self.cart_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("chart-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_c, laststep_attention_c = self.multihead_attention_ub(loc_embs_c,
                                                                               attention_mask=attention_mask_c,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_c = self.layer_normalization(loc_embs_c)
                loc_embs_c = self.feed_forward(loc_embs_c)
                loc_embs_c *= self.cart_mask
                loc_embs_c = self.layer_normalization(loc_embs_c)
                loc_embs_c *= self.cart_mask

        attention_mask_f = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='f')
        loc_embs_f = inputs * self.favorite_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("fav-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_f, laststep_attention_f = self.multihead_attention_ub(loc_embs_f,
                                                                               attention_mask=attention_mask_f,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_f = self.layer_normalization(loc_embs_f)
                loc_embs_f = self.feed_forward(loc_embs_f)
                loc_embs_f *= self.favorite_mask
                loc_embs_f = self.layer_normalization(loc_embs_f)
                loc_embs_f *= self.favorite_mask

        attention_mask_v = self.create_behavior_attention_mask(padding_mask=padding_mask, causality=True,
                                                               behavior='v')
        loc_embs_v = inputs * self.view_mask
        for b in range(args.num_blocks_behavior):
            with tf.variable_scope("view-self-attention_blocks_%d" % b):
                # multi-head attention
                loc_embs_v, laststep_attention_v = self.multihead_attention_ub(loc_embs_v,
                                                                               attention_mask=attention_mask_v,
                                                                               num_heads=self.num_heads,
                                                                               dropout_rate=self.dropout_rate,
                                                                               score='scaled_dot',
                                                                               causality=True,
                                                                               get_att='lastq_ave',
                                                                               residual=True)
                loc_embs_v = self.layer_normalization(loc_embs_v)
                loc_embs_v = self.feed_forward(loc_embs_v)
                loc_embs_v *= self.view_mask
                loc_embs_v = self.layer_normalization(loc_embs_v)
                loc_embs_v *= self.view_mask

        loc_embs_multiview = loc_embs_b + loc_embs_c + loc_embs_f + loc_embs_v  # [bs, L, d]
        context_b, context_c, context_f, context_v = self.get_ubp_memory(loc_embs_multiview)
        self.wo = args.wo
        target_view = self.get_target_view(context_b, context_c, context_f, context_v)
        # for test
        target_view_test = tf.expand_dims(context_b[:, -1, :], 1)  # [bs, 1, d]

        with tf.variable_scope("gating_parameters"):
            if args.dataset == 'Tmall':
                if self.wo == 'v':
                    stacked = tf.concat([context_b, context_f], axis=1)  # [bs, 2, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 2, self.num_units])
                elif self.wo == 'b':
                    stacked = tf.concat([context_f, context_v], axis=1)  # [bs, 2, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 2, self.num_units])
                elif self.wo == 'f':
                    stacked = tf.concat([context_b, context_v], axis=1)  # [bs, 2, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 2, self.num_units])
                else:
                    stacked = tf.concat([context_b, context_f, context_v], axis=-1)  # [bs, L, 3d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 3, self.num_units])  # [bs, L, 4, d]
            else:
                if self.wo == 'v':
                    stacked = tf.concat([context_b, context_c, context_f], axis=1)  # [bs, 3, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 3, self.num_units])
                elif self.wo == 'b':
                    stacked = tf.concat([context_c, context_f, context_v], axis=1)  # [bs, 3, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 3, self.num_units])
                elif self.wo == 'c':
                    stacked = tf.concat([context_b, context_f, context_v], axis=1)  # [bs, 3, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 3, self.num_units])
                elif self.wo == 'f':
                    stacked = tf.concat([context_b, context_c, context_v], axis=1)  # [bs, 3, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 3, self.num_units])
                else:
                    stacked = tf.concat([context_b, context_c, context_f, context_v], axis=-1)  # [bs, 4, d]
                    stacked = tf.reshape(stacked, [-1, self.maxlen, 4, self.num_units])
            # TBCG
            # Use the target behavior embedding as the query to aggregate the context of different view
            context = self.dynamic_behavior_attention(query=target_beha_emb, inputs=stacked,
                                                      dropout_rate=args.dropout_rate)
            context *= padding_mask  # [bs, L, d]
            test_beha_emb = tf.tile(tf.reshape(self.beha_emb_mat[0], [1, 1, self.num_units]),
                                    [tf.shape(self.input_id)[0], self.maxlen, 1])  # [bs, L, d]
            context_test = self.dynamic_behavior_attention(query=test_beha_emb, inputs=stacked,
                                                           dropout_rate=args.dropout_rate)
            context_test = tf.expand_dims(context_test[:, -1, :], 1)
            outputs = loc_embs
            outputs = target_view * 0.5 + context * 0.5 + loc_embs
            outputs = self.layer_normalization(outputs)
            outputs_te = tf.expand_dims(loc_embs[:, -1, :], 1)
            outputs_te = target_view_test * 0.5 + context_test * (
                    1 - 0.5) + tf.expand_dims(loc_embs[:, -1, :], 1)
            outputs_te = self.layer_normalization(outputs_te)

        # ==== ref. MBSRec (RecSys'24): weighted binary cross-entropy loss ====
        with tf.variable_scope("behavior_weighted"):
            if args.bw == 'static':
                behavior_weight_map = tf.constant([[0.75], [0.1], [0.1], [0.1], [0.0]], dtype=tf.float32)
            elif args.bw == 'dynamic':
                behavior_weight_map = tf.get_variable('lookup_table',
                                                    dtype=tf.float32,
                                                    shape=[args.num_behavior + 1, 1],
                                                    initializer=tf.constant_initializer(1.0),
                                                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            else:
                behavior_weight_map = tf.constant([[1.0], [1.0], [1.0], [1.0], [0.0]], dtype=tf.float32)
            behavior_weight = tf.nn.embedding_lookup(behavior_weight_map, self.target_behavior)
            behavior_weight = tf.sigmoid(behavior_weight)
            behavior_weight = tf.squeeze(behavior_weight, -1)

        # ==== training loss normal item2item ====
        outputs_pos = outputs
        outputs_neg = outputs
        positive_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.pos_id)
        negative_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.neg_id)
        positive_rating = tf.reduce_sum(outputs_pos * positive_embs, -1)
        negative_rating = tf.reduce_sum(outputs_neg * negative_embs, -1)
        flag_exist = tf.to_float(tf.not_equal(self.pos_id, 0))
        loss_normal = tf.reduce_sum(- tf.log(tf.sigmoid(positive_rating) + 1e-24) * flag_exist * behavior_weight
                                    - tf.log(1 - tf.sigmoid(negative_rating) + 1e-24) * flag_exist)

        # ==== training contrastive loss for target behaviors ====
        purchase_mask = tf.to_float(tf.equal(self.target_behavior, 0))  # [bs,L]
        user_c_emb = memory1
        user_f_emb = memory2
        user_v_emb = memory3
        user_embs_1 = tf.reshape(user_embs_1, [-1, self.num_units])
        chart_emb = self.gather_indexes(user_c_emb, positions_mask=purchase_mask)
        fav_emb = self.gather_indexes(user_f_emb, positions_mask=purchase_mask)
        view_emb = self.gather_indexes(user_v_emb, positions_mask=purchase_mask)
        pos_buy = self.gather_indexes(positive_embs, positions_mask=purchase_mask)  # [num_buys, d]
        pos_final = tf.reshape(self.repeat(user_embs_1, mask=purchase_mask), [-1, self.num_units])
        self.final_w = tf.Variable(tf.truncated_normal(shape=[self.num_units, self.num_units],
                                                       mean=0.0,
                                                       stddev=tf.sqrt(tf.div(2.0, 3 * self.num_units + 1))),
                                   name='weights_for_rating',
                                   dtype=tf.float32)
        pos_final = tf.matmul(pos_final, self.final_w)
        self.pos_logits = tf.reshape(tf.reduce_sum(pos_final * pos_buy, -1), [-1, 1])  # [num_buys, 1]
        self.neg_c = tf.reshape(tf.reduce_sum(pos_final * chart_emb, -1), [-1, 1])  # [num_buys, 1]
        self.neg_f = tf.reshape(tf.reduce_sum(pos_final * fav_emb, -1), [-1, 1])
        self.neg_v = tf.reshape(tf.reduce_sum(pos_final * view_emb, -1), [-1, 1])

        if args.dataset == 'Tmall':
            all_logits = tf.concat([tf.expand_dims(self.pos_logits, 2),  # [num_buys, 1, 1]
                                    tf.expand_dims(self.neg_f, 2),
                                    tf.expand_dims(self.neg_v, 2)],
                                   2)  # [num_buys, 1, 3]
        else:
            all_logits = tf.concat([tf.expand_dims(self.pos_logits, 2),  # [num_buys, 1, 1]
                                    tf.expand_dims(self.neg_c, 2), tf.expand_dims(self.neg_f, 2),
                                    tf.expand_dims(self.neg_v, 2)],
                                   2)  # [num_buys, 1, 4]
        temperature = 0.07
        log_prob = tf.nn.log_softmax(all_logits / temperature, -1)[:, :, 0]  # [num_buys, 1]
        self.loss_contrastive = -tf.reduce_mean(tf.reduce_sum(log_prob, 1), 0)
        # ==== final loss ====
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = loss_normal / tf.reduce_sum(flag_exist)
        self.loss += sum(reg_loss)
        self.loss += self.loss_contrastive
        # evaluate on sampled item set
        self.candidate_id = tf.placeholder(tf.int32, shape=(None, 101))
        candidate_embs = tf.nn.embedding_lookup(self.item_emb_mat, self.candidate_id)
        test_emb = outputs_te
        self.cand_rating = tf.reduce_sum(test_emb * candidate_embs, -1)  # [bs,item_num]
        user_embs_1 = tf.expand_dims(user_embs_1, 1)
        user_rating = tf.reduce_sum(user_embs_1 * candidate_embs, -1)
        self.cand_rating = self.cand_rating + user_rating
        # ==== optimizer ====
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    # ================================================================
    # ======== multihead attention ========
    def multihead_attention(self, inputs, padding_mask=None,
                            num_heads=2, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.variable_scope("multihead_attention"):
            # linear projections, shape: [batch_size, seq_length, num_units]
            if score == 'location':
                weights = tf.get_variable('weights',
                                     dtype=tf.float32,
                                     shape=[1, self.num_units],
                                     regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                if causality == False: # obtain output shape: [batch_size, 1, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], 1, 1])
                else: # obtain output shape: [batch_size, seq_length, num_units]
                    Q = tf.tile(tf.expand_dims(weights, 0), [tf.shape(inputs)[0], tf.shape(inputs)[1], 1])
            else:
                Q = tf.layers.dense(inputs, self.num_units, activation=None)

            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)

            # split and place in parallel, shape: [batch_size * num_heads, seq_length, num_units / num_heads]
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # score function, shape: [batch_size * num_heads, seq_length_Q, seq_length_K]
            if score == 'scaled_dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)
            # elif score == 'global' or score == 'dot':
            elif score in 'location' or score == 'dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1]))

            # causality masking
            if causality == True:
                tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(outputs[0, :, :])).to_dense()
                # #v for tf.__version__='1.2.1'
                # tril = tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(outputs[0, :, :])).to_dense()
                causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                outputs = tf.where(tf.equal(causality_mask, 0), tf.ones_like(outputs)*(-2**32+1), outputs)

            # key masking
            if padding_mask is not None:
                key_mask = tf.transpose(tf.tile(padding_mask, [num_heads, 1, tf.shape(Q)[1]]), [0, 2, 1])
                outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs)*(-2**32+1), outputs)

            # softmax normalization
            outputs = tf.nn.softmax(outputs)

            # query masking
            if (score != 'location') and (padding_mask is not None):
                query_mask = tf.tile(padding_mask, [num_heads, 1, tf.shape(inputs)[1]])
                outputs = outputs * query_mask

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # return attention for visualization
            if get_att == 'lastq_ave':
                attention = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            if get_att == 'lastq_multi':
                attention = tf.split(outputs[:, -1], num_heads, axis=0)
            elif get_att == 'batch_multi':
                attention = outputs

            # weighted sum, shape: [batch_size * num_heads, seq_length_Q, num_units / num_heads]
            outputs = tf.matmul(outputs, Vh)

            # concatenate different heads, shape: [batch_size, seq_length_Q, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        if residual == True:
            outputs += inputs

        return outputs, attention

    # ======== feed forward ========
    def feed_forward(self, inputs,
                     inner_units=None, dropout_rate=None):
        if inner_units is None:
            inner_units = self.num_units
        if dropout_rate is None:
            dropout_rate = self.dropout_rate

        with tf.variable_scope("feed_forward"):
            # inner layer
            params = {"inputs": inputs, "filters": inner_units, "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # readout layer
            params = {"inputs": outputs, "filters": self.num_units, "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        outputs += inputs

        return outputs

    # ======== layer normalization ========
    def layer_normalization(self, inputs, epsilon=1e-8):
        with tf.variable_scope("layer_normalization"):
            alpha = tf.Variable(tf.ones(self.num_units))
            beta = tf.Variable(tf.zeros(self.num_units))

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
            outputs = alpha * normalized + beta

        return outputs

    def gather_indexes(self, sequence_tensor, positions_mask):
        """Gathers the vectors at the specific positions over a minibatch."""
        position = tf.where(tf.equal(positions_mask, 1))
        output_tensor = tf.gather_nd(sequence_tensor, position)
        return output_tensor

    def repeat(self, input, mask):
        """Gather tensor in a repeated way."""
        repeats = tf.to_float(tf.equal(mask, 1))
        repeats = tf.reduce_sum(repeats, axis=1)
        repeats = tf.cast(repeats, tf.int32)
        max_repeat = gen_math_ops.maximum(
            0, gen_math_ops._max(repeats, _all_dimensions(repeats)))
        mask = sequence_mask(repeats, max_repeat)
        expanded = tf.expand_dims(input, 1)
        multiples = [1] * expanded.shape.ndims
        multiples[1] = max_repeat
        tiled = tf.tile(expanded, multiples)
        return boolean_mask(tiled, mask)

    # ================================================================
    # ======== multihead attention ========
    def multihead_attention_ub(self, inputs, attention_mask=None,
                            num_heads=2, dropout_rate=None, score='scaled_dot',
                            causality=True, get_att='last_ave', residual=True):
        with tf.variable_scope("multihead_attention"):
            # linear projections, shape: [batch_size, seq_length+4, num_units]
            Q = tf.layers.dense(inputs, self.num_units, activation=None)
            K = tf.layers.dense(inputs, self.num_units, activation=None)
            V = tf.layers.dense(inputs, self.num_units, activation=None)

            # split and place in parallel, shape: [batch_size * num_heads, seq_length, num_units / num_heads]
            Qh = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            Kh = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            Vh = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # score function, shape: [batch_size * num_heads, seq_length_Q, seq_length_K]
            if score == 'scaled_dot':
                outputs = tf.matmul(Qh, tf.transpose(Kh, [0, 2, 1])) / (tf.to_float(tf.shape(Kh))[-1] ** 0.5)

            outputs = tf.where(tf.equal(attention_mask, 0), tf.ones_like(outputs) * (-2 ** 32 + 1), outputs)

            outputs = tf.nn.softmax(outputs)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

            # return attention for visualization
            if get_att == 'lastq_ave':
                attention = tf.reduce_mean(tf.split(outputs[:, -1], num_heads, axis=0), axis=0)
            if get_att == 'lastq_multi':
                attention = tf.split(outputs[:, -1], num_heads, axis=0)
            elif get_att == 'batch_multi':
                attention = outputs

            # weighted sum, shape: [batch_size * num_heads, seq_length_Q, num_units / num_heads]
            outputs = tf.matmul(outputs, Vh)

            # concatenate different heads, shape: [batch_size, seq_length_Q, num_units]
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))

        # residual connection
        if residual == True:
            outputs += inputs

        return outputs,attention

    def create_attention_mask(self, padding_mask=None, causality=True):
        """Create 3D attention mask from a 2D tensor mask.
        """
        batch_size = tf.shape(padding_mask)[0]
        seq_length = tf.shape(padding_mask)[1]
        to_mask = tf.cast(
            tf.reshape(padding_mask, [batch_size, 1, seq_length]), tf.float32)

        broadcast_ones = tf.ones(
            shape=[batch_size, seq_length, 1], dtype=tf.float32)

        mask = broadcast_ones * to_mask  # [bs,L,L]
        # causality masking
        if causality == True:
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(mask[0, :, :])).to_dense()
            # #v for tf.__version__='1.2.1'
            # tril = tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(outputs[0, :, :])).to_dense()
            mask_causality = tf.tile(tf.expand_dims(tril, 0), [batch_size, 1, 1])
            mask = mask * mask_causality

        purchase_mask = tf.cast(
            tf.reshape(self.purchase_mask, [batch_size, 1, self.maxlen]), tf.float32)
        cart_mask = tf.cast(
            tf.reshape(self.cart_mask, [batch_size, 1, self.maxlen]), tf.float32)
        favorite_mask = tf.cast(
            tf.reshape(self.favorite_mask, [batch_size, 1, self.maxlen]), tf.float32)
        view_mask = tf.cast(
            tf.reshape(self.view_mask, [batch_size, 1, self.maxlen]), tf.float32)
        padding = tf.zeros_like(view_mask)
        user_behavior_mask = tf.concat([purchase_mask, cart_mask, favorite_mask, view_mask, padding], axis=-2)

        behavior_id = tf.reshape(self.behavior_id, [-1])
        batch_len_index = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1]), [1, seq_length]), [-1])
        beha_index = tf.stack([batch_len_index, behavior_id], axis=1)
        user_mask = tf.gather_nd(user_behavior_mask, beha_index)
        user_mask = tf.reshape(user_mask, [-1, seq_length, seq_length])
        mask = mask * user_mask

        return mask

    def create_behavior_attention_mask(self, padding_mask=None, causality=True, behavior='v'):
        """
        Create 3D attention mask from a 2D tensor mask.
        v: view
        b: buy
        c: chart
        f: favor
        """
        batch_size = tf.shape(padding_mask)[0]
        seq_length = tf.shape(padding_mask)[1]
        to_mask = tf.cast(
            tf.reshape(padding_mask, [batch_size, 1, seq_length]), tf.float32)

        broadcast_ones = tf.ones(
            shape=[batch_size, seq_length, 1], dtype=tf.float32)

        mask = broadcast_ones * to_mask  # [bs,L,L]
        # causality masking
        if causality == True:
            tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(mask[0, :, :])).to_dense()
            # #v for tf.__version__='1.2.1'
            # tril = tf.contrib.linalg.LinearOperatorTriL(tf.ones_like(outputs[0, :, :])).to_dense()
            mask_causality = tf.tile(tf.expand_dims(tril, 0), [batch_size, 1, 1])
            mask = mask * mask_causality

        behavior_mask = None
        if behavior == 'v':
            behavior_mask = tf.cast(
                tf.reshape(self.view_mask, [batch_size, 1, self.maxlen]), tf.float32)
        if behavior == 'b':
            behavior_mask = tf.cast(
                tf.reshape(self.purchase_mask, [batch_size, 1, self.maxlen]), tf.float32)
        if behavior == 'c':
            behavior_mask = tf.cast(
                tf.reshape(self.cart_mask, [batch_size, 1, self.maxlen]), tf.float32)
        if behavior == 'f':
            behavior_mask = tf.cast(
                tf.reshape(self.favorite_mask, [batch_size, 1, self.maxlen]), tf.float32)

        mask = mask * behavior_mask

        return mask

    def get_target_view(self, a, b, c, d):
        target_purchase_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 0)), -1)  # mask purchase
        target_cart_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 1)), -1)  # mask cart
        target_favorite_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 2)), -1)  # mask favorite
        target_view_mask = tf.expand_dims(tf.to_float(tf.equal(self.target_behavior, 3)), -1)  # mask view

        a *= target_purchase_mask
        b *= target_cart_mask
        c *= target_favorite_mask
        d *= target_view_mask
        if self.wo == 'v':
            output = a + b + c
        elif self.wo == 'b':
            output = b + c + d
        elif self.wo == 'f':
            output = a + b + d
        else:
            output = a + b + c + d
        return output

    def get_ubp_memory(self, self_att_output):
        """Gather dynamic user_behavior preference or context information in a form of memory matrix"""
        batch_size = tf.shape(self_att_output)[0]
        padding = tf.expand_dims(tf.zeros([1, self.num_units]), 1)
        padding = tf.tile(padding, [batch_size, 1, 1])
        tensor = tf.concat([padding, self_att_output], -2)  # [bs,L+1,d]

        mask0 = tf.to_float(tf.equal(self.behavior_id, 0))  # mask purchase [bs, L]
        mask1 = tf.to_float(tf.equal(self.behavior_id, 1))  # mask cart
        mask2 = tf.to_float(tf.equal(self.behavior_id, 2))  # mask favorite
        mask3 = tf.to_float(tf.equal(self.behavior_id, 3))  # mask view

        def get_one_memory(mask):
            pad_mask = tf.ones([batch_size, 1])
            mask_with_pad = tf.concat([pad_mask, mask], -1)
            idx_bias = tf.reshape(tf.reduce_sum(mask, -1), [-1]) + 1  # [bs,1]
            idx_bias = tf.reshape(tf.cast(tf.cumsum(idx_bias, axis=-1, exclusive=True), tf.int32),
                                  [batch_size, 1])  # [bs,1]
            mask = tf.cast(tf.cumsum(mask, axis=-1), tf.int32)  # [bs, L]
            mask = mask + idx_bias  # [bs, L]
            position = tf.where(tf.equal(mask_with_pad, 1))
            embedding = tf.gather_nd(tensor, position)
            return tf.gather(embedding, mask)

        outputs0 = get_one_memory(mask0)
        outputs1 = get_one_memory(mask1)
        outputs2 = get_one_memory(mask2)
        outputs3 = get_one_memory(mask3)
        return outputs0, outputs1, outputs2, outputs3

    def dynamic_behavior_attention(self, query, inputs, dropout_rate=None):
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        with tf.variable_scope("behavior_attention", reuse=tf.AUTO_REUSE):
            query = tf.reshape(query, [-1, self.maxlen, 1, self.num_units])
            Q = tf.layers.dense(query, self.num_units, activation=None)    # [batch_size, seq_len, 1, num_units]
            K = tf.layers.dense(inputs, self.num_units, activation=None)   # [batch_size, seq_len, num_behavior, num_units]
            V = tf.layers.dense(inputs, self.num_units, activation=None)   # [batch_size, seq_len, num_behavior, num_units]
            outputs = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / (tf.to_float(tf.shape(K))[-1] ** 0.5)
            # softmax normalization
            outputs = tf.nn.softmax(outputs)
            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.matmul(outputs, V)
            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.reshape(outputs, [-1, self.maxlen, self.num_units])
        return outputs