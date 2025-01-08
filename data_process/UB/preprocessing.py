import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    data_directory = 'data'
    df = pd.read_csv(os.path.join(data_directory, 'UserBehavior.csv'), header=None)
    df.columns = ['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
    df = df.drop(columns=['category_id'])
    # https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
    # df['month_day'] = df['time_stamp'].apply(lambda x: x.strftime('%m-%d'))
    # df = df.drop(df[df['behavior'].isin(['cart', 'fav'])].index)

    df.drop_duplicates(subset=['user_id', 'item_id', 'behavior'], keep='first', inplace=True)

    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))

    df['is_buy'] = df['behavior'].map(lambda x: 1 if x == 'buy' else 0)
    df['valid_item'] = df.item_id.map(df.groupby('item_id')['is_buy'].sum() >= 10)
    df = df.loc[df.valid_item].drop('valid_item', axis=1)
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))
    # delete users without purchase
    df = df[df.user_id.isin(df[df['behavior']=='buy'].user_id.unique())]
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))

    df['valid_session'] = df.user_id.map(df[df['behavior'] == 'buy'].groupby('user_id')['item_id'].size() >= 5)
    df = df.loc[df.valid_session].drop('valid_session', axis=1)
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))
    # last_two_buy is the last second buy in df form
    last_two_buy = df[df['behavior'] == 'buy'].groupby(['user_id']).tail(2).groupby(['user_id']).head(1)
    last_one_buy = df[df['behavior'] == 'buy'].groupby(['user_id']).tail(2).groupby(['user_id']).tail(1)
    last_two_buy_map = dict(zip(last_two_buy['user_id'].values.tolist(), last_two_buy['timestamp'].values.tolist()))
    last_one_buy_map = dict(zip(last_one_buy['user_id'].values.tolist(), last_one_buy['timestamp'].values.tolist()))

    df['last_two_buy_time'] = df['user_id'].map(last_two_buy_map)
    df['last_one_buy_time'] = df['user_id'].map(last_one_buy_map)
    # delete the records of valid and test items
    last_two_buy_item_map = dict(
        zip(last_two_buy['user_id'].values.tolist(), last_two_buy['item_id'].values.tolist()))
    last_one_buy_item_map = dict(
        zip(last_one_buy['user_id'].values.tolist(), last_one_buy['item_id'].values.tolist()))
    df['last_two_buy_item'] = df['user_id'].map(last_two_buy_item_map)
    df['last_one_buy_item'] = df['user_id'].map(last_one_buy_item_map)
    df = df.groupby(['user_id']).apply(lambda x: x[~((x['behavior'] != 'buy') & (x['item_id'] == x['last_two_buy_item']))]).reset_index(drop=True)
    print(df)
    df = df.groupby(['user_id']).apply(lambda x: x[~((x['behavior'] != 'buy') & (x['item_id'] == x['last_one_buy_item']))]).reset_index(drop=True)
    print(df)
    ###################################################
    df_between_val_test = df.groupby(['user_id']).apply(
        lambda x: x[((x['behavior'] != 'buy') & (x['timestamp'] > x['last_two_buy_time']) & (x['timestamp'] < x['last_one_buy_time']))]).reset_index(drop=True)
    print(df_between_val_test)
    df = df.groupby(['user_id']).apply(lambda x: x[~((x['behavior'] != 'buy') & (x['timestamp'] > x['last_two_buy_time']))]).reset_index(drop=True)
    print(df)
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))
    df = df.groupby(['user_id']).tail(53)
    print(df.groupby(['user_id'])['item_id'].size().mean())
    # df = df.drop(columns=['last_two_buy_time', 'is_buy'])
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 'pv']),
        len(df[df['behavior'] == 'cart']),
        len(df[df['behavior'] == 'fav']),
        len(df[df['behavior'] == 'buy']),
    ))
    # 先把验证集和测试集间的堆叠起来 过完id后再分开
    df_len = df.shape[0]
    df = pd.concat([df, df_between_val_test], axis=0)
    df = df.drop(columns=['last_two_buy_time', 'last_one_buy_time', 'is_buy', 'last_two_buy_item', 'last_one_buy_item'])

    item_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df.user_id) + 1
    df['item_id'] = item_encoder.fit_transform(df.item_id) + 1

    behavior_encoder = LabelEncoder()
    df['behavior'] = behavior_encoder.fit_transform(df.behavior)
    # 0 is purchase, 1 is cart, 2 is favorite, 3 is view
    df_between_val_test = df.iloc[df_len:]
    df = df.iloc[:df_len]
    print("#user:{},#item:{},view:{},cart:{},fav:{},purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df[df['behavior'] == 3]),
        len(df[df['behavior'] == 1]),
        len(df[df['behavior'] == 2]),
        len(df[df['behavior'] == 0]),
    ))

    # ========================================================
    # split data into test set, valid set and train set,
    # adopting the leave-one-out evaluation for next-item recommendation task
    # ========================================
    # obtain possible records in test set
    df_test = df.groupby(['user_id']).tail(1)
    df.drop(df_test.index, axis='index', inplace=True)

    # ========================================
    # obtain possible records in valid set
    df_valid = df.groupby(['user_id']).tail(1)
    df.drop(df_valid.index, axis='index', inplace=True)

    # ========================================
    # drop cold-start items in valid set and test set
    df_valid = df_valid[df_valid.item_id.isin(df.item_id)] # 去掉训练集中不存在的物品
    df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (  # 跟验证集同样的用户,并且去调训练集和验证集上不存在的物品
        df_test.item_id.isin(df.item_id) | df_test.item_id.isin(df_valid.item_id))]

    processed_file_prefix = "processed_data/UB_"
    # output data file
    df_valid.to_csv(processed_file_prefix + "valid.csv", header=False, index=False)
    df_test.to_csv(processed_file_prefix + "test.csv", header=False, index=False)
    df.to_csv(processed_file_prefix + "train.csv", header=False, index=False)
    df_between_val_test.to_csv(processed_file_prefix + "between_val_test.csv", header=False, index=False)

# ========================================================
    # For each user, randomly sample some negative items,
    # and rank these items with the ground-truth item when testing or validation
    df_concat = pd.concat([df, df_valid, df_test, df_between_val_test], axis='index')
    # print(df_concat)
    sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
    # print(sr_user2items)
    df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})
    # print(df_negative)

    # ========================================
    # sample according to popularity
    sr_item2pop = df_concat.item_id.value_counts(sort=True, ascending=False)
    arr_item = sr_item2pop.index.values
    arr_pop = sr_item2pop.values


    def get_negative_sample(pos):
        neg_idx = ~np.in1d(arr_item, pos)
        neg_item = arr_item[neg_idx]
        neg_pop = arr_pop[neg_idx]
        neg_pop = neg_pop / neg_pop.sum()

        return np.random.choice(neg_item, size=100, replace=False, p=neg_pop)


    arr_sample = df_negative.user_id.apply(
        lambda x: get_negative_sample(sr_user2items[x])).values

    # output negative data
    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
    df_negative.to_csv(processed_file_prefix + "negative.csv", header=False, index=False)

