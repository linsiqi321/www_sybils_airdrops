import pandas as pd
import numpy as np
import time
import copy

# init data
invited_data = pd.read_csv('data/invited_data.csv')

child_dict = {}
parent_dict = {}
depth = {}
max_descendant_depth = {}
subtree_size = {}
subtree_width = {}
root = []
t = time.time()
empty_in_parent_column = invited_data['invited_by'].isna()
for i in range(0, len(invited_data)):
    depth[invited_data['address'][i]] = 0
    max_descendant_depth[invited_data['address'][i]] = 0
    if not empty_in_parent_column[i]:
        depth[invited_data['invited_by'][i]] = 0
        max_descendant_depth[invited_data['invited_by'][i]] = 0
        parent_dict[invited_data['address'][i]] = invited_data['invited_by'][i]
        if child_dict.get(invited_data['invited_by'][i]) is None:
            child_dict[invited_data['invited_by'][i]] = []
            child_dict[invited_data['invited_by'][i]].append(invited_data['address'][i])
        else:
            child_dict[invited_data['invited_by'][i]].append(invited_data['address'][i])
    else:
        root.append(invited_data['address'][i])
print('init time:', time.time() - t)

# calculate depth
t = time.time()
leaf = []
for i in range(0, len(root)):
    in_list = []
    in_list.append(root[i])
    while len(in_list) > 0:
        node = in_list.pop(0)
        if child_dict.get(node) is not None:
            for child in child_dict[node]:
                in_list.append(child)
                depth[child] = depth[node] + 1
        else:
            leaf.append(node)
print('depth time:', time.time() - t)

# calculate max_descendant_depth
t = time.time()
for i in range(0, len(leaf)):
    node = leaf[i]
    max_descendant_depth[node] = depth[node]
    while parent_dict.get(node) is not None:
        parent = parent_dict[node]
        if max_descendant_depth[parent] < max_descendant_depth[node]:
            max_descendant_depth[parent] = max_descendant_depth[node]
        else:
            break
        node = parent
print('max_descendant_depth time:', time.time() - t)

# calculate subtree_size
t = time.time()
in_list = []
out_dict = {}
for i in range(0, len(leaf)):
    in_list.append(leaf[i])
    subtree_size[leaf[i]] = 1
    out_dict[leaf[i]] = 1
while len(in_list) > 0:
    node = in_list.pop(0)
    if parent_dict.get(node) is not None:
        parent = parent_dict[node]
        if out_dict.get(parent) is None:
            out_dict[parent] = 1
            in_list.append(parent)
            subtree_size[parent] = 1
        subtree_size[parent] += subtree_size[node]

print('subtree_size time:', time.time() - t)

# calculate subtree_width
t = time.time()
for i in range(0, len(root)):
    in_list = []
    in_list.append(root[i])
    
    while len(in_list) > 0:
        node = in_list.pop(0)
        if child_dict.get(node) is not None:
            for child in child_dict[node]:
                in_list.append(child)
        
        current_layer = []
        next_layer = []

        current_layer.append(node)
        subtree_width[node] = 1
        while len(current_layer) > 0:
            current_width = 0
            for j in range(len(current_layer)):
                if child_dict.get(current_layer[j]) is not None:
                    for child in child_dict[current_layer[j]]:
                        next_layer.append(child)
                        current_width += 1
                    # next_layer += child_dict[current_layer[j]]
                    # current_width += len(child_dict[current_layer[j]])
            current_layer = copy.deepcopy(next_layer)
            next_layer = []
            subtree_width[node] = max(subtree_width[node], current_width)
print('subtree_width time:', time.time() - t)

# calculate family activity
activity = pd.read_csv('data/score_0811_with_effective_referral.csv')
activity['is_on_chain_interaction'] = activity['on_chain_interaction_count'].apply(lambda x: 1 if x > 0 else 0)
# activity = activity[['address', 'follow_on_x_count', 'join_telegram_count', 'join_discord_count', 'on_chain_interaction_count']]
x_dict = dict(zip(activity['address'], activity['follow_on_x_count']))
telegram_dict = dict(zip(activity['address'], activity['join_telegram_count']))
discord_dict = dict(zip(activity['address'], activity['join_discord_count']))
on_chain_dict = dict(zip(activity['address'], activity['on_chain_interaction_count']))
is_on_chain_interaction_dict = dict(zip(activity['address'], activity['is_on_chain_interaction']))

# analyze time data
invited_data_with_time = invited_data.dropna().drop_duplicates()
def parse_datetime(date_string):
    try:
        return pd.to_datetime(date_string, format='%d/%m/%Y %H:%M:%S.%f')
    except ValueError:
        try:
            return pd.to_datetime(date_string, format='%d/%m/%Y %H:%M:%S')
        except ValueError:
            print(f"无法解析日期时间: {date_string}")
            return pd.NaT

invited_data_with_time['address'] = invited_data_with_time['invited_by']
invited_data_with_time['timestamp'] = invited_data_with_time['created_at'].apply(parse_datetime)
invited_data_with_time['date'] = invited_data_with_time['timestamp'].dt.date
daily_invited_count = invited_data_with_time.groupby(['address', 'date']).agg(
    count=('timestamp', 'size'),
    first_time=('timestamp', 'min'),
    last_time=('timestamp', 'max')
).reset_index()

# daily_invited_count.to_csv('data/daily_invited_count.csv', index=False)

address_daily_stats = daily_invited_count.groupby('address').agg(
    max_daily_address_count=('count', 'max'),
    avg_daily_address_count=('count', 'mean'),
    var_daily_address_count=('count', 'var'),
)
address_daily_stats['cv_daily_address_count'] = np.sqrt(address_daily_stats['var_daily_address_count']) / address_daily_stats['avg_daily_address_count']

def max_invited_days(sub_df):
    sub_df = sub_df.sort_values(by='date')
    sub_df['diff'] = sub_df['date'].diff().dt.days
    sub_df['consecutive'] = sub_df['diff'].ne(1).cumsum()
    max_consecutive = sub_df.groupby('consecutive').size().max()
    return max_consecutive

# use groupby to calculate the maximum consecutive days for each address
daily_invited_count['date'] = pd.to_datetime(daily_invited_count['date'])
max_continous_invited_days = daily_invited_count.groupby('address').apply(max_invited_days).reset_index(name='max_continous_invited_days')

daily_invited_count['time_diff'] = (daily_invited_count['last_time'] - daily_invited_count['first_time']).dt.total_seconds() / 3600
daily_invited_count['time_diff'] = daily_invited_count['time_diff'].replace(0, pd.NA)
daily_invited_count['interval_hours'] = daily_invited_count['count'] / daily_invited_count['time_diff']
daily_invited_count['interval_hours'] = daily_invited_count['interval_hours'].fillna(0)
max_intervals = daily_invited_count.groupby('address')['interval_hours'].max().reset_index(name='max_interval_hours')
avg_intervals = daily_invited_count.groupby('address')['interval_hours'].mean().reset_index(name='avg_interval_hours')


address_daily_stats = address_daily_stats.merge(max_continous_invited_days, left_on='address', right_on='address', how='outer') \
                                         .merge(max_intervals, left_on='address', right_on='address', how='outer') \
                                         .merge(avg_intervals, left_on='address', right_on='address', how='outer')
address_daily_stats.to_csv('data/invited_data_with_time.csv', index=False)


# calculate brother number, child number, grandchild number, parent activity, child activity, parent max daily address count, parent avg daily address count, parent max continous invited days, child max daily address count, child avg daily address count, child max continous invited days
t = time.time()
brother_number = {}
child_number = {}
grandchild_number = {}

parent_x_activity = {}
parent_telegram_activity = {}
parent_discord_activity = {}
parent_on_chain_activity = {}
parent_is_on_chain_interaction_activity = {}
child_x_activity = {}
child_telegram_activity = {}
child_discord_activity = {}
child_on_chain_activity = {}
child_is_on_chain_interaction_activity = {}

parent_max_daily_address_count = {}
parent_avg_daily_address_count = {}
parent_max_continous_invited_days = {}
child_max_daily_address_count = {}
child_avg_daily_address_count = {}
child_max_continous_invited_days = {}

brother_s_avg_child_num = {}
brother_s_var_child_num = {}
brother_s_cv_child_num = {}
brother_s_child_num_over_0_count = {}
brother_s_child_num_over_3_count = {}
brother_s_child_num_over_5_count = {}
brother_s_child_num_over_10_count = {}
brother_s_child_num_over_20_count = {}

child_s_avg_child_num = {}
child_s_var_child_num = {}
child_s_cv_child_num = {}
child_s_child_num_over_0_count = {}
child_s_child_num_over_3_count = {}
child_s_child_num_over_5_count = {}
child_s_child_num_over_10_count = {}
child_s_child_num_over_20_count = {}

for i in range(0, len(invited_data)):
    current_address = invited_data['address'][i]
    if parent_dict.get(current_address) is not None:
        parent = parent_dict[current_address]
        brother_number[current_address] = len(child_dict[parent]) - 1
        if x_dict.get(parent) is not None:
            parent_x_activity[current_address] = x_dict[parent]
        if telegram_dict.get(parent) is not None:
            parent_telegram_activity[current_address] = telegram_dict[parent]
        if discord_dict.get(parent) is not None:
            parent_discord_activity[current_address] = discord_dict[parent]
        if on_chain_dict.get(parent) is not None:
            parent_on_chain_activity[current_address] = on_chain_dict[parent]
        if is_on_chain_interaction_dict.get(parent) is not None:
            parent_is_on_chain_interaction_activity[current_address] = is_on_chain_interaction_dict[parent]

        if address_daily_stats['max_daily_address_count'].get(parent) is not None:
            parent_max_daily_address_count[current_address] = address_daily_stats['max_daily_address_count'][parent]
        if address_daily_stats['avg_daily_address_count'].get(parent) is not None:
            parent_avg_daily_address_count[current_address] = address_daily_stats['avg_daily_address_count'][parent]
        if address_daily_stats['max_continous_invited_days'].get(parent) is not None:
            parent_max_continous_invited_days[current_address] = address_daily_stats['max_continous_invited_days'][parent]

        if len(child_dict[parent]) > 1:
            brother_s_child_num = []
            for child in child_dict[parent]:
                if child != current_address:
                    if child_dict.get(child) is not None:
                        brother_s_child_num.append(len(child_dict[child]))
            if len(brother_s_child_num) > 0:
                brother_s_child_num = np.array(brother_s_child_num)
                brother_s_avg_child_num[current_address] = np.mean(brother_s_child_num)
                brother_s_var_child_num[current_address] = np.var(brother_s_child_num)
                brother_s_cv_child_num[current_address] = np.sqrt(brother_s_var_child_num[current_address]) / brother_s_avg_child_num[current_address]
                brother_s_child_num_over_0_count[current_address] = len(brother_s_child_num[brother_s_child_num > 0])
                brother_s_child_num_over_3_count[current_address] = len(brother_s_child_num[brother_s_child_num > 3])
                brother_s_child_num_over_5_count[current_address] = len(brother_s_child_num[brother_s_child_num > 5])
                brother_s_child_num_over_10_count[current_address] = len(brother_s_child_num[brother_s_child_num > 10])
                brother_s_child_num_over_20_count[current_address] = len(brother_s_child_num[brother_s_child_num > 20])


    if child_dict.get(current_address) is not None:
        child_number_ = len(child_dict[current_address])
        child_number[current_address] = child_number_
        grandchild_number[current_address] = 0
        child_x_activity[current_address] = 0
        child_telegram_activity[current_address] = 0
        child_discord_activity[current_address] = 0
        child_on_chain_activity[current_address] = 0
        child_is_on_chain_interaction_activity[current_address] = 0

        child_max_daily_address_count[current_address] = 0
        child_avg_daily_address_count[current_address] = 0
        child_max_continous_invited_days[current_address] = 0

        child_s_child_num = []

        for child in child_dict[current_address]:

            if child_dict.get(child) is not None:
                grandchild_number[current_address] += len(child_dict[child])
                child_s_child_num.append(len(child_dict[child]))

            if x_dict.get(child) is not None:
                child_x_activity[current_address] += x_dict[child]
            if telegram_dict.get(child) is not None:
                child_telegram_activity[current_address] += telegram_dict[child]
            if discord_dict.get(child) is not None:
                child_discord_activity[current_address] += discord_dict[child]
            if on_chain_dict.get(child) is not None:
                child_on_chain_activity[current_address] += on_chain_dict[child]
            if is_on_chain_interaction_dict.get(child) is not None:
                child_is_on_chain_interaction_activity[current_address] += is_on_chain_interaction_dict[child]
            
            if address_daily_stats['max_daily_address_count'].get(child) is not None:
                child_max_daily_address_count[current_address] += address_daily_stats['max_daily_address_count'][child]
            if address_daily_stats['avg_daily_address_count'].get(child) is not None:
                child_avg_daily_address_count[current_address] += address_daily_stats['avg_daily_address_count'][child]
            if address_daily_stats['max_continous_invited_days'].get(child) is not None:
                child_max_continous_invited_days[current_address] += address_daily_stats['max_continous_invited_days'][child]
            
        child_x_activity[current_address] /= child_number_
        child_telegram_activity[current_address] /= child_number_
        child_discord_activity[current_address] /= child_number_
        child_on_chain_activity[current_address] /= child_number_
        child_is_on_chain_interaction_activity[current_address] /= child_number_
        child_max_daily_address_count[current_address] /= child_number_
        child_avg_daily_address_count[current_address] /= child_number_
        child_max_continous_invited_days[current_address] /= child_number_

        if len(child_s_child_num) > 0:
            child_s_child_num = np.array(child_s_child_num)
            child_s_avg_child_num[current_address] = np.mean(child_s_child_num)
            child_s_var_child_num[current_address] = np.var(child_s_child_num)
            child_s_cv_child_num[current_address] = np.sqrt(child_s_var_child_num[current_address]) / child_s_avg_child_num[current_address]
            child_s_child_num_over_0_count[current_address] = len(child_s_child_num[child_s_child_num > 0])
            child_s_child_num_over_3_count[current_address] = len(child_s_child_num[child_s_child_num > 3])
            child_s_child_num_over_5_count[current_address] = len(child_s_child_num[child_s_child_num > 5])
            child_s_child_num_over_10_count[current_address] = len(child_s_child_num[child_s_child_num > 10])
            child_s_child_num_over_20_count[current_address] = len(child_s_child_num[child_s_child_num > 20])

print('brother & child & grandchild number time:', time.time() - t)


invited_data['depth'] = invited_data['address'].apply(lambda x: depth.get(x, 0))
invited_data['max_descendant_depth'] = invited_data['address'].apply(lambda x: max_descendant_depth.get(x, 0))
invited_data['subtree_size'] = invited_data['address'].apply(lambda x: subtree_size.get(x, 0))
invited_data['subtree_width'] = invited_data['address'].apply(lambda x: subtree_width.get(x, 0))
invited_data['parent_number'] = invited_data['invited_by'].apply(lambda x: 1 if not pd.isna(x) else 0)
invited_data['brother_number'] = invited_data['address'].apply(lambda x: brother_number.get(x, 0))
invited_data['child_number'] = invited_data['address'].apply(lambda x: child_number.get(x, 0))
invited_data['grandchild_number'] = invited_data['address'].apply(lambda x: grandchild_number.get(x, 0))

invited_data['parent_x_activity'] = invited_data['address'].apply(lambda x: parent_x_activity.get(x, 0))
invited_data['parent_telegram_activity'] = invited_data['address'].apply(lambda x: parent_telegram_activity.get(x, 0))
invited_data['parent_discord_activity'] = invited_data['address'].apply(lambda x: parent_discord_activity.get(x, 0))
invited_data['parent_on_chain_activity'] = invited_data['address'].apply(lambda x: parent_on_chain_activity.get(x, 0))
invited_data['parent_is_on_chain_interaction_activity'] = invited_data['address'].apply(lambda x: parent_is_on_chain_interaction_activity.get(x, 0))
invited_data['child_x_activity'] = invited_data['address'].apply(lambda x: child_x_activity.get(x, 0))
invited_data['child_telegram_activity'] = invited_data['address'].apply(lambda x: child_telegram_activity.get(x, 0))
invited_data['child_discord_activity'] = invited_data['address'].apply(lambda x: child_discord_activity.get(x, 0))
invited_data['child_on_chain_activity'] = invited_data['address'].apply(lambda x: child_on_chain_activity.get(x, 0))
invited_data['child_is_on_chain_interaction_activity'] = invited_data['address'].apply(lambda x: child_is_on_chain_interaction_activity.get(x, 0))
invited_data['parent_max_daily_address_count'] = invited_data['address'].apply(lambda x: parent_max_daily_address_count.get(x, 0))
invited_data['parent_avg_daily_address_count'] = invited_data['address'].apply(lambda x: parent_avg_daily_address_count.get(x, 0))
invited_data['parent_max_continous_invited_days'] = invited_data['address'].apply(lambda x: parent_max_continous_invited_days.get(x, 0))
invited_data['child_max_daily_address_count'] = invited_data['address'].apply(lambda x: child_max_daily_address_count.get(x, 0))
invited_data['child_avg_daily_address_count'] = invited_data['address'].apply(lambda x: child_avg_daily_address_count.get(x, 0))
invited_data['child_max_continous_invited_days'] = invited_data['address'].apply(lambda x: child_max_continous_invited_days.get(x, 0))

invited_data['brother_s_avg_child_num'] = invited_data['address'].apply(lambda x: brother_s_avg_child_num.get(x, 0))
invited_data['brother_s_var_child_num'] = invited_data['address'].apply(lambda x: brother_s_var_child_num.get(x, 0))
invited_data['brother_s_cv_child_num'] = invited_data['address'].apply(lambda x: brother_s_cv_child_num.get(x, 0))
invited_data['brother_s_child_num_over_0_count'] = invited_data['address'].apply(lambda x: brother_s_child_num_over_0_count.get(x, 0))
invited_data['brother_s_child_num_over_3_count'] = invited_data['address'].apply(lambda x: brother_s_child_num_over_3_count.get(x, 0))
invited_data['brother_s_child_num_over_5_count'] = invited_data['address'].apply(lambda x: brother_s_child_num_over_5_count.get(x, 0))
invited_data['brother_s_child_num_over_10_count'] = invited_data['address'].apply(lambda x: brother_s_child_num_over_10_count.get(x, 0))
invited_data['brother_s_child_num_over_20_count'] = invited_data['address'].apply(lambda x: brother_s_child_num_over_20_count.get(x, 0))

invited_data['child_s_avg_child_num'] = invited_data['address'].apply(lambda x: child_s_avg_child_num.get(x, 0))
invited_data['child_s_var_child_num'] = invited_data['address'].apply(lambda x: child_s_var_child_num.get(x, 0))
invited_data['child_s_cv_child_num'] = invited_data['address'].apply(lambda x: child_s_cv_child_num.get(x, 0))
invited_data['child_s_child_num_over_0_count'] = invited_data['address'].apply(lambda x: child_s_child_num_over_0_count.get(x, 0))
invited_data['child_s_child_num_over_3_count'] = invited_data['address'].apply(lambda x: child_s_child_num_over_3_count.get(x, 0))
invited_data['child_s_child_num_over_5_count'] = invited_data['address'].apply(lambda x: child_s_child_num_over_5_count.get(x, 0))
invited_data['child_s_child_num_over_10_count'] = invited_data['address'].apply(lambda x: child_s_child_num_over_10_count.get(x, 0))
invited_data['child_s_child_num_over_20_count'] = invited_data['address'].apply(lambda x: child_s_child_num_over_20_count.get(x, 0))

invited_data = invited_data[['address', 'invited_by', 'depth', 'max_descendant_depth', 'subtree_size', 'subtree_width', 'parent_number', 'brother_number', 'child_number', 'grandchild_number', 'parent_x_activity', 'parent_telegram_activity', 'parent_discord_activity', 'parent_on_chain_activity', 'parent_is_on_chain_interaction_activity', 'child_x_activity', 'child_telegram_activity', 'child_discord_activity', 'child_on_chain_activity', 'child_is_on_chain_interaction_activity', 'parent_max_daily_address_count', 'parent_avg_daily_address_count', 'parent_max_continous_invited_days', 'child_max_daily_address_count', 'child_avg_daily_address_count', 'child_max_continous_invited_days', 'brother_s_avg_child_num', 'brother_s_var_child_num', 'brother_s_cv_child_num', 'brother_s_child_num_over_0_count', 'brother_s_child_num_over_3_count', 'brother_s_child_num_over_5_count', 'brother_s_child_num_over_10_count', 'brother_s_child_num_over_20_count', 'child_s_avg_child_num', 'child_s_var_child_num', 'child_s_cv_child_num', 'child_s_child_num_over_0_count', 'child_s_child_num_over_3_count', 'child_s_child_num_over_5_count', 'child_s_child_num_over_10_count', 'child_s_child_num_over_20_count']]
invited_data.to_csv('data/invited_data_tree.csv', index=False)

