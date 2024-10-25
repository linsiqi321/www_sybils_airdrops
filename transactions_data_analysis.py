import pandas as pd
import numpy as np

# load data
addresses = pd.read_csv('data/address.csv')
transactions = pd.read_csv('data/raw_transactions_data.csv')

# preprocess
addresses['address'] = addresses['address'].str.lower()
transactions['from_address_hash'] = transactions['from_address_hash'].str.lower()
transactions['to_address_hash'] = transactions['to_address_hash'].str.lower()
transactions['timestamp'] = transactions['inserted_at']
transactions['value'] = transactions['value'].astype(float)
transactions['value'] = transactions['value'] / 10**18

from_stats = transactions.groupby('from_address_hash').agg(
    sum_value=('value', 'sum')
).reset_index()
to_stats = transactions.groupby('to_address_hash').agg(
    sum_value=('value', 'sum')
).reset_index()
from_stats.columns = ['address', 'from_sum_value']
to_stats.columns = ['address', 'to_sum_value']
earning_stats = from_stats.merge(to_stats, on='address', how='outer')
earning_stats.fillna(0, inplace=True)
earning_stats['earning_value'] = earning_stats['to_sum_value'] - earning_stats['from_sum_value']


from_tx = transactions[['from_address_hash', 'value', 'timestamp']]
from_tx.columns = ['address', 'value', 'timestamp']
to_tx = transactions[['to_address_hash', 'value', 'timestamp']]
to_tx.columns = ['address', 'value', 'timestamp']
combined_tx = pd.concat([from_tx, to_tx])
# print(len(combined_tx))
filtered_tx = combined_tx[combined_tx['address'].isin(addresses['address'])]
# print(len(filtered_tx))

# value mean, var, cv
stats = filtered_tx.groupby('address').agg(
    avg_value=('value', 'mean'),
    var_value=('value', 'var')
).reset_index()
stats['cv_value'] = np.sqrt(stats['var_value']) / stats['avg_value']

# incoming and outgoing value mean var and cv
filtered_tx_to = to_tx[to_tx['address'].isin(addresses['address'])]
filtered_tx_from = from_tx[from_tx['address'].isin(addresses['address'])]
incoming_value_stats = filtered_tx_to.groupby('address').agg(
    avg_incoming_value=('value', 'mean'),
    var_incoming_value=('value', 'var')
).reset_index()
incoming_value_stats['cv_incoming_value'] = np.sqrt(incoming_value_stats['var_incoming_value']) / incoming_value_stats['avg_incoming_value']
outgoing_value_stats = filtered_tx_from.groupby('address').agg(
    avg_outgoing_value=('value', 'mean'),
    var_outgoing_value=('value', 'var')
).reset_index()
outgoing_value_stats['cv_outgoing_value'] = np.sqrt(outgoing_value_stats['var_outgoing_value']) / outgoing_value_stats['avg_outgoing_value']
value_flow_stats = pd.merge(incoming_value_stats, outgoing_value_stats, on='address', how='outer')

# active days
filtered_tx['timestamp'] = pd.to_datetime(filtered_tx['timestamp'])
filtered_tx['date'] = filtered_tx['timestamp'].dt.date
active_days = filtered_tx.groupby('address')['date'].nunique().reset_index()
active_days.columns = ['address', 'active_days']

# transaction count and avg transaction count
transaction_counts = filtered_tx.groupby('address').size().reset_index(name='transaction_count')
transaction_counts = pd.merge(transaction_counts, active_days, on='address')
transaction_counts['avg_transaction_count'] = transaction_counts['transaction_count'] / transaction_counts['active_days']
transaction_counts = transaction_counts[['address', 'transaction_count', 'avg_transaction_count']]

# filtered_tx = filtered_tx.sort_values(by=['address', 'timestamp'])
# filtered_tx['prev_timestamp'] = filtered_tx.groupby('address')['timestamp'].shift(1)
# filtered_tx['interval'] = (filtered_tx['timestamp'] - filtered_tx['prev_timestamp']).dt.days
# interval_stats = filtered_tx.groupby('address')['interval'].mean().reset_index()
# interval_stats.columns = ['address', 'avg_interval']

# daily, weekly, monthly incoming and outgoing count, avg value, var value, cv value
filtered_original_tx = transactions[['from_address_hash', 'to_address_hash', 'value', 'timestamp']]
filtered_original_tx['timestamp'] = pd.to_datetime(filtered_original_tx['timestamp'])
filtered_original_tx['date'] = filtered_original_tx['timestamp'].dt.floor('D')
# filtered_original_tx['date'] = filtered_original_tx['timestamp'].dt.date
daily_incoming_stats = filtered_original_tx.groupby(['to_address_hash', 'date']).agg(
    incoming_count=('from_address_hash', 'nunique'),
    avg_incoming_value=('value', 'mean'),
    var_incoming_value=('value', 'var')
).reset_index()
daily_incoming_stats['cv_incoming_value'] = np.sqrt(daily_incoming_stats['var_incoming_value']) / daily_incoming_stats['avg_incoming_value']
daily_incoming_stats = daily_incoming_stats.loc[daily_incoming_stats.groupby('to_address_hash')['incoming_count'].idxmax()]
daily_incoming_stats = daily_incoming_stats[['to_address_hash', 'incoming_count', 'avg_incoming_value', 'var_incoming_value', 'cv_incoming_value']]
daily_incoming_stats.columns = ['address', 'daily_incoming_count', 'daily_avg_incoming_value', 'daily_var_incoming_value', 'daily_cv_incoming_value']
daily_outgoing_stats = filtered_original_tx.groupby(['from_address_hash', 'date']).agg(
    outgoing_count=('to_address_hash', 'nunique'),
    avg_outgoing_value=('value', 'mean'),
    var_outgoing_value=('value', 'var')
).reset_index()
daily_outgoing_stats['cv_outgoing_value'] = np.sqrt(daily_outgoing_stats['var_outgoing_value']) / daily_outgoing_stats['avg_outgoing_value']
daily_outgoing_stats = daily_outgoing_stats.loc[daily_outgoing_stats.groupby('from_address_hash')['outgoing_count'].idxmax()]
daily_outgoing_stats = daily_outgoing_stats[['from_address_hash', 'outgoing_count', 'avg_outgoing_value', 'var_outgoing_value', 'cv_outgoing_value']]
daily_outgoing_stats.columns = ['address', 'daily_outgoing_count', 'daily_avg_outgoing_value', 'daily_var_outgoing_value', 'daily_cv_outgoing_value']
daily_stats = pd.merge(daily_incoming_stats, daily_outgoing_stats, on='address', how='outer')

filtered_original_tx['week'] = filtered_original_tx['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_incoming_stats = filtered_original_tx.groupby(['to_address_hash', 'week']).agg(
    incoming_count=('from_address_hash', 'nunique'),
    avg_incoming_value=('value', 'mean'),
    var_incoming_value=('value', 'var')
).reset_index()
weekly_incoming_stats['cv_incoming_value'] = np.sqrt(weekly_incoming_stats['var_incoming_value']) / weekly_incoming_stats['avg_incoming_value']
weekly_incoming_stats = weekly_incoming_stats.loc[weekly_incoming_stats.groupby('to_address_hash')['incoming_count'].idxmax()]
weekly_incoming_stats = weekly_incoming_stats[['to_address_hash', 'incoming_count', 'avg_incoming_value', 'var_incoming_value', 'cv_incoming_value']]
weekly_incoming_stats.columns = ['address', 'weekly_incoming_count', 'weekly_avg_incoming_value', 'weekly_var_incoming_value', 'weekly_cv_incoming_value']
weekly_outgoing_stats = filtered_original_tx.groupby(['from_address_hash', 'week']).agg(
    outgoing_count=('to_address_hash', 'nunique'),
    avg_outgoing_value=('value', 'mean'),
    var_outgoing_value=('value', 'var')
).reset_index()
weekly_outgoing_stats['cv_outgoing_value'] = np.sqrt(weekly_outgoing_stats['var_outgoing_value']) / weekly_outgoing_stats['avg_outgoing_value']
weekly_outgoing_stats = weekly_outgoing_stats.loc[weekly_outgoing_stats.groupby('from_address_hash')['outgoing_count'].idxmax()]
weekly_outgoing_stats = weekly_outgoing_stats[['from_address_hash', 'outgoing_count', 'avg_outgoing_value', 'var_outgoing_value', 'cv_outgoing_value']]
weekly_outgoing_stats.columns = ['address', 'weekly_outgoing_count', 'weekly_avg_outgoing_value', 'weekly_var_outgoing_value', 'weekly_cv_outgoing_value']
weekly_stats = pd.merge(weekly_incoming_stats, weekly_outgoing_stats, on='address', how='outer')

filtered_original_tx['month'] = filtered_original_tx['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)
monthly_incoming_stats = filtered_original_tx.groupby(['to_address_hash', 'month']).agg(
    incoming_count=('from_address_hash', 'nunique'),
    avg_incoming_value=('value', 'mean'),
    var_incoming_value=('value', 'var')
).reset_index()
monthly_incoming_stats['cv_incoming_value'] = np.sqrt(monthly_incoming_stats['var_incoming_value']) / monthly_incoming_stats['avg_incoming_value']
monthly_incoming_stats = monthly_incoming_stats.loc[monthly_incoming_stats.groupby('to_address_hash')['incoming_count'].idxmax()]
monthly_incoming_stats = monthly_incoming_stats[['to_address_hash', 'incoming_count', 'avg_incoming_value', 'var_incoming_value', 'cv_incoming_value']]
monthly_incoming_stats.columns = ['address', 'monthly_incoming_count', 'monthly_avg_incoming_value', 'monthly_var_incoming_value', 'monthly_cv_incoming_value']
monthly_outgoing_stats = filtered_original_tx.groupby(['from_address_hash', 'month']).agg(
    outgoing_count=('to_address_hash', 'nunique'),
    avg_outgoing_value=('value', 'mean'),
    var_outgoing_value=('value', 'var')
).reset_index()
monthly_outgoing_stats['cv_outgoing_value'] = np.sqrt(monthly_outgoing_stats['var_outgoing_value']) / monthly_outgoing_stats['avg_outgoing_value']
monthly_outgoing_stats = monthly_outgoing_stats.loc[monthly_outgoing_stats.groupby('from_address_hash')['outgoing_count'].idxmax()]
monthly_outgoing_stats = monthly_outgoing_stats[['from_address_hash', 'outgoing_count', 'avg_outgoing_value', 'var_outgoing_value', 'cv_outgoing_value']]
monthly_outgoing_stats.columns = ['address', 'monthly_outgoing_count', 'monthly_avg_outgoing_value', 'monthly_var_outgoing_value', 'monthly_cv_outgoing_value']
monthly_stats = pd.merge(monthly_incoming_stats, monthly_outgoing_stats, on='address', how='outer')


# 80%, 50%, 30% value concentration days
def calculate_concentration_days(df, percentage):
    df = df.sort_values(by=['value','timestamp'], ascending=[False, True])

    df['cumulative_value'] = df['value'].cumsum()
    total_value = df['cumulative_value'].iloc[-1]
    df['cumulative_percentage'] = df['cumulative_value'] / total_value

    threshold_date = df[df['cumulative_percentage'] <= percentage]['timestamp'].max()

    df = df[df['timestamp'] <= threshold_date]
    concentration_days = df['date'].nunique()

    return concentration_days

concentration_stats = filtered_tx.groupby('address').apply(
    lambda x: pd.Series({
        'days_80_value_percent': calculate_concentration_days(x, 0.80),
        'days_50_value_percent': calculate_concentration_days(x, 0.50),
        'days_30_value_percent': calculate_concentration_days(x, 0.30)
    })
).reset_index()
concentration_stats = pd.merge(concentration_stats, active_days, on='address', how='outer')
concentration_stats['prop_80_value_percent'] = concentration_stats['days_80_value_percent'] / concentration_stats['active_days']
concentration_stats['prop_50_value_percent'] = concentration_stats['days_50_value_percent'] / concentration_stats['active_days']
concentration_stats['prop_30_value_percent'] = concentration_stats['days_30_value_percent'] / concentration_stats['active_days']
concentration_stats = concentration_stats[['address', 'days_80_value_percent', 'prop_80_value_percent', 'days_50_value_percent', 'prop_50_value_percent', 'days_30_value_percent', 'prop_30_value_percent']]

# invited data preprocess
from_to_addresses = transactions[['from_address_hash', 'to_address_hash']].dropna().drop_duplicates()
invited_data = pd.read_csv('data/invited_data.csv')
invited_data['address'] = invited_data['address'].str.lower()
invited_data['invited_by'] = invited_data['invited_by'].str.lower()
child_dict = {}
parent_dict = {}
empty_in_parent_column = invited_data['invited_by'].isna()
for i in range(0, len(invited_data)):
    if not empty_in_parent_column[i]:
        parent_dict[invited_data['address'][i]] = invited_data['invited_by'][i]
        if child_dict.get(invited_data['invited_by'][i]) is None:
            child_dict[invited_data['invited_by'][i]] = []
            child_dict[invited_data['invited_by'][i]].append(invited_data['address'][i])
        else:
            child_dict[invited_data['invited_by'][i]].append(invited_data['address'][i])

from_to_addresses['from_s_parent'] = from_to_addresses['from_address_hash'].apply(lambda x: parent_dict.get(x, None))
from_to_addresses['to_s_parent'] = from_to_addresses['to_address_hash'].apply(lambda x: parent_dict.get(x, None))
from_to_addresses['relation1'] = from_to_addresses['from_s_parent'] == from_to_addresses['to_address_hash']
from_to_addresses['relation2'] = from_to_addresses['to_s_parent'] == from_to_addresses['from_address_hash']
from_to_addresses['relation'] = from_to_addresses['relation1'] + from_to_addresses['relation2']
from_to_addresses['from_s_child_num'] = from_to_addresses['from_address_hash'].apply(lambda x: len(child_dict.get(x, [])))
from_to_addresses['to_s_child_num'] = from_to_addresses['to_address_hash'].apply(lambda x: len(child_dict.get(x, [])))
from_to_addresses = from_to_addresses[['from_address_hash', 'to_address_hash', 'from_s_parent', 'to_s_parent', 'relation', 'from_s_child_num', 'to_s_child_num']]

# probability of a transaction object who is a parent or child
# probability of a parent or child who is a transaction object
from_relation = from_to_addresses.groupby('from_address_hash').agg({'relation': 'sum', 'to_address_hash': 'count'}).reset_index()
from_relation.columns = ['address', 'relation', 'tx_count']
to_relation = from_to_addresses.groupby('to_address_hash').agg({'relation': 'sum', 'from_address_hash': 'count'}).reset_index()
to_relation.columns = ['address', 'relation', 'tx_count']
relation = pd.concat([from_relation, to_relation])
relation = relation.groupby('address').agg({'relation': 'sum', 'tx_count': 'sum'}).reset_index()

from_to_addresses['from_s_parent_num'] = from_to_addresses['from_address_hash'].apply(lambda x: 1 if parent_dict.get(x, None) is not None else 0)
from_to_addresses['to_s_parent_num'] = from_to_addresses['to_address_hash'].apply(lambda x: 1 if parent_dict.get(x, None) is not None else 0)
from_parent_child_num = from_to_addresses[['from_address_hash', 'from_s_parent_num', 'from_s_child_num']].drop_duplicates()
to_parent_child_num = from_to_addresses[['to_address_hash', 'to_s_parent_num', 'to_s_child_num']].drop_duplicates()
from_parent_child_num.columns = ['address', 'parent_num', 'child_num']
to_parent_child_num.columns = ['address', 'parent_num', 'child_num']
parent_child_num = pd.concat([from_parent_child_num, to_parent_child_num])
parent_child_num = parent_child_num[['address', 'parent_num', 'child_num']].drop_duplicates()
relation_stats = relation.merge(parent_child_num, on='address', how='outer')
relation_stats['tx_object_prob'] = relation_stats['relation'] / relation_stats['tx_count']
relation_stats['parent_child_num'] = relation_stats['parent_num'] + relation_stats['child_num']
relation_stats['parent_child_prob'] = relation_stats.apply(lambda row: row['relation'] / row['parent_child_num'] if row['parent_child_num'] != 0 else 0, axis=1)
relation_stats = relation_stats[['address', 'tx_object_prob', 'parent_child_prob']]

final_stats = pd.merge(stats, earning_stats, on='address', how='outer')\
                .merge(value_flow_stats, on='address', how='outer')\
                .merge(active_days, on='address', how='outer')\
                .merge(transaction_counts, on='address', how='outer')\
                .merge(interval_stats, on='address', how='outer')\
                .merge(daily_stats, on='address', how='outer')\
                .merge(weekly_stats, on='address', how='outer')\
                .merge(monthly_stats, on='address', how='outer')\
                .merge(concentration_stats, on='address', how='outer') \
                .merge(relation_stats, on='address', how='outer')

final_stats.to_csv('data/transactions_stats_all_address.csv', index=False)