import pandas as pd

dataset_deeplearning = pd.read_csv('data/address.csv')
dataset_deeplearning['address'] = dataset_deeplearning['address'].str.lower()
df1 = pd.read_csv('data/invited_data_tree.csv')
df1['address'] = df1['address'].str.lower()
df1['grandchild_child_ratio'] = df1['grandchild_number'] / df1['child_number']
df1['depth_ratio'] = df1['depth'] / df1['max_descendant_depth']
df1['child_brother_self_ratio'] = (df1['child_number']) / (df1['brother_number'] + 1)
df1['brother_s_child_num_over_0_ratio'] = df1['brother_s_child_num_over_0_count'] / df1['brother_number']
df1['brother_s_child_num_over_3_ratio'] = df1['brother_s_child_num_over_3_count'] / df1['brother_number']
df1['brother_s_child_num_over_5_ratio'] = df1['brother_s_child_num_over_5_count'] / df1['brother_number']
df1['brother_s_child_num_over_10_ratio'] = df1['brother_s_child_num_over_10_count'] / df1['brother_number']
df1['brother_s_child_num_over_20_ratio'] = df1['brother_s_child_num_over_20_count'] / df1['brother_number']
df1['child_s_child_num_over_0_ratio'] = df1['child_s_child_num_over_0_count'] / df1['child_number']
df1['child_s_child_num_over_3_ratio'] = df1['child_s_child_num_over_3_count'] / df1['child_number']
df1['child_s_child_num_over_5_ratio'] = df1['child_s_child_num_over_5_count'] / df1['child_number']
df1['child_s_child_num_over_10_ratio'] = df1['child_s_child_num_over_10_count'] / df1['child_number']
df1['child_s_child_num_over_20_ratio'] = df1['child_s_child_num_over_20_count'] / df1['child_number']
df1 = df1[['address', 'depth', 'max_descendant_depth','subtree_size', 'subtree_width', 'parent_number', 'brother_number', 'child_number', 'grandchild_number', 'depth_ratio', 'grandchild_child_ratio', 'child_brother_self_ratio', 'parent_x_activity', 'parent_telegram_activity','parent_discord_activity', 'parent_on_chain_activity','parent_is_on_chain_interaction_activity', 'child_x_activity','child_telegram_activity', 'child_discord_activity','child_on_chain_activity', 'child_is_on_chain_interaction_activity','parent_max_daily_address_count', 'parent_avg_daily_address_count','parent_max_continous_invited_days', 'child_max_daily_address_count','child_avg_daily_address_count', 'child_max_continous_invited_days','brother_s_avg_child_num', 'brother_s_var_child_num','brother_s_cv_child_num', 'brother_s_child_num_over_0_count','brother_s_child_num_over_0_ratio', 'brother_s_child_num_over_3_count','brother_s_child_num_over_3_ratio', 'brother_s_child_num_over_5_count','brother_s_child_num_over_5_ratio', 'brother_s_child_num_over_10_count','brother_s_child_num_over_10_ratio', 'brother_s_child_num_over_20_count','brother_s_child_num_over_20_ratio', 'child_s_avg_child_num','child_s_var_child_num', 'child_s_cv_child_num','child_s_child_num_over_0_count', 'child_s_child_num_over_0_ratio','child_s_child_num_over_3_count', 'child_s_child_num_over_3_ratio','child_s_child_num_over_5_count', 'child_s_child_num_over_5_ratio','child_s_child_num_over_10_count', 'child_s_child_num_over_10_ratio','child_s_child_num_over_20_count', 'child_s_child_num_over_20_ratio']]

df2 = pd.read_csv('data/score_0811_with_effective_referral.csv')
df2['address'] = df2['address'].str.lower()
df2['is_on_chain_interaction_activity'] = df2['on_chain_interaction_count'].apply(lambda x: 1 if x > 0 else 0)
df2 = df2[['address', 'follow_on_x_count', 'join_telegram_count','join_discord_count', 'on_chain_interaction_count','is_on_chain_interaction_activity']]

df3 = pd.read_csv('data/invited_data_with_time.csv')
df3['address'] = df3['address'].str.lower()
df3 = df3[['address', 'max_daily_address_count', 'avg_daily_address_count','var_daily_address_count', 'cv_daily_address_count','max_continous_invited_days', 'max_interval_hours', 'avg_interval_hours']]

df4 = pd.read_csv('data/invited_data_graph.csv')
df4['address'] = df4['address'].str.lower()
df4 = df4[['address', 'invited_data_pagerank', 'invited_data_degree_centrality', 'invited_data_avg_neighbor_degree']]

df5 = pd.read_csv('data/transactions_data_graph.csv')
df5['address'] = df5['address'].str.lower()
df5 = df5[['address', 'transactions_data_pagerank', 'transactions_data_in_degree_centrality', 'transactions_data_out_degree_centrality', 'transactions_data_degree_centrality', 'transactions_data_in_avg_neighbor_degree', 'transactions_data_out_avg_neighbor_degree', 'transactions_data_avg_neighbor_degree', 'transactions_data_eigenvector_centrality', 'transactions_data_katz_centrality']]

df6 = pd.read_csv('data/transactions_stats.csv')
df6['address'] = df6['address'].str.lower()
df6 = df6[['address', 'earning_value', 'avg_value', 'var_value', 'cv_value', 'avg_incoming_value','var_incoming_value', 'cv_incoming_value', 'avg_outgoing_value','var_outgoing_value', 'cv_outgoing_value', 'transaction_count', 'avg_transaction_count','daily_incoming_count', 'daily_avg_incoming_value','daily_var_incoming_value', 'daily_cv_incoming_value','daily_outgoing_count', 'daily_avg_outgoing_value','daily_var_outgoing_value', 'daily_cv_outgoing_value','weekly_incoming_count', 'weekly_avg_incoming_value','weekly_var_incoming_value', 'weekly_cv_incoming_value','weekly_outgoing_count', 'weekly_avg_outgoing_value','weekly_var_outgoing_value', 'weekly_cv_outgoing_value','monthly_incoming_count', 'monthly_avg_incoming_value','monthly_var_incoming_value', 'monthly_cv_incoming_value','monthly_outgoing_count', 'monthly_avg_outgoing_value','monthly_var_outgoing_value', 'monthly_cv_outgoing_value','days_80_value_percent', 'prop_80_value_percent','days_50_value_percent', 'prop_50_value_percent','days_30_value_percent', 'prop_30_value_percent', 'tx_object_prob','parent_child_prob']]
dataset_deeplearning = pd.merge(dataset_deeplearning, df1, on='address', how='outer')
dataset_deeplearning = pd.merge(dataset_deeplearning, df2, on='address', how='outer')
dataset_deeplearning = pd.merge(dataset_deeplearning, df3, on='address', how='outer')
dataset_deeplearning = pd.merge(dataset_deeplearning, df4, on='address', how='outer')
dataset_deeplearning = pd.merge(dataset_deeplearning, df5, on='address', how='outer')
dataset_deeplearning = pd.merge(dataset_deeplearning, df6, on='address', how='outer')
dataset_deeplearning = dataset_deeplearning[dataset_deeplearning['points'].notna()]
dataset_deeplearning = dataset_deeplearning.fillna(0)
dataset_deeplearning.to_csv('data/dataset_deeplearning.csv', index=False)
