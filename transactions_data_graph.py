import pandas as pd
import numpy as np
import networkx as nx
import time

transactions_data = pd.read_csv('data/raw_transactions_data.csv')
g = nx.DiGraph()

t = time.time()
src = transactions_data['from_address_hash'].to_numpy()
dst = transactions_data['to_address_hash'].to_numpy()
g.add_nodes_from(src)
g.add_nodes_from(dst)
for i in range(len(src)):
    g.add_edge(src[i], dst[i])
print('init graph:', time.time() - t)

t = time.time()
pagerank = nx.pagerank(g)
print('pagerank:', time.time() - t)

t = time.time()
in_degree_centrality = nx.in_degree_centrality(g)
print('in_degree_centrality:', time.time() - t)

t = time.time()
out_degree_centrality = nx.out_degree_centrality(g)
print('out_degree_centrality:', time.time() - t)

t = time.time()
degree_centrality = nx.degree_centrality(g)
print('degree_centrality:', time.time() - t)

t = time.time()
in_avg_neighbor_degree = nx.average_neighbor_degree(g, source='in', target='in')
print('in_avg_neighbor_degree', time.time() - t)

t = time.time()
out_avg_neighbor_degree = nx.average_neighbor_degree(g, source='out', target='out')
print('out_avg_neighbor_degree', time.time() - t)

t = time.time()
avg_neighbor_degree = nx.average_neighbor_degree(g, source='in+out', target='in+out')
print('avg_neighbor_degree', time.time() - t)

t = time.time()
eigenvector_centrality = nx.eigenvector_centrality(g)
print('eigenvector_centrality', time.time() - t)

t = time.time()
katz_centrality = nx.katz_centrality(g, alpha=0.01)
print('katz_centrality', time.time() - t)

address = pd.concat([transactions_data['from_address_hash'], transactions_data['to_address_hash']]).unique()
address = pd.DataFrame(address, columns=['address'])

dict = {}
for key, value in pagerank.items():
    dict[key] = value
address['transactions_data_pagerank'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in in_degree_centrality.items():
    dict[key] = value
address['transactions_data_in_degree_centrality'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in out_degree_centrality.items():
    dict[key] = value
address['transactions_data_out_degree_centrality'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in degree_centrality.items():
    dict[key] = value
address['transactions_data_degree_centrality'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in in_avg_neighbor_degree.items():
    dict[key] = value
address['transactions_data_in_avg_neighbor_degree'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in out_avg_neighbor_degree.items():
    dict[key] = value
address['transactions_data_out_avg_neighbor_degree'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in avg_neighbor_degree.items():
    dict[key] = value
address['transactions_data_avg_neighbor_degree'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in eigenvector_centrality.items():
    dict[key] = value
address['transactions_data_eigenvector_centrality'] = address['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in katz_centrality.items():
    dict[key] = value
address['transactions_data_katz_centrality'] = address['address'].apply(lambda x: dict.get(x, 0))


address = address[['address', 'transactions_data_pagerank', 'transactions_data_in_degree_centrality', 'transactions_data_out_degree_centrality', 'transactions_data_degree_centrality', 'transactions_data_in_avg_neighbor_degree', 'transactions_data_out_avg_neighbor_degree', 'transactions_data_avg_neighbor_degree', 'transactions_data_eigenvector_centrality', 'transactions_data_katz_centrality']]
address.to_csv('data/transactions_data_graph.csv', index=False)

