import pandas as pd
import numpy as np
import networkx as nx
import time

invited_data = pd.read_csv('data/invited_data.csv')
g = nx.Graph()
empty_in_parent_column = invited_data['invited_by'].isna()

t = time.time()
for i in range(0, len(invited_data)):
    g.add_node(invited_data['address'][i])
    if not empty_in_parent_column[i]:
        g.add_node(invited_data['invited_by'][i])
        g.add_edge(invited_data['invited_by'][i], invited_data['address'][i])
print('init graph:', time.time() - t)

t = time.time()
pagerank = nx.pagerank(g)
print('pagerank:', time.time() - t)

t = time.time()
degree_centrality = nx.degree_centrality(g)
print('degree_centrality:', time.time() - t)

t = time.time()
avg_neighbor_degree = nx.average_neighbor_degree(g)
print('avg_neighbor_degree', time.time() - t)

dict = {}
for key, value in pagerank.items():
    dict[key] = value
invited_data['invited_data_pagerank'] = invited_data['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in degree_centrality.items():
    dict[key] = value
invited_data['invited_data_degree_centrality'] = invited_data['address'].apply(lambda x: dict.get(x, 0))

dict = {}
for key, value in avg_neighbor_degree.items():
    dict[key] = value
invited_data['invited_data_avg_neighbor_degree'] = invited_data['address'].apply(lambda x: dict.get(x, 0))

invited_data = invited_data[['address', 'invited_data_pagerank', 'invited_data_degree_centrality', 'invited_data_avg_neighbor_degree']]
invited_data.to_csv('data/invited_data_graph.csv', index=False)

