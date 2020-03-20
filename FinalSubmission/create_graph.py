import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from skimage import io
import statistics as stat
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

#GRAPH CLASS
class Graph:

	extractedFeatures = ['nuclei_count','density','avg_degree','avg_closeness','btw_centrality','harmonic','avg_eccen','radius','efficiency','transitivity','avg_cluste_coef',
	'avg_voteRank','avg_ic','avg_lc']
	
	def __init__(self, centroids_csv=None):
		self.centroids_csv = centroids_csv

	def graph_by_threshold(self, threshold=0):
		if (threshold > 0) and (self.centroids_csv):
			coord=pd.read_csv(self.centroids_csv)
			self.num_nuclei = len(coord.index)
			nodes=list(range(len(coord)))
			positions=dict(zip(nodes, zip(coord['x'],coord['y'])))
			distmatrix=distance.cdist(coord,coord)
			distmatrix[distmatrix>threshold] = 0

			for i in range(len(distmatrix)):
				distmatrix[i,i]=0
			#ADJACENCY MATRIX
			G = nx.from_numpy_matrix(distmatrix)
			return G, positions
		else:
			return None

	#removes nodes with no edges
	def remove_single_nodes(self, g, _):
		uniques_nodes = []
		for arg in g.degree:
			#IF NO CONNECTION, IT IS A NOTE A UNIQUE NODE
			if arg[1] == 0:
				uniques_nodes.append(arg[0])
		g.remove_nodes_from(uniques_nodes)
		for arg in uniques_nodes:
			_.pop(arg)
		return g, _

	#gives the largest connected graph
	def get_connected_components(self, G):
		components = list(nx.connected_components(G))
		components_len = [len(arg) for arg in components]
		max_connected_component_index = components_len.index(max(components_len))
		return G.subgraph(components[max_connected_component_index])

	#number of average connections
	def get_average_degree(self, G=None):
		if G is not None:
			connections = []
			for arg in G.degree:
				if arg[1] != 0:
					connections.append(arg[1])
			return np.mean(connections)
		else:
			None

	def get_graphFeatures(self, G=None):
		if G is not None:
			#density
			density = nx.density(G)
			#average degree
			avg_degree = self.get_average_degree(G)
			#average closeness centrality
			avg_closeness = np.array(list(nx.closeness_centrality(G, distance='weight').values()))
			avg_closeness = np.mean(avg_closeness[avg_closeness != 0.0])
			#average betweenness centrality
			btw_centrality = np.array(list(nx.betweenness_centrality(G, weight='weight').values()))
			btw_centrality = np.mean(btw_centrality[btw_centrality != 0.0])
			#average harmonic centrality
			harmonic = np.array(list(nx.harmonic_centrality(G, distance='weight').values()))
			harmonic = np.mean(harmonic[harmonic != 0.0])
			#get eccentricity, radius, efficiency
			eccen = nx.eccentricity(G)
			radius = nx.radius(G, e=eccen)
			eccen = np.array(list(eccen.values()))
			avg_eccen = np.mean(eccen[eccen != 0.0])
			efficiency = nx.global_efficiency(G)
			#get transitivity and average cluster coefficient
			transitivity = nx.transitivity(G)
			avg_cluster_coef = nx.average_clustering(G, weight='weight', count_zeros=False)
			#avg vote rank
			avg_voteRank = stat.mean(nx.voterank(G))
			#avg information centrality
			ic = np.array(list(nx.information_centrality(G, weight='weight').values()))
			avg_ic = np.mean(ic[ic != 0.0])
			#avg load centrality
			lc = np.array(list(nx.load_centrality(G, weight='weight').values()))
			avg_lc = np.mean(lc[lc != 0.0])
			return [self.num_nuclei, density, avg_degree, avg_closeness, btw_centrality, harmonic, avg_eccen, radius, efficiency, transitivity, avg_cluster_coef, avg_voteRank, avg_ic, avg_lc]
		else:
			return None