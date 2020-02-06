import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def find_index(wcss):
	seen_val = []
	difference = []
	for index, arg in enumerate(wcss):
		seen_val.append(int(arg))
		if index != 0:
			diff = seen_val[index-1] - int(arg)
			if (diff in difference) or (diff <= 10) or (len(difference)>=6):
				return difference
			else:
				difference.append(diff)
	return difference

def get_cluster_number(data):
	#Scale the data
	sc_X = StandardScaler()
	data = sc_X.fit_transform(data)
	wcss = []
	for i in range(1, 11):
	    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
	    kmeans.fit(data)
	    wcss.append(kmeans.inertia_)

	#print(wcss)
	difference = find_index(wcss)
	return len(difference)
	
	"""
	print("Number of Clusters sholud be:", n_cluster)
	plt.plot(range(1, 11), wcss)
	plt.title('The Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS')
	plt.show()
	"""