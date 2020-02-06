import numpy as np
import matplotlib.pyplot as plt

class KMeans_Cluster:
	def __init__(self, k=6, tol=0.001, max_iter=300):
		self.k= k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self, data):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]
		#OPTIMIZATION
		for each_iter in range(self.max_iter):
			self.classifications = {}
			for i in range(self.k):
				self.classifications[i] = []

			#POPULATING EMPTY CLASSIFICATION SET
			for featureset in data:
				#CALCULATE ECULIDEAN DISTANCE BETWEEN CENTROIDS AND DATA POINTS: ONE DATA POINTS DISTANCE IS CALCULATED WITH ALL THE CENTROIDS
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				"""
				for centroid in self.centroids:
					print("{} - {}".format(featureset, self.centroids[centroid]))
					distances = [np.linalg.norm(featureset-self.centroids[centroid])]
					print(distances)
				"""
				#GET INDEX OF MINIMUM DISTANCE
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			#UPDATE CENTROIDS
			prev_centroids = dict(self.centroids)
			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)
			
			#OPTIMIAZTION CONDITION
			optimized = True
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
					#print("-"*25)
					#print("Moving by {} %".format(np.sum((current_centroid - original_centroid)/original_centroid*100.0)))
					optimized = False
			if optimized:
				return each_iter+1, self.centroids, self.classifications

			"""
			prev_centroids = [arg for arg in self.centroids.values()]
			#TERMINATING LOOP CONDITIONS
			cur_centroids = [arg for arg in self.centroids.values()]
			if np.array_equal(prev_centroids,cur_centroids):
				return (each_iter+1, self.centroids, self.classifications)
			else:
				continue
			"""

	def predict(self, data, centroids):
		distances = [np.linalg.norm(data-centroids[centroid]) for centroid in centroids]
		classification = distances.index(min(distances))

	def visualize(self, data, centroids, classified_points, save_fig_path):
		#PLOT CENTROIDS
		prop_cycle = plt.rcParams['axes.prop_cycle']
		colors = prop_cycle.by_key()['color']
		for centroid in centroids:
			arr = centroids[centroid]
			x, y = arr
			plt.scatter(x,y, marker='o', color=colors[centroid], linewidth=4)
		
		#PLOT DATA POINTS
		for classification in classified_points:
			color = colors[classification]
			for featureset in classified_points[classification]:
				x, y = featureset
				plt.scatter(x,y, marker='x', color=colors[classification], s=50, linewidth=2)
		plt.savefig(save_fig_path)
		plt.clf() #CLEAR CURRENT FIGURE FOR REUSE