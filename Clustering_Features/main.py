import os
from os import listdir
from os.path import isfile, join
import csv
from kmeans import *
from elbow_method import *

#GENERATE CLUSTERS
def get_clusters(path_to_centroid_csv, save_fig_path):
	csvs = [os.path.join(path_to_centroid_csv, f) for f in listdir(path_to_centroid_csv) if isfile(join(path_to_centroid_csv, f))]
	for index, csv_file in enumerate(csvs):
		if index <= 20: #CONTROL CHECK: ALLOWS TO RUN CLUSTERING ON REQUIRED NUMBER OF CSVS
			coords = []
			print("Processing CSV:", index)
			with open(csv_file, 'r') as file:
				csv_reader = csv.reader(file, delimiter=',')
				next(csv_reader)
				for row in csv_reader:
					coords.append([float(row[0]), float(row[1])])
		
			#CALL CLUSTERING ON EACH CSV coords
			n_cluster = get_cluster_number(np.array(coords))
			cluster = KMeans_Cluster(k=n_cluster, tol=0.001, max_iter=300)
			n_iter, centroids, classified_points = cluster.fit(np.array(coords))
			
			#GET FILE NAME FOR SAVING PLOTS
			file_name = csv_file.split("\\")[-1].split(".")[0]+".png"
			save_path = save_fig_path + file_name
			cluster.visualize(np.array(coords), centroids, classified_points, save_path)

path_to_centroids = "C:\\Users\\AnilYadav\\Desktop\\Projects\\prostate\\ml_models\\centroids\\"
save_fig_path = "C:\\Users\\AnilYadav\\Desktop\\Projects\\prostate\\ml_models\\logistic\\cluster\\cluster_figures\\"

if __name__ == "__main__":
	get_clusters(path_to_centroids, save_fig_path)