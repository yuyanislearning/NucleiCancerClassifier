import numpy as np
from os import listdir
from os.path import isfile, join
from skimage import io
from create_graph import *
from radiomicsfinal import *
from prediction import *
import csv
import argparse


def createFeatures(path_to_centroids, path_to_mask_csvs, path_to_tiles, label_file=None):
	masks_csvs = [f for f in listdir(path_to_mask_csvs) if isfile(join(path_to_mask_csvs, f))]
	
	#CREATE LABELS FOR CREATING TRAIN LABELS
	if label_file is not None:
			get_label_dict = create_old_feature_dict(label_file)
	
	with open('features.csv', 'w', newline='') as csv_file:
		#CSV WRITER
		csv_writer = csv.writer(csv_file, delimiter=',')
		for index, arg in enumerate(masks_csvs):
			#LOOP CONTROL
			if index <= 3:
				print("Processing File {}: {}".format(index+1, arg))
				try:
					path_to_mask = os.path.join(path_to_mask_csvs, arg)
					path_to_centroid_file = os.path.join(path_to_centroids, arg)
					tile_file_name = arg.split('.')[0]+".png"
					tile_img_path = os.path.join(path_to_tiles, tile_file_name)
				
					#RADIOMICS FEATURES
					radiomics_features = list(createfeatures(tile_img_path, path_to_mask))

					#CONSTRUCT GRAPH AND EXTRACT FEATURES
					graph_obj = Graph(path_to_centroid_file) 
					G, _ = graph_obj.graph_by_threshold(50) 
					new_g, new_pos = graph_obj.remove_single_nodes(G, _) 
					largest_connected_subgraph = graph_obj.get_connected_components(new_g)
					graph_features = graph_obj.get_graphFeatures(largest_connected_subgraph)

					#WRITING OUT FILE HEADER
					if index == 0:
						graph_header = graph_obj.extractedFeatures
						radiomics_header = ['rf_{}'.format(count) for count in range(len(radiomics_features))]
						graph_header.insert(0,'slide_id')
						graph_header.extend(radiomics_header)

						if label_file is not None:
							graph_header.append('label')

						csv_writer.writerow(graph_header)

					#COMBINE FEATURES
					graph_features.insert(0, arg.split('.')[0])
					graph_features.extend(radiomics_features)
					if label_file is not None:
							graph_features.append(get_label_dict[arg.split('.')[0]])

					#WRITE TO FILE
					csv_writer.writerow(graph_features)

				except Exception as e:
					print(e)

	print("FILE CREATED!")

def create_old_feature_dict(label_csv):
	with open(label_csv, 'r', newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		new_dict = {}
		for row in reader:
			new_dict[row[1]] = int(row[2])
	return new_dict

#SET PATH FOR FILES
path_to_centroids = "data\\centroids\\"
path_to_mask_csvs = "data\\masks_512x512\\"
path_to_tiles = "data\\tiles_rois\\normalized\\"
#FOR TEST DATA, SET 'path_to_label=None'
path_to_label = "data\\tiles_rois\\dataset.csv"

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_to_train_centroids", default=path_to_centroids, type=str, required=False,
						help="The path to dir of train mask csv.")
	parser.add_argument("--path_to_test_centroids", default=path_to_centroids, type=str, required=False,
						help="The path to dir of train mask csv.")
	parser.add_argument("--path_to_train_mask_csvs", default=path_to_mask_csvs, type=str, required=False,
						help="The path to dir of train mask csv.")
	parser.add_argument("--path_to_test_mask_csvs", default=path_to_mask_csvs, type=str, required=False,
						help="The path to dir of test mask csv.")
	parser.add_argument("--path_to_test_tiles", default=path_to_tiles, type=str, required=False,
						help="The path to test tiles.")
	parser.add_argument("--path_to_train_tiles", default=path_to_tiles, type=str, required=False,
						help="The path to train tiles.")
	parser.add_argument("--path_to_label", default=path_to_label, type=str, required=False,
						help="The path to label.")
	args = parser.parse_args()
	if not os.path.exists('train_features.csv') and not os.path.exists('test_features.csv'):
		createFeatures(args.path_to_train_centroids, args.path_to_train_mask_csvs, args.path_to_train_tiles, args.path_to_label)
		createFeatures(args.path_to_test_centroids, args.path_to_test_mask_csvs, args.path_to_test_tiles, None)
	predict('train_features.csv', 'test_features.csv')





