import numpy as np
import os
from os import listdir
from os.path import isfile, join
from skimage.measure import label, regionprops
from skimage import io
import csv
import pandas as pd

def getCentroid(path_to_masks, centroids_path):
	masks_csvs = [f for f in listdir(path_to_masks) if isfile(join(path_to_masks, f))]
	
	if not os.path.exists(centroids_path):
		os.makedirs(centroids_path)

	for index, arg in enumerate(masks_csvs):
		if index <= len(masks_csvs):
			print('Processing file {}: {}'.format(index+1, arg))
			file_path = os.path.join(centroids_path, arg)
			with open(file_path, 'w', newline='') as csv_file:
				csv_writer = csv.writer(csv_file, delimiter=',')
				csv_writer.writerow(['x','y'])
				mask_csv = os.path.join(path_to_masks, arg)
				maskpts=pd.read_csv(mask_csv)

				for i in range(max(maskpts['mask_id'])):
					#create mask
					mask = getMask(maskpts, i)
					#get centroids
					label_image = label(mask)
					for region in regionprops(label_image):
						coords = region.centroid
						csv_writer.writerow([coords[1], coords[0]])


def getMask(maskpts, n_id):
	my_list=maskpts.loc[maskpts['mask_id']==n_id]
	mask=np.zeros([512,512],dtype='int')
	mask[my_list['y'], my_list['x']]=1
	return mask

path_to_masks = "PATH_TO_MASKS"
centroids_path = "PATH_TO_SAVE_CENTROIDS"

if __name__ == "__main__":
	getCentroid(path_to_masks, centroids_path)