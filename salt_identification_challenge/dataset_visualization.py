import os
import numpy as np
import imageio
import pandas as pd
import torch
from torch.utils import data

datapath = '../../../../'
root_path =os.path.join(datapath, '/mnt/data/rajiv/akash/salt_identification_challenge/')

class TGSsaltDataset(data.Dataset):
	# Input Dataset

	def __init__(self, root_path, file_list):
		# root_path - Location of dataset ,List of files - Images
		self.root_path = root_path
		self.file_list = file_list

	def __len__(self):
		return len(self.file_list)


	def __getitem(self, index):
		# retireve the data at that index
		file_id = self.file_list[index]

		# image folder + path
		image_folder = os.path.join(self.root_path, 'images')
		image_path = os.path.join(image_folder, file_id + '.png')

		# label folder + path
		mask_folder = os.path.join(self.root_path, 'masks')
		mask_path = os.path.join(mask_folder, file_id + '.png')


		# read it vectorized form
		image = np.array(imageio.imread(image_path), dtype=np.uint8)
		mask = np.array(imageio.imread(mask_path), dtype=np.uint8)


		return image, mask


# init our new class dataset
train_mask = pd.read_csv(root_path + 'train.csv')
depth = pd.read_csv(root_path + 'depths.csv')

# print(train_mask.head())
# print(depth.head())


train_path = './'

file_list = list(train_mask['id'].values)

print(file_list)
dataset = TGSsaltDataset(train_path, file_list)

# # image visualizing
# def plot2x2array(image, mask):
# 	# invoke matplotlib
# 	f, axarr = plt.subplots(1, 2)
# 	axarr[0].imshow(image)
# 	axarr[1].imshow(mask)
# 	axarr[0].grid()
# 	axarr[1].grid()
# 	axarr[0].set_title('Image')
# 	axarr[1].set_title('Mask')


# for i in range(5):
# 	image, mask = dataset[np.random.randn(0, len(dataset))]
# 	plot2x2array(image, mask)

# # Matrix of pixel values to learn and learn the label


# plt.figure(figsize = (6, 6))
# plt.hist(depth['z'], bins=50)
# plt.title('depth_distribution')

# # Run length encoding losslessdata compression in which runs of data
# # how data is 

# # convert run length to images that can be input to our model

# def rleToMask(rleString, height, width):
# 	# width, height
# 	rows, cols = height, width

# 	try:
# 		rleNumbers = [int(numstring) for numstring in rleString(' ')]
# 		rlePairs = np.array(rleNumbers).reshape(-1, 2)
# 		img = np.zeros(rows*cols, dtype=np.uint8)

# 		# fill that image with values

# 		for index, length in rlePairs:
# 			index-=1
# 			img[index:index+length] = 255

# 		img =img.reshape(cols, rows)
# 		img = img.T

# 	except:
# 		img = np.zeros((cols, rows))

# 	return img


# # function for measuring how salty an image is
# def salt_proportion(imgArray):
# 	try:
# 		unique, counts = np.unique(imgArray, return_counts=True)
# 		return counts[1]/10201.

# 	except:
# 		return 0.0

# # prepare to merge depth
# train_mask['mask'] = train_mask['rle_mask'].apply(lambda x: rleToMask(x, 101, 101))
# train_mask['salt_proportion'] = train_mask['mask'].apply(lambda x:salt_proportion(x))

# # merge 
# merged = train_mask.merge(depth, how='left')
# merged.head()

# # show proportion of salt vs depth

# plt.figure(figsize = (12, 6))
# plt.scatter(merged['salt_proportion'], merged['z'])
# plt.title('Proportion of salt vs depth')

# correlation



		