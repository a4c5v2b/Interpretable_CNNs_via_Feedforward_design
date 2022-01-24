import numpy as np
from skimage.util.shape import view_as_windows

from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce

import matplotlib.pyplot as plt

def parse_list_string(list_string):
	"""Convert the class string to list."""
	elem_groups=list_string.split(",")
	results=[]
	for group in elem_groups:
		term=group.split("-")
		if len(term)==1:
	  		results.append(int(term[0]))
		else:
	  		start=int(term[0])
	  		end=int(term[1])
	  		results+=range(start, end+1)
	return results

# convert responses to patches representation
def window_process(samples, wp_kernel_size, stride):
	'''
	Create patches
	:param samples: [num_samples, feature_height, feature_width, feature_channel]
	:param kernel_size: int i.e. patch size
	:param stride: int
	:return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

	'''
	n, h, w, c = samples.shape # (10000, 32, 32, 1)
	output_h = (h - wp_kernel_size)//stride + 1  # ((32-5)//1) + 1 = 28
	output_w = (w - wp_kernel_size)//stride + 1 # ((32-5)//1) + 1 = 28
	patches = view_as_windows(samples, (1, wp_kernel_size, wp_kernel_size, c), step=(1, stride, stride, c)) # (10000, 28, 28, 1, 1, 5, 5, 1)
	patches = patches.reshape(n, output_h, output_w, c*wp_kernel_size*wp_kernel_size) #(10000, 28, 28, 25)
	return patches

def remove_mean(features, axis):
	'''
	Remove the dataset mean.
	:param features [num_samples,...]
	:param axis the axis to compute mean
	
	'''
	feature_mean=np.mean(features,axis=axis,keepdims=True)
	feature_remove_mean=features-feature_mean
	return feature_remove_mean,feature_mean

def select_balanced_subset(images, labels, use_num_images, use_classes): # Make the dataset's class distribution becomes uniform (Each class has 1000 samples)
	'''
	select equal number of images from each classes
	'''
	# Shuffle
	num_total=images.shape[0]
	shuffle_idx=np.random.permutation(num_total)
	images=images[shuffle_idx]
	labels=labels[shuffle_idx]

	num_class=len(use_classes)
	num_per_class=int(use_num_images/num_class)
	selected_images=np.zeros((use_num_images,images.shape[1],images.shape[2],images.shape[3]))
	selected_labels=np.zeros(use_num_images)
	for i in range(num_class):
		images_in_class=images[labels==i]
		selected_images[i*num_per_class:(i+1)*num_per_class]=images_in_class[:num_per_class]
		selected_labels[i*num_per_class:(i+1)*num_per_class]=np.ones((num_per_class))*i

	# Shuffle again
	shuffle_idx=np.random.permutation(num_per_class*num_class)
	selected_images=selected_images[shuffle_idx]
	selected_labels=selected_labels[shuffle_idx]
	# For test
	# print(selected_images.shape)
	# print(selected_labels[0:10])
	# plt.figure()
	# for i in range (10):
	# 	img=selected_images[i,:,:,0]
	# 	plt.imshow(img)
	# 	plt.show()
	"""
	  Showing the class distribution of the original train data
	  """
	sum = 0
	for m in range(num_class):
		images_in_class = images[labels == m]
		print("Number of samples of class {} is: {}".format(m, images_in_class.shape[0]))
		sum += images_in_class.shape[0]
	print("Sum = ", sum)

	"""
	   Showing the class distribution of the train data after the redistribution above
	   """
	sum = 0
	for m in range(num_class):
		images_in_class = selected_images[selected_labels == m]
		print("Number of samples of class {} is: {}".format(m, images_in_class.shape[0]))
		sum += images_in_class.shape[0]
	print("Sum = ", sum)


	return selected_images,selected_labels

def find_kernels_pca(samples, num_kernels, energy_percent): #Samples.shape = (7840000, 25)
	'''
	Do the PCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.

	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels, sample_mean
	'''

	pca=PCA(n_components=samples.shape[1], svd_solver='full') # Decrease Dimension to 25
	#sklearn expects input shape = (n_samples,n_features)
	pca.fit(samples) # (7840000, 25)

	# Compute the number of kernels corresponding to preserved energy
	if  energy_percent:
		energy=np.cumsum(pca.explained_variance_ratio_)
		num_components=np.sum(energy<energy_percent)+1
	else:
		num_components=num_kernels #5 in first layer, 15 in second layer.

	kernels=pca.components_[:num_components,:] #(5,25) We just need the five PC having the largest variance. num of features.
											   # pca.components_ is the set of all eigenvectors (aka loadings) for your projection space
	mean=pca.mean_ #(25,)

	print("Num of kernels: %d"%num_components)
	print("Energy percent: %f"%np.cumsum(pca.explained_variance_ratio_)[num_components-1])
	return kernels, mean


def multi_Saab_transform(images, labels, kernel_sizes, num_kernels, energy_percent, use_num_images, use_classes):
	'''
	Do the PCA "training".
	:param images: [num_images, height, width, channel]
	:param labels: [num_images]
	:param kernel_sizes: list, kernel size for each stage,
	       the length defines how many stages conducted
	:param num_kernels: list the number of kernels for each stage,
	       the length should be equal to kernel_sizes.
	:param energy_percent: the energy percent to be kept in all PCA stages.
	       if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and mean
    '''

	num_total_images=images.shape[0] #60000
	if use_num_images<num_total_images and use_num_images>0: # 10000 < 60000 and 10000 > 0
		sample_images, selected_labels=select_balanced_subset(images, labels, use_num_images, use_classes)
	else:
		sample_images=images
	# sample_images=images
	num_samples=sample_images.shape[0] # 10000
	num_layers=len(kernel_sizes) # 2 layers
	pca_params={}
	pca_params['num_layers']=num_layers
	pca_params['kernel_size']=kernel_sizes

	for i in range(num_layers):
		print('--------stage %d --------'%i)
    	# Create patches
		# sample_patches=window_process(sample_images,kernel_sizes[i],kernel_sizes[i]) # nonoverlapping
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping # (10000, 28, 28, 25)
		h=sample_patches.shape[1] #28
		w=sample_patches.shape[2] #28
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])  # (7840000, 25)

    	# Remove feature mean (Set E(X)=0 for each dimension)
		sample_patches_centered, feature_expectation=remove_mean(sample_patches, axis=0) #(7840000, 25), (1, 25)
    	# Remove patch mean
		training_data, dc=remove_mean(sample_patches_centered, axis=1) #(7840000, 25), (1, 25)

    	# Compute PCA kernel
		if not num_kernels is None: # If num_kernels is not None
			num_kernel=num_kernels[i]
		kernels, mean=find_kernels_pca(training_data, num_kernel, energy_percent) # (5,25)

    	# Add DC kernel
		num_channels=sample_patches.shape[-1]  # 25
		dc_kernel=1/np.sqrt(num_channels)*np.ones((1,num_channels)) #(1,25)
		kernels=np.concatenate((dc_kernel, kernels), axis=0) # (6,25)

		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered, np.transpose(kernels))  # (7840000, 25)*(25, 6) =(7840000, 6)
		else:
	    	# Compute bias term
			bias=LA.norm(sample_patches, axis=1) # (7840000,)
			bias=np.max(bias)
			pca_params['Layer_%d/bias'%i]=bias
			# Add bias
			sample_patches_centered_w_bias=sample_patches_centered+1/np.sqrt(num_channels)*bias # (7840000, 25)  #Why second layer need to do this?
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels)) # (7840000, 25)*(25, 6)=(7840000, 6)
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0])) #(1, 6)  kl
			e[0,0]=1
			transformed-=bias*e  # transofmred = transformed - (bias*e) # (7840000, 6) # Remove DC bias **** Why do this?

    	# Reshape: place back as a 4-D feature map
		sample_images=transformed.reshape(num_samples, h, w,-1) # (10000, 28, 28, 6)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max) # (10000, 14, 14, 6)


		print('Sample patches shape after flatten:', sample_patches.shape) # (7840000, 25)
		print('Kernel shape:', kernels.shape) # (6,25)
		print('Transformed shape:', transformed.shape) # (7840000, 6)
		print('Sample images shape:', sample_images.shape)  # # (10000, 14, 14, 6)
    	
		pca_params['Layer_%d/feature_expectation'%i]=feature_expectation
		pca_params['Layer_%d/kernel'%i]=kernels
		pca_params['Layer_%d/pca_mean'%i]=mean

	return pca_params

# Initialize
def initialize(sample_images, pca_params):

	num_layers=pca_params['num_layers']
	kernel_sizes=pca_params['kernel_size']

	for i in range(num_layers):
		print('--------stage %d --------'%i)
		# Extract parameters
		feature_expectation=pca_params['Layer_%d/feature_expectation'%i]
		kernels=pca_params['Layer_%d/kernel'%i]
		mean=pca_params['Layer_%d/pca_mean'%i]

    	# Create patches
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping
		h=sample_patches.shape[1]
		w=sample_patches.shape[2]
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

    	# Remove feature mean (Set E(X)=0 for each dimension)
		sample_patches_centered, feature_expectation=remove_mean(sample_patches, axis=0)
		# sample_patches_centered=sample_patches-feature_expectation
    	
    	# Remove patch mean
		training_data, dc=remove_mean(sample_patches_centered, axis=1)

		num_channels=sample_patches.shape[-1]
		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered, np.transpose(kernels))
		else:
			bias=pca_params['Layer_%d/bias'%i]
			# Add bias
			sample_patches_centered_w_bias=sample_patches_centered+1/np.sqrt(num_channels)*bias
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0]))
			e[0,0]=1
			transformed-=bias*e
    	
    	# Reshape: place back as a 4-D feature map
		num_samples=sample_images.shape[0]
		sample_images=transformed.reshape(num_samples, h, w,-1)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max)

		print('Sample patches shape after flatten:', sample_patches.shape)
		print('Kernel shape:', kernels.shape)
		print('Transformed shape:', transformed.shape)
		print('Sample images shape:', sample_images.shape)
	return sample_images




