import data
import pickle
import numpy as np
import sklearn
import cv2
#import keras
import tensorflow.keras as keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def main():
	# load data
	fr=open('pca_params.pkl','rb')
	pca_params=pickle.load(fr, encoding='latin')
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape) # (60000, 32, 32, 1)
	print('Testing_image size:', test_images.shape) # (10000, 32, 32, 1)

	# load feature
	fr=open('feat.pkl','rb')
	feat=pickle.load(fr, encoding='latin')
	fr.close()
	feature=feat['feature']
	print("S4 shape:", feature.shape) # (60000, 400)
	print('--------Finish Feature Extraction subnet--------')

	# feature normalization
	std_var=(np.std(feature, axis=0)).reshape(1,-1) # (1, 400)
	feature=feature/std_var # (60000, 400)

	num_clusters=[120, 84, 10] # class of each layer
	use_classes=10
	weights={}
	bias={}
	for k in range(len(num_clusters)):
		if k!=len(num_clusters)-1: # if not the last output layer
			# Kmeans_Mixed_Class (too slow for CIFAR, changed into Fixed Class)
			kmeans=KMeans(n_clusters=num_clusters[k]).fit(feature)
			pred_labels=kmeans.labels_ #(60000, )
			num_clas=np.zeros((num_clusters[k],use_classes)) # (120, 10) for 1st layer
			for i in range(num_clusters[k]): # 0-119
				for t in range(use_classes): # 0-9
					for j in range(feature.shape[0]): # 0-59999
						if pred_labels[j]==i and train_labels[j]==t:  # if j = 13, pred_labels[13]=114, class=114, train_labels[13]=6
							#if k == 0 and j < 20:        #j = 10, pred_labels[10] = 20, class =20, train_labels[10] = 10
								#print(j, "i={},t={}".format(i,t))
							num_clas[i,t]+=1 # num_clas[114,6]+1
			acc_train=np.sum(np.amax(num_clas, axis=1))/feature.shape[0]  # calculate the accuracy # The label that has the two classes matched/number of samples
			print(k,' layer Kmean (just ref) training acc is {}'.format(acc_train))


			# Original label: 60000, 10 classes [0,3,4,...9] , K means: 60000, 120 classes (60000, ) [0,12,119,50...

			# Compute centroids
			clus_labels=np.argmax(num_clas, axis=1)  #(120,)
			centroid=np.zeros((num_clusters[k], feature.shape[1])) # (120, 400) # The distance between the 120 centroids and the 400 features.
			for i in range(num_clusters[k]): # 0-119
				t=0
				for j in range(feature.shape[0]): # 0-59999
					if pred_labels[j]==i and clus_labels[i]==train_labels[j]: # e.g. j=59978, i=119,t=513 if pred_labels[59978]=119. pred_class = 119. clus_labels[119] = train_labels[59978] = 4
						#print(j, "i={},t={}".format(i, t)) # The 59978th sample's pred_label class is 119. Its predicted output class is equal to the original label class which are both 4.
						if t==0:
							feature_test=feature[j].reshape(1,-1) # feature is (60000, 400) After reshape, feature_test is : (1,400).
																  # We extracted the samples that jts predicted final output class is equal to the original label class feature.
																  # E.g. The 59978th sample features.
						else:
							feature_test=np.concatenate((feature_test, feature[j].reshape(1,-1)), axis=0) # For the cluster 0, (The pred_labels = 0) (469, 400)
																										  # Only 469th samples' predicted final output class and the original class is matched.
						t+=1
				centroid[i]=np.mean(feature_test, axis=0, keepdims=True) # (120, 400) # feature_test is the sample that has the correct predicted final output label.
																		 # We find out the centroid of each cluster by finding the means of 400 features of each clusters.
																		 # e.g. For cluster 0, the pred_labels = 0 class. The centroid of 0 class is to calculate the mean of all samples that have this pred_label row by row.
																		 # So we can get the mean of all 400 features of all samples that are grouped in group 0 which is the centroid of group 0.
																		 # We repeat this step until the centroids of all 120 pred_labels are found.

			# Compute one hot vector
			t=0
			labels=np.zeros((feature.shape[0], num_clusters[k])) #(60000, 120)
			for i in range(feature.shape[0]):
				if clus_labels[pred_labels[i]]==train_labels[i]: # i=59999, clus_labels[pred_labels[i]]=8, pred_labels[i]=31,train_labels[i]=8
																 # The 59999th sample's pred_label is 31. Its original class is 8. If the clus_labels[31] = 8, means that the pred_label class 31 is a correct class.
					# if k == 0:
					# 	print("i={}, clus_labels[pred_labels[i]]={}, pred_labels[i]={},train_labels[i]={}".format(i,clus_labels[pred_labels[i]],pred_labels[i],train_labels[i]))
					labels[i,pred_labels[i]]=1                   # Then, labels[59999, 31] = 1, means for the 59999th sample, its pseudo class in this layer is 31. But we represent it in one hot form.
																 # e.g.: [30 zeros,...,1,0,0,...0]
				else:
					distance_assigned=euclidean_distances(feature[i].reshape(1,-1), centroid[pred_labels[i]].reshape(1,-1)) # Shape of feature[i]: (1,400) Shape of centroid[pred_labels[i]: (1, 400)
					cluster_special=[j for j in range(num_clusters[k]) if clus_labels[j]==train_labels[i]]  # If the sample pred_label can't link to the original correct label, then we find out which classes can link to the correct class.
																											# E.g.: k=0, i = 2 (2nd sample),  j=4
																											# clus_labels[4] = train_labels[2] = 4 but pred_labels[2] = 81 and clus_labels[81] = 2 not 4.
																											# For 2nd sample, it has 12 this type labels.
					distance=np.zeros(len(cluster_special)) # For 2nd sample: (12, )
					for j in range(len(cluster_special)):
						distance[j]=euclidean_distances(feature[i].reshape(1,-1), centroid[cluster_special[j]].reshape(1,-1)) # Shape of feature[i]: (1, 400) For 2nd sample: Shape of centroid[j]: (1, 12)
					labels[i, cluster_special[np.argmin(distance)]]=1  # Calculate the distances between all centroid and all special_cluster. We use the shortest distance one to be the group of that sample.

			# least square regression
			A=np.ones((feature.shape[0],1)) # (60000, 1)
			feature=np.concatenate((A,feature),axis=1) #  Original feature shape: (60000, 400), After reshape: (60000, 401)
			weight=np.matmul(LA.pinv(feature),labels)# (401, 60000) *(60000, 120) = (401, 120)
			feature=np.matmul(feature,weight) # (60000, 401)*(401, 120) = (60000, 120)
			weights['%d LLSR weight'%k]=weight[1:weight.shape[0]] # Exclude the first row of the weight
			bias['%d LLSR bias'%k]=weight[0].reshape(1,-1) # The first row of the weight is bias
			print(k,' layer LSR weight shape:', weight.shape) # (401, 120)
			print(k,' layer LSR output shape:', feature.shape) # (60000, 120)

			pred_labels=np.argmax(feature, axis=1) # (60000, )
			num_clas=np.zeros((num_clusters[k],use_classes)) # (120, 10)
			for i in range(num_clusters[k]): # 0-119
				for t in range(use_classes): # 0-9
					for j in range(feature.shape[0]): #0-59999
						if pred_labels[j]==i and train_labels[j]==t:
							if k == 0 and j < 20:        # e.g.: j = 11 i = 12,t = 5, pred_labels[11] = 12, train_labels[11] = 5
								print(j, "i={},t={}".format(i,t))
							num_clas[i,t]+=1
			acc_train=np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
			print(k,' layer LSR training acc is {}'.format(acc_train))

			# Relu
			for i in range(feature.shape[0]): # the shape of feature: (60000, 120) # 0-59999
				for j in range(feature.shape[1]): # 0-119
					if feature[i,j]<0:
						feature[i,j]=0

			# # Double relu
			# for i in range(feature.shape[0]):
			# 	for j in range(feature.shape[1]):
			# 		if feature[i,j]<0:
			# 			feature[i,j]=0
			# 		elif feature[i,j]>1:
			# 			feature[i,j]=1
		else: # Final output layer
			# least square regression
			labels=keras.utils.to_categorical(train_labels,10) #To do the one hot vector form. Shape = (60000, 10)
			A=np.ones((feature.shape[0],1)) #(60000, 1) The bias term
			feature=np.concatenate((A,feature),axis=1)
			weight=np.matmul(LA.pinv(feature),labels)
			feature=np.matmul(feature,weight)
			weights['%d LLSR weight'%k]=weight[1:weight.shape[0]]
			bias['%d LLSR bias'%k]=weight[0].reshape(1,-1)
			print(k,' layer LSR weight shape:', weight.shape)
			print(k,' layer LSR output shape:', feature.shape)
			
			pred_labels=np.argmax(feature, axis=1)
			acc_train=sklearn.metrics.accuracy_score(train_labels,pred_labels)
			print('training acc is {}'.format(acc_train))
	# save data
	fw=open('llsr_weights.pkl','wb')    
	pickle.dump(weights, fw)    
	fw.close()
	fw=open('llsr_bias.pkl','wb')    
	pickle.dump(bias, fw)    
	fw.close()

if __name__ == '__main__':
	main()

