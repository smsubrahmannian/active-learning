import numpy as np

# Representative Sampling
def matrix_cosine(x, y):
	return np.einsum('ij,ij->i', x, y) / (
			  np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
	)

def get_representative_samples(labeled_data, unlabeled_data):
	"""Gets the most representative unlabeled items, compared to training data

	Keyword arguments:
		training_data -- data with a label, that the current model is trained on
		unlabeled_data -- data that does not yet have a label
		num_samples -- number of items to sample

	Creates one cluster for each data set: training and unlabeled
	""" 
	labeled_centroid = np.sum(labeled_data, axis = 0)/np.shape(labeled_data)[0]
	unlabeled_centroid = np.sum(unlabeled_data, axis = 0)/np.shape(unlabeled_data)[0]	
	
	# repeat centroids to match unlabeled data size
	l_centroid_mat = np.tile(labeled_centroid, (np.shape(unlabeled_data)[0], 1))
	u_centroid_mat = np.tile(unlabeled_centroid, (np.shape(unlabeled_data)[0], 1))

	# calculate cosine similarity between samples to train and unlabeled
	labeled_scores = matrix_cosine(unlabeled_data, l_centroid_mat)
	unlabeled_scores = matrix_cosine(unlabeled_data, u_centroid_mat)

	representativeness_scores = unlabeled_scores - labeled_scores
	
	return representativeness_scores
