import numpy as np

# Paper reference: https://www.sciencedirect.com/science/article/abs/pii/S0925231217313371?via%3Dihub

# Uncertainty Sampling
def bordas_sample_score(ranking_matrix):
    """
    Score for each sample based on its ranking in every class. 
    Args:
        inp_array: 2D ranking matrix where each entry denotes sample's rank in that class

    Returns:

    """	
	U_s = np.shape(ranking_matrix)[0]
	q = np.shape(ranking_matrix)[-1]
	scores = np.sum((U_s - ranking_matrix), axis = -1) / (q*( U_s - 1))
	return scores

def get_ranking(inp_array):
    """
    Ranking of samples based on prediction scores of a certain class
    Args:
        inp_array: 1D array of size (number of samples, 1) of a certain class

    Returns:

    """	
	temp = inp_array.argsort()
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(inp_array))+1
	return ranks

def uncertainty_sampling(prediction_matrix):
    """
    Uncertainty sampling on a 2D prediciton matrix
    Args:
        prediction_matrix: raw unlabelled predictions

    Returns:

    """
	margin_matrix = np.absolute((2*prediction_matrix) - 1)
	ranking_matrix = np.zeros(np.shape(margin_matrix))
	for i in range(np.shape(margin_matrix)[-1]):
		ranking_matrix[:, i] = get_ranking(margin_matrix[:, i])
	sample_scores = bordas_sample_score(ranking_matrix)

	return sample_scores
