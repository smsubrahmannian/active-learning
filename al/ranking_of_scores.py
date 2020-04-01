import numpy as np
from numba import njit, prange

# Uncertainty Sampling
@njit
def bordas_sample_score(ranking_matrix):
	U_s, q = ranking_matrix.shape()
	scores = np.sum((U_s - ranking_matrix), axis = -1) / (q*( U_s - 1))
	return scores
@njit
def get_ranking(inp_array):
	temp = inp_array.argsort()
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(inp_array))
	return ranks

@njit(parallel = True)
def uncertainty_sampling(prediction_matrix):
	margin_matrix = np.absolute((2*prediction_matrix) - 1)
	ranking_matrix = np.zeros(np.shape(margin_matrix))
	for i in prange(margin_matrix.shape()[-1]):
		ranking_matrix[:, i] = get_ranking(margin_matrix[:, i])
	sample_scores = bordas_sample_score(ranking_matrix)

	return sample_scores
