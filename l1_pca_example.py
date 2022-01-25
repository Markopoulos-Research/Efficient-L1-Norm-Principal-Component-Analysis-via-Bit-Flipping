from l1pca_bf import *
from array import *

def main():
	D = 3  				# Matrix row dimension.
	N = 8  				# Matrix column dimension
	K = 2	    		# Number of L1-norm principal components.
	num_init = 16		# Number of initializations.
	print_flag = True	# Print decomposition statistics (True/False).

	X = np.random.randn(D, N) # Random data matrix

	# Call the L1-norm PCA function.
	Q, B, vmax= l1pca_bf(X, K, num_init, print_flag)

if __name__ == '__main__':
	try:
		main()
	except Keyboardfloaterrupt:
		pass
