#implementation for the output readout layer of the ESN (only trained part)
import numpy as np
import scipy

#the regularization amount for tihkonov regularization (ride regression) 
reg = 1e-8

#method to create the output layer of the matrix (ridge regression)
#uses tihknov regression
#@param  X          : the states matrix ( res_size X num_states_collected  )
#@param  outputs    : what the machine should try to produce
#@return            : the (output_size,res_size) matrix of output weights
def create ( X, train_truth ):
    res_size = X.shape[0]
    #this is just the formula for ridge reg. solving the ax=b
    X_T   = X.T
    Yt    = np.array( train_truth ).reshape( 1, len(train_truth) )
    term1 = np.dot( Yt, X_T )
    term2 = scipy.linalg.inv( np.dot(X,X_T) + reg*scipy.eye(res_size) )
    W_out = np.dot( term1, term2 )
    return W_out
