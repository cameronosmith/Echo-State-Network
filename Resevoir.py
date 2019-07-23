#file contains the implementation of the actual resevoir (middle nodes stuff)
#also contains a mini node class which is the base unit in the resevoir
import numpy as np
import scipy.linalg as linalg
import matplotlib.pylab as plt

range_shift = .3

class Resevoir : 

    #class to be the basic math unit of the resevoir
    class Resevoir_Node : 

        def __init__( self, resevoir ):

            self.resevoir = resevoir

            #the hidden state for this node ( for now just a 1x1 val )
            self.state    = 0

        #method to update the hidden state for this node
        #@param weighted_data: the weights_in * data since not unique to node
        #@param adj_nodes    : list of weights (row of adj matrix)
        #@return             : none, just updates this node's hidden state
        def run ( self, weighted_data, weights_row ):
            
            #the collected aggregate of val*weight of hidden inputs from adj's 
            adj_sum = 0

            #get the inputs from all of the nodes that feed into here
            for node_idx, weight in enumerate( weights_row ):
                adj_node = self.resevoir.res_nodes[ node_idx ]
                adj_sum += adj_node.state * weight
            
            #perform update to our hidden 
            hidden_update   = np.tanh( weighted_data + adj_sum )
            self.state      = (1-self.resevoir.leak_rate)*self.state + \
                                self.resevoir.leak_rate*hidden_update
    
    #constructor
    #@param res_size     : the number of hidden nodes to use    
    #@param connectivity : the connectivity of the nodes as float
    #@param leak_rate    : how much the nodes states should flow into next
    def __init__( self, res_size, connectivity,  leak_rate ):

        self.res_size  = res_size 
        self.leak_rate = leak_rate

        #create our weight matrix and res nodes
        self.res_nodes    = [self.Resevoir_Node(self) for _ in range(res_size)]
        self.res_weights  = self.init_res_weights(self.res_nodes, connectivity)
        
        #limit the spectral radius of our node weights
        print("computing max eigen val...")
        max_eigen         = max( abs( linalg.eig( self.res_weights )[0] ) ) 
        self.res_weights /= max_eigen
        print("done limiting spectral radius (max eigen)")
    
    #helper method to setup adjacency matrix of the resevoir aka node weights
    #@param res_nodes   : the nodes in the resevoir
    #@param connectivity: float - the percentage of nodes a node should be conn.
    #@param plot        : true if we want to show plot of the adjacency matrix
    #@return            : the adj. matrix of nodes.connection is [0,1)
    def init_res_weights( self, res_nodes, connectivity, plot=True ):

        #setup weights from default to [0,1) to [.5,1]
        weights = np.random.rand( self.res_size, self.res_size ) + range_shift
        weights = np.clip( weights, None, 1 )

        #pick the rand indices each node will drop (limiting connectivity)
        num_nodes_dropped = round( self.res_size * (1-connectivity) )
        for row in weights : 
            drop_indices = np.random.choice( self.res_size, \
                                        num_nodes_dropped, replace = False )
            row[ drop_indices ] = 0

        #plot adj matrix for illustration if desired
        if plot :
            plt.imshow(weights)
            plt.colorbar()
            plt.show()

        weights = np.random.rand( len(res_nodes), len(res_nodes) )
         
        return weights
    
    #method to run the resevoir (throw data into the resevoir) 
    #@param weights_in  : the input weights
    #@param data_t      : the data to pass through the resevoir on 
    #@return            : none 
    def run (self, weights_in, data_t ):
        
        #compute here so every node doesn't have to compute it
        weighted_data = np.dot( weights_in, data_t )

        #run each node to get new activation states
        for row_i, node_i in enumerate( self.res_nodes ) :
            node_i.run( weighted_data[row_i,0], self.res_weights[ row_i, : ] )

    #getter for the array of hidden states
    #@return : the (res_sizeX1) matrix of hidden states
    def get_hidden_states( self ):

        #format hidden states as np array
        raw_states  = [ node.state for node in self.res_nodes ]
        formatted   = np.array( raw_states ).reshape( (self.res_size, 1) )

        return formatted

