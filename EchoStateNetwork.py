#this file is the wrapper class that runs the resevoir and such. :P
import numpy as np
import Resevoir
import OutputLayer

lucky_num       = 42
np.random.seed( lucky_num )
range_shift     = .5 #to move from [0,1) to [-.5,.5)

#properties of our ESN
input_size       = 1
output_size      = 1
res_size         = 100
leak_rate        = .3
connectivity     = .2

update_info_interval  = 100

#class to create the resevoir and run the computations 
class EchoStateNetwork : 

    #constructor
    #@param input_size  : the length of a single input data 
    #@param output_size : the length of a single output data 
    def __init__( self, input_size, output_size ):

        self.input_size  = input_size
        self.output_size = output_size

        #init our random weights and resevoir
        self.weights_in  = np.random.rand ( res_size, input_size ) - range_shift 
        self.resevoir    = Resevoir.Resevoir( res_size, connectivity, leak_rate )
        self.weights_out = None

    #method to run the network on some input data and get the machine output
    #used by testing but not training cus training needs to get the state matrix
    #@param dataset : the dataset to feed into the network
    #@return        : the output of the system
    def run ( self, dataset ):
         
        outputs = []

        #run the machine and compute output
        for t, data_t in enumerate( dataset ) : 
            if t % update_info_interval == 0 :
                print("running on data idx: ",t," out of ",len(dataset))
            self.resevoir.run( self.weights_in, data_t )
            output = np.dot(self.weights_out, self.resevoir.get_hidden_states())
            outputs.append( output )
        
        return outputs
         
    
    #method to train the network (make our output weights)
    #@param train_input : the data to train our network for
    #@param train_output: the data our network should strive to output
    #@return            : none
    def train ( self, train_input, train_output ):

        #holds our states at every time step (used to construct outputs weights)
        states_matrix = np.zeros( (res_size, len(train_input)) )

        #go run the res to collect the states matrix  
        for time_t, data_t in enumerate( train_input ) :
            if time_t % update_info_interval == 0 :
                print("on training data idx: ",time_t," out of ",len(train_input))
            self.resevoir.run( self.weights_in, data_t )
            states_matrix[:,time_t] = self.resevoir.get_hidden_states()[:,0]

        #create our output layer based on our states matrix
        print("creating output layer...")
        self.weights_out = OutputLayer.create ( states_matrix, train_output )
        print("output layer found.")

    #method to test/validate the network (check our error)
    #@param test_input  : the data to feed our machine to get output
    #@param test_truth  : the data our network should have outputted
    #@return            : the outputs, and our total error
    def test ( self, test_input, test_truth ):
    
        #make sure we have output weights
        if self.weights_out is None : 
            print("error: haven't trained yet")
            return
        
        #run the machine and collect the outputs and compute mean sq error
        machine_outputs = self.run( test_input )
        error = [ abs(x-y) for x,y in zip( machine_outputs, test_truth ) ] 
        mse   = np.sum( error ) / len( machine_outputs )
        machine_outputs = [x[0,0] for x in machine_outputs]

        return machine_outputs, mse*100
