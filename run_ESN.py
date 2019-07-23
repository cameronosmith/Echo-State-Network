#method to run/test our esn with some data

import numpy as np
from EchoStateNetwork import EchoStateNetwork
import matplotlib.pylab as plt

#get our data and format it and such ya know
data                = np.loadtxt( "data/MackeyGlass_t17.txt" )
train_len           = 2000
test_len            = 200
train_data_input    = data[:train_len]
train_data_output   = data[1:train_len+1]
test_data_input     = data[train_len:train_len+test_len]
test_data_output    = data[train_len+1:train_len+test_len+1]

#create our ESN and run it. our input and outputs are just 1 number
machine = EchoStateNetwork( 1, 1 )
machine.train( train_data_input, train_data_output )
outputs, err = machine.test( test_data_input, test_data_output )
print("error is: ",err)

#plot the outputs of our machine against the truth
time_seq = list(range(test_len))
plt.plot(time_seq, test_data_output, label="truth")
plt.plot(time_seq, outputs, label="machine outputs")
plt.legend(loc='best')
plt.show()
