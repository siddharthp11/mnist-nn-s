# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        C = self.num_classes
        D = self.input_size
        H = self.hidden_size
        N = X.shape[0]  


        X = np.concatenate((X, np.ones((N, 1))), axis = -1)                             #(N, D + 1)
        w1 = np.concatenate((self.weights['W1'], self.weights['b1'].reshape(1, H)))     #(D+1, H)
        w2 = np.concatenate((self.weights['W2'], self.weights['b2'].reshape(1, C)))     #(H + 1, C)

        z1 = X @ w1                     #(N, H)
        s1 = super().sigmoid(z1)        #(N, H)
        a1 = np.concatenate((s1, np.ones((N, 1))), axis = -1) #(N, H+1)

        z2 = a1 @ w2                #(N, C)
        s2 = super().softmax(z2)    #(N, C)

        loss = super().cross_entropy_loss(s2, y)
        accuracy = super().compute_accuracy(s2, y)

        if mode != 'train': return loss, accuracy
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        
        #W2

        dL_dz2 = (s2 - np.eye(C)[y]).transpose()[:, np.newaxis, :]      #(N, C)->  #(C, N) ->  #(C, 1, N)
        
        dz_dw2 = a1.transpose()[np.newaxis, :, :]                       #(N, H+1) -> (H+1, N) -> (1, H+1, N)

        dL_dw2 = np.mean((dL_dz2 * dz_dw2), axis = -1).transpose()      #(C, H+1, N) -> (C, H+1) -> (H+1, C)

        self.gradients['W2'], self.gradients['b2'] = dL_dw2[:H, :], dL_dw2[H, :].reshape(C,)

        #W1  
        
        dL_da1 = w2[:H, :] @ dL_dz2.reshape((C, N))        #(H, C) X (C, N) = (H, N)
        
        da_dz1 = self.sigmoid_dev(z1).transpose()          #(N, H) ->(H, N)
        
        dL_dz1 = (dL_da1 * da_dz1) [np.newaxis, :, :]      #(H,N) -> (1, H, N)

        X = X.transpose()[:, np.newaxis, :]                #(D+1, 1, N)
       
        dL_dw1 = np.mean(X * dL_dz1, axis = -1)            #(D+1, H, N) -> (D+1, N)
        
        self.gradients["W1"], self.gradients['b1'] = dL_dw1[:D, :], dL_dw1[D, :].reshape(H,)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


