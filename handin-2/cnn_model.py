import os
import tensorflow as tf
import numpy as np
from model import Model as TfModel

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    width = 28
    height = 28
    n_features = width * height
    n_classes = 10
    ## Architecture configuration
    # Each conv layer is a convolution using h x w convolution, c, channels, and there are k of them. Stored in tuple (h, w, c, k)
    # The input is grayscale so the first layer has only one channel. For rgb color images there would be 3 for example.
    conv_layers = [(5, 5, 1, 32), (5, 5, 32, 64)]
    pool_sizes = [2, 2] # only use square pools i.e. t x t
    # compute the output size of the convolutional layers i.e. how many values do we get back after the two steps of convolution and pooling.    
    conv_output_size = int(0)
    
    padding = 2
    stride_conv = 1
    stride_pool = pool_sizes[0]
    height_conv = height
    values_conv = []
    values_pool = []
    
    for layer in conv_layers:
        activation_map = int(((height_conv + 2*padding) - layer[0])/stride_conv) + 1
        down_sampling =  int((height_conv - pool_sizes[0])/stride_pool) + 1
        values_conv.append((activation_map,activation_map,layer[3]))
        values_pool.append((down_sampling,down_sampling,layer[3]))
        height_conv = down_sampling

    conv_output_size = values_pool[1][0]*values_pool[1][1]*values_pool[1][2]

    hidden_size = 1024
    dropout = 0.5 #
    weight_decay = 1e-4
    batch_size = 32
    n_epochs = 5
    lr = 0.001
    
    def __init__(self, name_suffix=None, **kwargs):        
        ## overwrite values and set paths
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.name = 'cnn_h{0}'.format(self.hidden_size)
        if name_suffix is not None:
            self.name = '{0}_{1}'.format(self.name, name_suffix)
        weights_path = os.path.join(os.getcwd(), 'model_weights')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        self.weights_path = weights_path
        self.weights_file = os.path.join(self.weights_path, "{0}.weight".format(self.name))

class ConvolutionalModel(TfModel):
    
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None, ), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32
        weight_decay_placeholder: Weight Decay Value (scalar), type tf.float32

        Adding these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder
            self.weight_decay_placeholder

        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.weight_decay_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, weight_decay = 0, dropout=1):
        """Creates the feed_dict for the neural net.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        It is the same as you did for nn_model

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
            weight_decay: the lambda weight decay scale parameter
        Returns:
            feed_dict: dict, mapping from placeholders to values.
        """
        feed_dict={}
        feed_dict[self.input_placeholder]=inputs_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder]=labels_batch
        feed_dict[self.dropout_placeholder]=dropout
        feed_dict[self.weight_decay_placeholder]=weight_decay    
        return feed_dict

    def add_prediction_op(self):
        """Adds 2 layer convolution, 1-hidden-layer CNN:            
            l1 = max_pool(Relu(conv(x, C1)))
            l2 = max_pool(Relu(conv(l1, C2)))
            f = flatten(c2) - make into  [-1, self.config.conv_output_size] shape
            h = Relu(fW + b3) - hidden layer
            h_drop = Dropout(h, dropout_rate) - use dropout
            pred = h_dropU + b4 - compute output layer
                
        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits. Also it saves us some code.
        
        tf.reshape, and tf.nn.conv2d, tf.nn.max_pool should be vital for implementing the convolution layers.

        conv2d: tf.nn.conv2d
           filter is [filter_height, filter_width, in_channels, out_channels] and is configured in config.conv_layers
           set padding='SAME' to keep the input size unchaged
           Strides, we match the convolution filter to all positions in the input thus set strides to [1, 1, 1, 1] accordingly.

        max_pool: tf.nn.max_pool  
            ksize height and width are defined in config.pool_sizes (we use quadratic poolings) 
            strides must be the same as ksize to tile the max pool filter non-overlapping
            So strides and ksize should be 1, pool_size_height, pool_size_width, 1
            set padding='SAME' to keep the input size unchaged      
           
        Using tf.contrib.xavier_initializer to initialize Variablers C1, C2, W, U
        you can initialize bias b1, b2, b3, b4 with zeros

        Here are the dimensions of the various variables you will need to create
                    C1:    (first convolution) # conv_layers[0]
                    b1:    (number of convolutions in first conv. layer, ) 
                    C2:    (second convolutional layer)
                    b2:    (number of convolutions in second conv. layer, ) 
                    W:     (conv_output_size, hidden_layer)
                    b3:    (hidden_size,)
                    U:     (hidden_size, n_classes)
                    b4:    (n_classes,)
        
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument. 
            The keep probability should be set to the value of self.dropout_placeholder
        

        Adding these placeholders to self as the instance variables (need them for weigth decay)
            self.W
            self.W

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.input_placeholder
        x_image = tf.reshape(x, [-1, 28, 28, 1]) # (batchsize) inputs of 28 x 28 and 1 channel
        xavier_init = tf.contrib.layers.xavier_initializer()
               
        Wshape = (self.config.values_pool[1][0]**2*64, self.config.hidden_size)
        self.W = tf.Variable(xavier_init(Wshape))
        Ushape = (self.config.hidden_size, self.config.n_classes)
        self.U = tf.Variable(xavier_init(Ushape))
        C1shape = (self.config.conv_layers[0])
        C1 = tf.Variable(xavier_init(C1shape))
        C2shape = (self.config.conv_layers[1])
        C2 = tf.Variable(xavier_init(C2shape))
        b1 = tf.Variable(tf.zeros(shape=(self.config.conv_layers[0][3])))
        b2 = tf.Variable(tf.zeros(shape=(self.config.conv_layers[1][3])))
        b3 = tf.Variable(tf.zeros(shape=(self.config.hidden_size)))
        b4 = tf.Variable(tf.zeros(shape=(self.config.n_classes)))

        l1 = tf.nn.max_pool(value = tf.nn.relu(tf.add(tf.nn.conv2d(x_image,C1,padding = 'SAME',strides = [1,1,1,1]),b1)),ksize = [1,self.config.pool_sizes[0],self.config.pool_sizes[1],1],strides = [1,self.config.pool_sizes[0],self.config.pool_sizes[1],1],padding = 'SAME')
        l2 = tf.nn.max_pool(value = tf.nn.relu(tf.add(tf.nn.conv2d(l1,C2,padding = 'SAME', strides = [1,1,1,1]),b2)),ksize = [1,self.config.pool_sizes[0],self.config.pool_sizes[1],1],strides = [1,self.config.pool_sizes[0],self.config.pool_sizes[1],1],padding = 'SAME')
        f=tf.reshape(l2,[-1,self.config.conv_output_size])
        h = tf.nn.relu(tf.add(tf.matmul(f, self.W),b3))
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        pred = tf.add(tf.matmul(h_drop, self.U),b4)
       
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        I compute 
        loss = sum(softmax_loss) + self.weight_decay_placeholder * ((sum_{i,j} W_{i,j}^2)+(sum_{i,j} U_{i,j}^2))
        Where W are the weights for the hidden layer and into softmax

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        reg = self.weight_decay_placeholder*tf.add(tf.reduce_mean(tf.square(self.W)),tf.reduce_mean(tf.square(self.U))) #check!!!!
        ### END CODE
        return loss + reg

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See


        Using tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)

        return train_op

    def predict_labels_on_batch(self, session, inputs_batch):
        """ Make label predictions for the provided batch of data - helper function
                        
        Args:
               session: tf.Session()
               input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
               predicted_labels: np.ndarray of shape (n_samples,)        
        """
        predicted_labels = None
        logits = self.predict_on_batch(session, inputs_batch)
        predicted_labels_tensor = tf.argmax(logits, 1)
        predicted_labels =  session.run(predicted_labels_tensor)
        return predicted_labels

                       
if __name__=='__main__':
    print('DOOH')
