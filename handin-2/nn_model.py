import os
import tensorflow as tf
from model import Model as TfModel

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. 
    """
    n_features = 28*28
    n_classes = 10
    dropout = 0.5
    hidden_size = 128
    weight_decay = 1e-4
    batch_size = 32
    n_epochs = 5
    lr = 0.001
    
    def __init__(self, name_suffix=None, **kwargs):        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # overwrite values and set paths
        self.name = 'nn_h{0}'.format(self.hidden_size)
        if name_suffix is not None:
            self.name = '{0}_{1}'.format(self.name, name_suffix)
        weights_path = os.path.join(os.getcwd(), 'model_weights')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        self.weights_path = weights_path
        self.weights_file = os.path.join(self.weights_path, "{0}.weight".format(self.name))
        
class FeedForwardModel(TfModel):
    
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors (the input data)

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None, ), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32
        weight_decay_placeholder: Weight Decay Value (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder
            self.weight_decay_placeholder

        """
        
        n_features = self.config.n_features

        self.input_placeholder = tf.placeholder(tf.float32,shape = (None,n_features))
        self.labels_placeholder = tf.placeholder(tf.int32,shape = (None))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.weight_decay_placeholder = tf.placeholder(tf.float32)
        #self.dropout_placeholder = tf.placeholder(type = tf.float32,dropout)
        #self.weight_decay_placeholder = tf.placeholder(type = tf.float32,weight_decay)

    def create_feed_dict(self, inputs_batch, labels_batch=None, weight_decay = 0, dropout=1):
        """Creates the feed_dict for the neural net

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }
        The keys for the feed_dict should be a subset of the placeholder 
        tensors created in add_placeholders. When an argument is None, don't add it to the feed_dict.
        (We only add labels when we train)

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
            weight_decay: the lambda weight decay scale parameter
        Returns:
            feed_dict: dict, mapping from placeholders to values.
        """
        
        feed_dict = {}

        
        if inputs_batch is not None:
            feed_dict[self.input_placeholder] = inputs_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if weight_decay is not None:
            feed_dict[self.weight_decay_placeholder] = weight_decay
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout

        return feed_dict

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1) - hidden layer
            h_drop = Dropout(h, dropout_rate) - use dropout
            pred = h_dropU + b2 - output layer

        Note that we are not applying a softmax transform to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        build-in tensorflow function tf.nn.softmax_cross_entropy_with_logits

        Before using the Variables, W, b1, U, b2 they must be defined and initialized. 

        Use tf.contrib.xavier_initializer to initialize W and U 
        We have already initialized W for you so you can see how it is done.
        You can initialize b1 and b2 with zeros (tf.zeros)

        Here are the dimensions of the various variables you will need to create
                    W:  (n_features, hidden_size)
                    b1: (hidden_size) - shape hidden_size, (like the numpy vectors that were not matrices)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes) - shape hidden_size, (like the numpy vectors that were not matrices)
        
        Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument. 
        The keep probability should be set to the value of self.dropout_placeholder

        Add these placeholders to self as the instance variables (need them for weigth decay)
            self.W
            self.U

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
            
        """
        
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        Wshape = (self.config.n_features, self.config.hidden_size)
        self.W = tf.Variable(xavier_initializer(Wshape))
        
        Ushape = (self.config.hidden_size,self.config.n_classes)
        self.U = tf.Variable(xavier_initializer(Ushape))
        
        #b1 = tf.Variable(tf.zeros(shape = self.config.hidden_size))
        b1 = tf.Variable(tf.zeros(shape = self.config.hidden_size))
        
        
        
        b2 = tf.Variable(tf.zeros(shape = self.config.n_classes))
        
        x = self.input_placeholder
        
        print("x : ",x.shape)
        print("self.w : ",self.W.shape)
        print("self.U : ",self.U.shape)
        
        
        #input_layer = tf.add(tf.matmul(x,self.W),b1)
        h = tf.nn.relu(tf.matmul(x,self.W) + b1)
        h_drop = tf.nn.dropout(h, self.dropout_placeholder)
        print("h_drop : ",h_drop.shape)
        pred = tf.add(tf.matmul(h_drop,self.U),b2)
        print("pred : ",pred.shape)
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss + weight_decay
        The loss should be averaged over all examples in the current minibatch.
        
        You should use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.

        loss = sum(softmax_loss) + self.weight_decay_placeholder * (sum_{i,j} W_{i,j}^2 + \sum_{i,j} U_{i,j}^2)
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.

        Returns:
            loss: A 0-d tensor (scalar)
        """
        
     
        #pred = tf.transpose(pred)
        #print("pred : ",pred.shape)
       # for i in range(self.config.n_classes):
        #temp_pred = tf.squeeze(pred)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred,labels = self.labels_placeholder))
        #    loss += reduce_loss
        
        reg = self.config.weight_decay
        print("reg : ",reg)

        
        return loss + reg

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. 


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
    
    
            
