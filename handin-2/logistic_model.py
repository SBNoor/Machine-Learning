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
    batch_size = 16
    n_epochs = 10
    lr = 0.001
    weight_decay = 1e-4    
    
    def __init__(self, name_suffix=None, **kwargs):        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # overwrite values and set paths
        self.name = 'logistic'
        if name_suffix is not None:
            self.name = '{0}_{1}'.format(self.name, name_suffix)
        weights_path = os.path.join(os.getcwd(), 'model_weights')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        self.weights_path = weights_path
        self.weights_file = os.path.join(self.weights_path, "{0}.weight".format(self.name))

class LogisticRegressionModel(TfModel):
    
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors (the input data)

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None, ), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.weight_decay_placeholder


        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ))
        self.weight_decay_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs_batch, labels_batch=None, weight_decay=0):
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
        Returns:
            feed_dict: dict, mapping from placeholders to values.
        """
        
        feed_dict = {self.input_placeholder : inputs_batch, self.weight_decay_placeholder: weight_decay}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        """Adds the logistic model computation 
            ls = xW + bias - linear signal on input batch
            pred = sigmoid(ls) - class 1 probabilities for the batch

        We must initialize the variables W, bias and we initialize them to zero
        Store W in se

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features, 1)
                    b:  scalar (1,)

        Add these placeholders to self as the instance variables
            self.W - for weight decay later

        Returns:
            pred: tf.Tensor of shape (batch_size, ) (reshape to (-1,)
        """
        x = self.input_placeholder
        self.W = tf.Variable(tf.zeros([self.config.n_features, 1]))
        bias = tf.Variable(tf.zeros([]))
        pred = tf.matmul(x, self.W) + bias
        pred = tf.reshape(pred, [-1])
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch and weight decay added

        You should use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.

        loss = sum(softmax_loss) + self.weight_decay_placeholder * (sum_i W_i ^2)

        Args:
            pred: A tensor of shape (batch_size, 1) containing the logits for logistic regression

        Returns:
            loss: A 0-d tensor (scalar)
        """
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=tf.cast(self.labels_placeholder, pred.dtype))
        loss = tf.reduce_mean(cross_entropy)
        reg = self.weight_decay_placeholder * tf.reduce_sum(self.W**2)
        return loss + reg

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. 

        See https://www.tensorflow.org/api_guides/python/train#Optimizer
        for more information or the tensorflow guide for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def predict_labels_on_batch(self, session, inputs_batch):
        """ Make label predictions for the provided batch of data - helper function
                        
        Args:
               session: tf.Session()
               input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
               predicted_labels: np.ndarray of shape (n_samples,)        
        """
        logits = self.predict_on_batch(session, inputs_batch)
        predicted_labels_tensor = tf.cast(logits > 0, tf.int64)
        predicted_labels =  session.run(predicted_labels_tensor)
        return predicted_labels

            
    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     weight_decay=self.config.weight_decay)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
