import numpy as np
import tensorflow as tf

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    Each algorithm you will construct in this homework will
    inherit from this Model object.
    """
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/api_docs/python/tf/placeholder
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

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
                                     dropout=self.config.dropout, weight_decay=self.config.weight_decay)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, session, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            session: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = session.run(self.pred, feed_dict=feed)
        return predictions

    def predict_labels_on_batch(self, session, inputs_batch):
        """ Make label predictions for the provided batch of data 

        Args:
            session: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples,)        
        """
        raise NotImplementedError("Each Model must re-implement this method.")

      
    def build(self):
        """ Add place holders, prediction, loss and train_op
        
        Save pred, loss, and train_op so we can access them.
        """
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def accuracy(self, sess, X, y):
        """ Simple helper function - only works for multi-class """
        pred_labels = self.predict_labels_on_batch(sess, X)
        acc = np.mean(pred_labels==y)
        return acc
    
    def run_epoch(self, session, X, y):
        """ Randomly permute the data and run a training epoch in session on data X with labels y 
        
        Args:
        session: tf.session
        X: numpy array shape (n, d)
        y: numpy array shape (n,)
        """
        n = y.size
        rp = np.random.permutation(y.shape[0])
        rpx = X[rp, :]
        rpy = y[rp]       
        for j in range(0, n, self.config.batch_size):
            l = min(j + self.config.batch_size, n)
            xchunk = rpx[j:l]
            ychunk = rpy[j:l]
            self.train_on_batch(session, xchunk, ychunk)

    def fit(self, session, X_train, y_train, X_val, y_val, saver):
        """ Basic training algorithm 

        Args:
            session: tf.session
            X_train: numpy array shape (n, d)
            y_train: numpy array shape (n,)
            X_val: numpy array shape (n', d)
            y_val: numpy array shape (n',)
            saver: tf.saver
        """        
        best_dev_acc = 0
        train_acc = []
        val_acc = []
        for epoch in range(self.config.n_epochs):
            self.run_epoch(session, X_train, y_train)
            train_accuracy = self.accuracy(session, X_train, y_train)
            val_accuracy = self.accuracy(session, X_val, y_val)
            print("step {0}, training accuracy {1}, validation accuracy {2}".format(epoch, 100*train_accuracy, 100*val_accuracy))            
            if val_accuracy > best_dev_acc:
                # Early stopping 
                print("New best validation score saving weights in {0}".format(self.config.weights_file))
                saver.save(session, self.config.weights_file)                                    
                best_dev_acc = val_accuracy
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
        print('Training done...')
        # Save the results in a file for plotting
        return train_acc, val_acc
     
    def __init__(self, config):
        self.config = config
        self.build()
        # remove build from here maybe
        
class Classifier():
    """ Simple Classifier class to encapsulate train and predict function for Model Objects """

    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def train(self, X_train, y_train, X_val, y_val):
        ### ADD TIMER 
        with tf.Graph().as_default():
            model = self.model(self.config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(init)
                train_acc, val_acc = model.fit(session, X_train, y_train, X_val, y_val, saver)
        return train_acc, val_acc

    def predict(self, X):
        """ Use trained model for predictions """
        with tf.Graph().as_default():
            print("Building model...")
            model = self.model(self.config)
            # tf could rebuild model it self but for now we just create the same graph again.
            saver = tf.train.Saver()

            with tf.Session() as session:
                print('Restore Model:', self.config.weights_file)
                saver.restore(session, self.config.weights_file)
                print('Start predicting on the data')
                pred_labels = model.predict_labels_on_batch(session, X)
        return pred_labels

    def get_nn_weights(self):
        with tf.Graph().as_default():
            print("Building model...")
            model = self.model(self.config)
            # tf could rebuild model it self but for now we just create the same graph again.
            saver = tf.train.Saver()

            with tf.Session() as session:
                print('Restore Model')
                saver.restore(session, self.config.weights_file)
                print('Retrieve Weight on the data')
                W = session.run(model.W)
                return W

    def get_cnn_weights(self):
        with tf.Graph().as_default():
            print("Building model...")
            model = self.model(self.config)
            # tf could rebuild model it self but for now we just create the same graph again.
            saver = tf.train.Saver()

            with tf.Session() as session:
                print('Restore Model')
                saver.restore(session, self.config.weights_file)
                print('Retrieve Weight on the data')
                C1 = session.run(model.C1)
                return C1
        
    def get_cnn_conv(self, X):
        with tf.Graph().as_default():
            print("Building model...")
            model = self.model(self.config)
            # tf could rebuild model it self but for now we just create the same graph again.
            saver = tf.train.Saver()

            with tf.Session() as session:
                print('Restore Model')
                saver.restore(session, self.config.weights_file)
                print('Retrieve Weight on the data')
                conv1 = session.run(model.conv1, model.create_feed_dict(X))
                conv1_relu = session.run(model.conv1_relu, model.create_feed_dict(X))                                
                return conv1, conv1_relu
        
