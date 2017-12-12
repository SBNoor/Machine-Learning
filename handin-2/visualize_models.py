import nn_model as nn
import cnn_model as cnn
from model import Classifier
import matplotlib.pyplot as plt
from h2_util import load_train_data
import numpy as np

def visualize_nn(config = {'hidden_size': 256}):
    """ Visualize the hidden layer of the nn_model

    To use this you must add the weigth matrix W to the nn_model object (self)  so we can access it.
    Should already be done.
    """    
    model = Classifier(nn.FeedForwardModel, nn.Config(**config))
    W = model.get_nn_weights()# 784 x 256    
    # norms = np.sum(W**2, axis=0)
    # norms[norms < 1e-1] = 1
    # Wnormed = W / norms
    Wnormed = W
    # normalize
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])     
    ims = Wnormed.reshape(28, 28, 16, 16)
    rs = ims.transpose(2, 0, 3, 1).reshape(16*28, 16*28)
    ax.matshow(rs, cmap='bone', vmin=.5 * Wnormed.min(), vmax=.5 * Wnormed.max())
    ax.set_title('Neural Net Hidden Layer Visualization')
    

def visualize_cnn(config = {'hidden_size': 1024}):
    """ Visualize the convolutions features found in cnn model somehow.

    To use this update add_prediction op and store references to the variable weights for the convolution and tensors representing the computation
    save the first convolution variable in self.C1 
    save the output of the first convolution in self.conv1
    save the output of the first convolution after relu in self.conv1_relu
    
    """
    config = {'hidden_size': 1024}
    model = Classifier(cnn.ConvolutionalModel, cnn.Config(**config))
    W = model.get_cnn_weights()
    convolutions = W.transpose(3, 0, 1, 2).squeeze()
    # convolutions = W.transpose(3, 0, 1, 2).reshape(32,25).T
    # fig2 = plt.figure()
    # ax = fig2.add_axes([0.05, 0.05, 0.9, 0.9])
    # ax.matshow(convolutions.reshape(5, 5, 4, 8).transpose(2,0,3,1).reshape(4*5, 8*5), cmap='gray')
    fig, axes = plt.subplots(8, 4)
    vmin, vmax = W.min(), W.max()
    for coef, ax in zip(convolutions, axes.ravel()):
        ax.matshow(coef, cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    fig.savefig('results/convolution_filters.png')

    img, lab = load_train_data()    
    conv, relu = model.get_cnn_conv(img[0:64])
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.05, 0.05, 0.9, 0.9])
    c0 = conv[0]
    tmp = c0.transpose(2,0,1).reshape(32, 784).T.reshape(28, 28, 4, 8)
    one_plot = tmp.transpose(2, 0, 3, 1).reshape(4*28, 8*28)
    ax2.matshow(one_plot, cmap='gray')
    ax2.set_title('convolution output of first point')
    fig2.savefig('results/convolution_of_first_image.png')
    fig3 = plt.figure()
    ax3 = fig3.add_axes([0.05, 0.05, 0.9, 0.9])    
    c0 = relu[0]
    tmp = c0.transpose(2,0,1).reshape(32, 784).T.reshape(28, 28, 4, 8)
    one_plot = tmp.transpose(2, 0, 3, 1).reshape(4*28, 8*28)
    ax3.matshow(one_plot, cmap='gray')
    ax3.set_title('convolution relu output of first point')
    fig3.savefig('results/convolution_of_first_image_after_relu.png')

    
if __name__=='__main__':
    visualize_nn()
    visualize_cnn()
    plt.show()
