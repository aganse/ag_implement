"""
Three implementations of neural net MLPs for regression, in SKL-style template:
Keras_MLP (a wrapping of Keras' Sequential class forming an MLP)
TF_MLP (adapted from some Ng class code implementing TensorFlow directly)
SKL_MLP (just a wrapping of SKL's MLPRegressor)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.base import BaseEstimator #, ClassifierMixin, TransformerMixin
#from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#from sklearn.utils.multiclass import unique_labels
#from sklearn.metrics import euclidean_distances
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras import regularizers  # l2, l1, l1_l2
from keras.callbacks import History


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=(n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name = "Y")

    return X, Y


def initialize_parameters(layerunits):
    """
    Initializes parameters to build a neural network with tensorflow.

    Returns:
    parameters -- a list of tensors containing W1, b1, W2, b2, W3, b3
    """

    params = []
    for l in range(len(layerunits)-1):
        params.append(tf.get_variable('W'+str(l+1), [layerunits[l+1],layerunits[l]], initializer = tf.contrib.layers.xavier_initializer()))
        params.append(tf.get_variable('b'+str(l+1), [layerunits[l+1],1], initializer = tf.zeros_initializer()))
    #print(params)
    return params


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z4 -- the output of the last LINEAR unit
    """

    nl = int(len(parameters)/2)  # num layers
    out = X
    for l in range(nl):
        out = tf.add( tf.matmul(parameters[2*l],out), parameters[2*l+1])
        if l != nl-1:  # ie for all but last layer
            out = tf.nn.relu(out)
        # elif l == len(parameters)/2:   # for classification
        #     out = tf.nn.sigmoid(out)

    return out


def compute_cost(ZL, Y, numdata):
    """
    Computes the cost

    Arguments:
    ZL -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # if doing classification:
    # (to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...))
    # logits = tf.transpose(Z3)
    # labels = tf.transpose(Y)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    # if doing regression:
    #cost = tf.metrics.root_mean_squared_error(ZL, Y)  # I think this fails because it's a metric not reduce_
    cost = tf.reduce_sum(tf.pow(ZL-Y, 2))/numdata
    return cost


def predict_fn(Xinput, parameters):
    Xinput = Xinput.T
    n_x = Xinput.shape[0]
    Xinput = np.float32(Xinput)
    X = tf.placeholder(tf.float32, shape=(n_x, None), name = "X")
    pred = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(pred, feed_dict={X: Xinput}).T


def modelRegress(X_train, Y_train, lr=0.001, hlay=[20,20,20,20], alpha=0.01,
          epochs=1500, display_step=50, minibatch_size=32, showplot=True, print_cost=True):
    """
    Implements a tensorflow neural network

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    lr -- learning rate of the optimization
    epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    X_train = X_train.T
    Y_train = Y_train.T
    (n_x, m) = X_train.shape       # (n_x: input size, m: number of examples in the train set)
    (n_y, n) = Y_train.shape       # (n_y: output size, n: number of examples in the train set, should = m)
    if m!=n:
        print('Error: X and y do not have same number of samples (columns)!')
    numdata = n_y * n
    costs = []                     # To keep track of the cost
    X_train = np.float32(X_train)
    Y_train = np.float32(Y_train)
    ops.reset_default_graph()      # to be able to rerun the model without overwriting tf variables

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters([n_x]+hlay+[n_y])

    # Forward propagation: Build the forward propagation in the tensorflow graph
    pred = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(pred, Y, numdata)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        sess.run(init)

        # this doesn't appear to have an effect, but maybe because tvars includes bias vars?:
        regularizer = tf.contrib.layers.l2_regularizer(alpha)
        tvars   = tf.trainable_variables()
        # Note this suggestion from a forum:
        #    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.001
        # They added that lossL2 directly into cost fn, but could just use the 'bias'not part here...
        # loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_train_labels)) + lossL2
        # https://stackoverflow.com/questions/38286717/tensorflow-regularization-with-l2-loss-how-to-apply-to-all-weights-not-just
        tf.contrib.layers.apply_regularization(regularizer, weights_list=tvars)

        # Fit training data - all in one batch (ie no minibatches)
        for epoch in range(epochs):
            _ , c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            costs.append(c)

            i = 0
            if epoch % display_step == 0:
            #    print("Epoch:", '%04d' % (epoch+1), "  cost=", "{:.9f}".format(c)) #, "  lr=", "{:.4f}".format(lr))
                print('.', end='', flush=True)
                i += 1
                if i % 100 == 0:
                    print('\n')
        print('\n')
        #final_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
        #print("Last epoch:", '%04d' % (epoch+1), "  final cost=", "{:.9f}".format(final_cost))
        #print("Last epoch:", '%04d' % (epoch), "  final cost=", "{:.9f}".format(costs[-1]))

        if showplot:
            # plot the cost
            costs = np.squeeze(costs)
            plt.semilogy(costs,'.-')
            plt.grid()
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(lr))
            plt.show()

        # save the parameters in a variable
        parameters = sess.run(parameters)

    return parameters,costs



class TF_MLP(BaseEstimator):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, lr=0.0025, epochs=10000, display_step=50,
                hlay=[20,20,20,20], alpha=0.01, showplot=True):
        self.lr = lr
        self.epochs = epochs
        self.display_step = display_step
        self.hlay = hlay
        self.alpha = alpha
        self.showplot = showplot

    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        #X, y = check_X_y(X, y)

        self.parameters,costs = modelRegress(X, y, lr=self.lr,
                epochs=self.epochs, display_step=self.display_step,
                hlay=self.hlay, alpha=self.alpha, showplot=self.showplot)

        # Return the estimator
        return costs

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        #X = check_array(X)
        return predict_fn(X, self.parameters)

    def get_params(self):
        return (self.lr,self.epochs,self.hlay,self.alpha)



class Keras_MLP(Sequential):
    """ A template estimator to be used as a reference implementation .
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, lr=0.0025, epochs=100, hlay=[20,20,20], alpha=0.01, showplot=True):

        self.lr = lr
        self.epochs = epochs
        self.hlay = hlay
        self.alpha = alpha
        self.showplot = showplot
        super(Keras_MLP, self).__init__()

    def fit(self, X, y):
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        #X, y = check_X_y(X, y)

        (m, n_x) = X.shape       # (n_x: input size, m: number of examples in the train set)
        (n, n_y) = y.shape       # (n_y: output size, n: number of examples in the train set, should = m)
        if m!=n:
            print('Error: X and y do not have same number of samples (columns)!')
        #numdata = n_y * n

        # input and hidden layers:
        for l in range(len(self.hlay)):
            if l==0:
                super(Keras_MLP, self).add(Dense(units=self.hlay[0], input_dim=n_x,
                    kernel_regularizer=regularizers.l2(self.alpha)))
            else:
                super(Keras_MLP, self).add(Dense(units=self.hlay[l],
                    kernel_regularizer=regularizers.l2(self.alpha)))
            super(Keras_MLP, self).add(Activation('relu'))
        # output layer:
        super(Keras_MLP, self).add(Dense(units=n_y))
        super(Keras_MLP, self).add(Activation('linear'))

        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        super(Keras_MLP, self).compile(loss='mean_squared_error', optimizer=adam)
            # , metrics=['mean_absolute_error','mean_absolute_percentage_error'])

        # classifier settings:
        # (see metrics at https://github.com/GeekLiB/keras/blob/master/keras/metrics.py)
        #model.compile(loss=keras.losses.categorical_crossentropy,
        #              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
        #self.compile(optimizer='rmsprop', loss='mse')
        #  # metrics=['binary_accuracy','precision','recall','f1score']

        history = History()
        super(Keras_MLP, self).fit(X, y, epochs=self.epochs, verbose=0, callbacks=[history])  #, batch_size=128)
        losshist = history.history['loss']
        #print(super(Keras_MLP, self).summary())

        if self.showplot:
            # plot the cost
            plt.semilogy(losshist,'.-')
            plt.grid()
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title('Loss history')
            plt.show()

    #     # loss_and_metrics = model.evaluate(X_test, y_test)  #, batch_size=128)
    #     # print(loss_and_metrics)

        # Return the estimator
        return losshist

    # def predict(self, X):
    #     """ A reference implementation of a predicting function.
    #     Parameters
    #     ----------
    #     X : array-like of shape = [n_samples, n_features]
    #         The input samples.
    #     Returns
    #     -------
    #     y : array of shape = [n_samples]
    #         Returns :math:`x^2` where :math:`x` is the first column of `X`.
    #     """
    #     #X = check_array(X)
    #     return self.predict(x_test)  #, batch_size=128)

    def get_params(self):
        return (self.lr,self.epochs,self.hlay,self.alpha)



# class SKL_MLP(MLPRegressor):
#     """ A template estimator to be used as a reference implementation .
#     Parameters
#     ----------
#     demo_param : str, optional
#         A parameter used for demonstation of how to pass and store paramters.
#     """
#     def __init__(self, lr=0.0025, epochs=100, hlay=[20,20,20], alpha=0.01, showplot=True):
#
#         self.lr = lr
#         self.epochs = epochs
#         self.hlay = hlay
#         self.alpha = alpha
#         self.showplot = showplot
#         super(SKL_MLP, self).__init__()
#
#     def fit(self, X, y):
#         """A reference implementation of a fitting function
#         Parameters
#         ----------
#         X : array-like or sparse matrix of shape = [n_samples, n_features]
#             The training input samples.
#         y : array-like, shape = [n_samples] or [n_samples, n_outputs]
#             The target values (class labels in classification, real numbers in
#             regression).
#         Returns
#         -------
#         self : object
#             Returns self.
#         """
#         #X, y = check_X_y(X, y)
#
#         (m, n_x) = X.shape       # (n_x: input size, m: number of examples in the train set)
#         (n, n_y) = y.shape       # (n_y: output size, n: number of examples in the train set, should = m)
#         if m!=n:
#             print('Error: X and y do not have same number of samples (columns)!')
#         #numdata = n_y * n
#
#         #alphas = np.logspace(-4,4,17)
#         alphas = np.linspace(0,1000,11)
#         names = []
#         for i in alphas:
#             names.append('alpha ' + str(i))
#
#         classifiers = []
#         for i in alphas:
#             classifiers.append(MLPRegressor(alpha=i, hidden_layer_sizes=(21,21,21), max_iter=250))
#
#         # from sklearn.model_selection import KFold
#         # kf = KFold(n_splits=2)
#         # for itrain, itest in kf.split(X):
#         # scores = []
#         # for name, clf in zip(names, classifiers):
#         #     clf.fit(X_train, y_train)
#         #     score = clf.score(X_test, y_test)
#         #     print(name,': ',score)
#         #     scores.append(score)
#
#         clf = MLPRegressor(hidden_layer_sizes=(21,21,21,21,21,21), max_iter=50000)
#         clf.fit(X_train, y_train)
#         print(clf.score(X_test, y_test))
#         y_trainpred = clf.predict(X_train)
#         y_testpred = clf.predict(X_test)
