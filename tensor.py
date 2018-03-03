import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# #
# # import tensorflow as tf
# #
# # softmax_data = [0.7, 0.2, 0.1]
# # one_hot_data = [1.0, 0.0, 0.0]
# #
# # softmax = tf.placeholder(tf.float32)
# # one_hot = tf.placeholder(tf.float32)
# #
# # # TODO: Print cross entropy from session
# # cross_entropy = -tf.reduce_sum(tf.multiply(tf.log(softmax),one_hot))
# #
# # with tf.Session() as sess:
# #     print(sess.run( cross_entropy, feed_dict = {softmax: softmax_data, one_hot: one_hot_data}))
#
#
# # import numpy as np
# #
# # def sigmoid(x):
# #     # TODO: Implement sigmoid function
# #     return (1/(1+ np.exp(-x)))
# #
# #
# #
# # inputs = np.array([0.7, -0.3])
# # weights = np.array([0.1, 0.8])
# # bias = -0.1
# #
# # # TODO: Calculate the output
# # x = 0
# # for i in range (2):
# #     x += inputs[i]*weights[i]
# # output = sigmoid(x + bias)
# #
# # print('Output:')
# # print(output)
#
#
# # import numpy as np
# #
# # def sigmoid(x):
# #     """
# #     Calculate sigmoid
# #     """
# #     return 1/(1+np.exp(-x))
# #
# # def sigmoid_prime(x):
# #     """
# #     # Derivative of the sigmoid function
# #     """
# #     return sigmoid(x) * (1 - sigmoid(x))
# #
# # learnrate = 0.5
# # x = np.array([1, 2, 3, 4])
# # y = np.array(0.5)
# #
# # # Initial weights
# # w = np.array([0.5, -0.5, 0.3, 0.1])
# #
# # ### Calculate one gradient descent step for each weight
# # ### Note: Some steps have been consilated, so there are
# # ###       fewer variable names than in the above sample code
# #
# # # TODO: Calculate the node's linear combination of inputs and weights
# # h = np.dot(x,w)
# #
# #
# # # TODO: Calculate output of neural network y -hat
# # nn_output = sigmoid(h)
# #
# # # TODO: Calculate error of neural network
# # error = y- nn_output
# #
# # # TODO: Calculate the error term
# # #       Remember, this requires the output gradient, which we haven't
# # #       specifically added a variable for.
# # error_term = error* sigmoid_prime(h)
# #
# # # TODO: Calculate change in weights
# # del_w = learnrate * error_term * x
# #
# # print('Neural Network output:')
# # print(nn_output)
# # print('Amount of Error:')
# # print(error)
# # print('Change in Weights:')
# # print(del_w)
#
#
#
#
# # import numpy as np
# # from data_prep import features, targets, features_test, targets_test
# #
# #
# # def sigmoid(x):
# #     """
# #     Calculate sigmoid
# #     """
# #     return 1 / (1 + np.exp(-x))
# #
# # # TODO: We haven't provided the sigmoid_prime function like we did in
# # #       the previous lesson to encourage you to come up with a more
# # #       efficient solution. If you need a hint, check out the comments
# # #       in solution.py from the previous lecture.
# #
# # # Use to same seed to make debugging easier
# # np.random.seed(42)
# #
# # n_records, n_features = features.shape
# # last_loss = None
# #
# # # Initialize weights
# # weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
# #
# # # Neural Network hyperparameters
# # epochs = 1000
# # learnrate = 0.5
# #
# # for e in range(epochs):
# #     del_w = np.zeros(weights.shape)
# #     for x, y in zip(features.values, targets):
# #         # Loop through all records, x is the input, y is the target
# #
# #         # Note: We haven't included the h variable from the previous
# #         #       lesson. You can add it if you want, or you can calculate
# #         #       the h together with the output
# #
# #         # TODO: Calculate the output
# #         output = sigmoid(np.dot(x,weights))
# #
# #         # TODO: Calculate the error
# #         error = y-output
# #
# #         # TODO: Calculate the error term
# #         error_term = error* output*(1-output)
# #
# #         # TODO: Calculate the change in weights for this sample
# #         #       and add it to the total weight change
# #         del_w += error_term * x * learnrate
# #
# #     # TODO: Update weights using the learning rate and the average change in weights
# #     weights += del_w / n_records
# #
# #     # Printing out the mean square error on the training set
# #     if e % (epochs / 10) == 0:
# #         out = sigmoid(np.dot(features, weights))
# #         loss = np.mean((out - targets) ** 2)
# #         if last_loss and last_loss < loss:
# #             print("Train loss: ", loss, "  WARNING - Loss Increasing")
# #         else:
# #             print("Train loss: ", loss)
# #         last_loss = loss
# #
# #
# # # Calculate accuracy on test data
# # tes_out = sigmoid(np.dot(features_test, weights))
# # predictions = tes_out > 0.5
# # accuracy = np.mean(predictions == targets_test)
# # print("Prediction accuracy: {:.3f}".format(accuracy))
#
#
#
#
#
# # import numpy as np
# #
# # def sigmoid(x):
# #     """
# #     Calculate sigmoid
# #     """
# #     return 1/(1+np.exp(-x))
# #
# # # Network size
# # N_input = 4
# # N_hidden = 3
# # N_output = 2
# #
# # np.random.seed(42)
# # # Make some fake data
# # X = np.random.randn(4)
# #
# # weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
# # weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
# #
# #
# # # TODO: Make a forward pass through the network
# # hidden_layer_in = np.dot(X, weights_input_to_hidden)
# # hidden_layer_out = sigmoid(hidden_layer_in)
# #
# # print('Hidden-layer Output:')
# # print(hidden_layer_out)
# #
# # output_layer_in = np.dot(hidden_layer_out,weights_hidden_to_output)
# # output_layer_out = sigmoid(output_layer_in)
# #
# # print('Output-layer Output:')
# # print(output_layer_out)
#
#
#
#
# import numpy as np
# from data_prep import features, targets, features_test, targets_test
#
# np.random.seed(21)
#
# def sigmoid(x):
#     """
#     Calculate sigmoid
#     """
#     return 1 / (1 + np.exp(-x))
#
#
# # Hyperparameters
# n_hidden = 2  # number of hidden units
# epochs = 900
# learnrate = 0.005
#
# n_records, n_features = features.shape
# last_loss = None
# # Initialize weights
# weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))
# weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden)
#
# for e in range(epochs):
#     del_w_input_hidden = np.zeros(weights_input_hidden.shape)
#     del_w_hidden_output = np.zeros(weights_hidden_output.shape)
#     for x, y in zip(features.values, targets):
#         ## Forward pass ##
#         # TODO: Calculate the output
#         hidden_input = np.dot(x, weights_input_hidden)
#
#         hidden_output = sigmoid(hidden_input)
#         output = sigmoid(np.dot(hidden_output,weights_hidden_output))
#
#         ## Backward pass ##
#         # TODO: Calculate the network's prediction error
#         error = y-output
#
#         # TODO: Calculate error term for the output unit
#         output_error_term = error* output*(1-output)
#
#         ## propagate errors to hidden layer
#
#         # TODO: Calculate the hidden layer's contribution to the error
#         hidden_error = np.dot(output_error_term, weights_hidden_output)
#
#         # TODO: Calculate the error term for the hidden layer
#         hidden_error_term = hidden_error* hidden_output * (1-hidden_output)
#
#         # TODO: Update the change in weights
#         del_w_hidden_output += output_error_term*hidden_output
#         del_w_input_hidden +=  hidden_error_term* x[:,None]
#
#     # TODO: Update weights
#     weights_input_hidden += learnrate*del_w_input_hidden
#     weights_hidden_output += learnrate*del_w_hidden_output
#
#     # Printing out the mean square error on the training set
#     if e % (epochs / 10) == 0:
#         hidden_output = sigmoid(np.dot(x, weights_input_hidden))
#         out = sigmoid(np.dot(hidden_output,
#                              weights_hidden_output))
#         loss = np.mean((out - targets) ** 2)
#
#         if last_loss and last_loss < loss:
#             print("Train loss: ", loss, "  WARNING - Loss Increasing")
#         else:
#             print("Train loss: ", loss)
#         last_loss = loss
#
# # Calculate accuracy on test data
# hidden = sigmoid(np.dot(features_test, weights_input_hidden))
# out = sigmoid(np.dot(hidden, weights_hidden_output))
# predictions = out > 0.5
# accuracy = np.mean(predictions == targets_test)
# print("Prediction accuracy: {:.3f}".format(accuracy))
