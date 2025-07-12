# Name: Joshua Lindsey
# Date: 07/11/2025
# Project Name: Recognizing Handwritten Digits in Scikit Learn
# Source: Stock Market Price Prediction, Fraud Detection,

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_data():
    digits = datasets.load_digits()
    print()
    #print(dir(digits))
    return digits

def var_setup(digits):
    x = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    n_len = len(x)//2

    x_train = x[:n_len]
    y_train = y[:n_len]

    x_test = x[n_len:]
    y_test = y[n_len:]

    return x_train, y_train, x_test, y_test

def pipeline(x_train, y_train, x_test, y_test, alpha,  learning_rate):
    activation = ['relu', 'identity', 'logistic']
    solver = [ 'sgd', 'adam']

    metrics = {}
    for i in activation:
        for j in solver:
            #print(i,j)
            accuracy = model(x_train, y_train, x_test, y_test, i, alpha, j, learning_rate)
            #print(i,j, accuracy)
            metrics["{}-{}".format(i,j)] = accuracy

    highest_accuracy = max(metrics, key=metrics.get)
    print("Highest Accuracy is: ", highest_accuracy, metrics[highest_accuracy])

    return None

def model(x_train, y_train, x_test, y_test, activation, alpha, solver, tol, learning_rate):
    mlp = MLPClassifier(hidden_layer_sizes=(15,), 
                    activation=activation, # (parameter) activation: Literal['relu', 'identity', 'logistic', 'tanh']
                    alpha=alpha,
                    solver=solver, # (parameter) solver: Literal['lbfgs', 'sgd', 'adam']
                    tol=tol,
                    random_state=1,
                    learning_rate_init=learning_rate,
                    verbose=False # Whether to print progress messages to stdout
                    )
    
    mlp.fit(x_train, y_train)
    
    #fig, axes = plt.subplots(1,1)
    #axes.plot(mlp.loss_curve_, '-o')
    #axes.set_title("{}-{}".format(activation, solver))
    #axes.set_xlabel("Number of Iterations")
    #axes.set_ylabel("Loss")
    #plt.show()

    predictions = mlp.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def alpha_tol_lr_pipeline(x_train, y_train, x_test, y_test, activation, solver):
    alpha_lst = [0.1, 0.001, 0.0001, 0.00001]
    tol_lst = [0.1, 0.001, 0.0001, 0.00001]
    learn_rate = [0.1, 0.001, 0.0001, 0.00001]

    metrics = {}
    for i in alpha_lst:
        for j in tol_lst:
            for k in  learn_rate:
                    accuracy = model(x_train, y_train, x_test, y_test, activation, i, solver, j, k)
                    print((i,j,k), accuracy)
                    metrics["{}_{}_{}".format(i,j,k)] = accuracy

    highest_accuracy = max(metrics, key=metrics.get)
    print("Highest Accuracy is: ", highest_accuracy, metrics[highest_accuracy])


    #metrics = model(x_train, y_train, x_test, y_test, activation, alpha, solver, learning_rate)
    #print(metrics)
    return metrics


if __name__ == "__main__":
    alpha = 0.0001
    learning_rate = 0.01

    digits = load_data()
    #print(type(digits))
    x_train, y_train, x_test, y_test = var_setup(digits)

    # Run to determine which combination of activation and solver performs with highest accuracy
    #pipeline(x_train, y_train, x_test, y_test, alpha,  learning_rate)

    # Set activation and solver from results of pipeline run
    activation = "identity"
    solver = "adam"
    met = alpha_tol_lr_pipeline(x_train, y_train, x_test, y_test, activation, solver)
    print(met)