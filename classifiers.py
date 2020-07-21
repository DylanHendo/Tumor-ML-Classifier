'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, neighbors ,svm, neural_network

import time

def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(0, 'Dylan', 'Henderson'), (1, 'NA', 'NA'), (2, 'NA', 'NA')]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    # Load data using numpy. Split on commas.
    data = np.genfromtxt(dataset_path, delimiter=',',dtype=None, encoding=None)

    # If tumour is malignant, set to 1, otherwise set to 0.
    y = [1 if row[1] == 'M' else 0 for row in data]

    # Create a new list (so it becomes 2 dimensional) containing the real-valued data fields.
    x = [list(row)[2:] for row in data]

    return np.array(x), np.array(y)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Max we will allow is 100, so it is fast and we avoid overfitting.
    depth = 100
    max_depth = range(1, depth + 1)

    dt_classifier = tree.DecisionTreeClassifier()
    # Use 'max_depth' as hyperparameter to test.
    parameters = [
        {
            'max_depth': max_depth
        }
    ]
    # Find the best value for max_depth using a cross validated gridsearch
    dd_clf = GridSearchCV(dt_classifier, parameters)

    dd_clf.fit(X_training, y_training)
    return dd_clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    max_neighbours = 20
    n_neighbors = range(1, max_neighbours + 1)

    knn = neighbors.KNeighborsClassifier()

    parameters = [
        {
            'n_neighbors': n_neighbors
        }
    ]

    clf = GridSearchCV(knn, parameters)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    svm_clf = svm.SVC(gamma=0.0001)

    # only use parameter C
    # C : float, optional (default=1.0) Penalty parameter C of the error term.

    C_range = np.logspace(-2, 2, 5)

    params = [
        {
            "C": C_range
        }
    ]

    clf = GridSearchCV(svm_clf, params)
    clf.fit(X_training, y_training)
    return clf



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Max we will allow is 40, so it is fast and we avoid overfitting.
    # Operate with layers from 20 to 40, this was found to be more accurate when experimenting.
    max = 40
    hidden_layers = [(i,) for i in range(20,max + 1)]

    # Set the random_state, max_iter and tol to ensure that the model converges and it replicable
    nn_classifier = neural_network.MLPClassifier(random_state=1, max_iter= 1000, tol= 0.001)
    # Use 'hidden_layer_sizes' as hyperparameter to test.
    parameters = [
        {
            'hidden_layer_sizes': hidden_layers,
        }
    ]
    # Find the best value for max_depth using a cross validated gridsearch
    # run block of code and catch warnings
    nn_clf = GridSearchCV(nn_classifier, parameters)

    nn_clf.fit(X_training, y_training)
    return nn_clf



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here
    print(my_team())

    X, y = prepare_dataset('medical_records.data')

    # We will split the data into 25% for testing and 75% for training.
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create a list of classifiers so we can iterate over them.
    # The code to run them is the same so we can turn it into a for loop.
    # You can just comment out the classfiers you don't want to test.
    classifiers = [
                   [build_DecisionTree_classifier, "Decision Tree"],
                   [build_NearrestNeighbours_classifier, "Nearest Neighbour"],
                   [build_SupportVectorMachine_classifier, "Support Vector Machine"],
                   [build_NeuralNetwork_classifier, "Neural Network"]
                   ]

    # Test each classifier and output values
    for classifier_func, classifier_name in classifiers:

        # t0 = time.time()

        # Call the function with the training data.
        classifier = classifier_func(X_train, y_train)

        # Print out the best value for the hyperparameter.
        print(classifier_name + " Best Parameters:", classifier.best_params_)

        # Generate classification report for training data
        training_predictions = classifier.predict(X_train)

        print(classifier_name + " Training Data Classification Report:")
        print(classification_report(y_train, training_predictions))

        # Do the same thing for the test data
        test_predictions = classifier.predict(X_test)
        
        print(classifier_name + " Test Data Classification Report:")
        print(classification_report(y_test, test_predictions))


        # t1 = time.time()
        # print("{}: {}".format(classifier_name, (t1-t0)))
