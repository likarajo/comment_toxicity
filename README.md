# Comment Toxicity
A Multi-label classification model to predict the probability of different types of toxicity for comments.

## Goal
To develop a text classification model that analyzes a textual comment and predicts multiple labels associated with the comment. 

### Other usage
To perform general multi-label classification with an image as input and to predict the image category and description.

### Multi-class Vs Multi-label classification
* In multi-class classification problem, an instance or a record can belong to one and only one of the multiple output classes. For instance, in the sentiment analysis problem, a text review could be either "good", "bad", or "average". It could not be both "good" and "average" at the same time. 
* In multi-label classification problems, an instance can have multiple outputs at the same time. For instance, in the comment toxicity problem, a comment can have multiple tags. These tags include "toxic", "obscene", "insulting", etc., at the same time.

## Dataset
* Comments from [Wikipedia's talk page edits](https://en.wikipedia.org/wiki/Help:Talk_pages). 
* There are six output labels for each comment: toxic, severe_toxic, obscene, threat, insult and identity_hate. 
* Kaggle Link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data | https://www.kaggle.com/likarajo/toxic-comments

## Dependencies
* Pandas
* Scikit-learn
* Numpy
* Tensorflow
* Keras
* Matplotlib
* Seaborn

## Create Multi-Label Text Classification model

### Model with Single Dense Output layer
* Single dense layer with six outputs 
    * Each neuron in the output dense layer will represent one of the six output labels.
* Sigmoid activation functions 
    * Return a value between 0 and 1 for each neuron.
    * If any neuron's output value is greater than 0.5, it is assumed that the comment belongs to the class represented by that particular neuron.
* Binary cross entropy loss functions

### Model with Multiple Dense Output layers
* Six dense output layers, one for each label.
* Each layer has its own Sigmoid activation function.

## Conclusion
* There can be two deep learning approaches for multi-label text classification.
    * We can use a single dense output layer with multiple neurons, each representing one label.
    * We can use separate dense layers with one neuron for each label. 
* In our case, single output layer with multiple neurons works better than multiple output layers.

## Improvements
* Change the activation function 
* Change the the train test split 
* Modify batch size and epochs
