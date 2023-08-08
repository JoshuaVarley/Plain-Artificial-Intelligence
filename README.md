# Plain-Neural-Network
PAI - Plain Neural Network, written in C#, from scratch.
This is a simple neural network trained on MNIST, as a learning experience, it is easily configurable in code and can be reworked to fit every need.
By default, it uses, ReLU combined with Softmax as the activation functions, and Binary Cross entropy as the loss function, but other activation functions and loss functions are included.
After 8 epochs it reached a percentage correctly guessed on the MNIST test data of approx. 97%.
The network can be saved and loaded using Binary Serialization.
In the future, this project will be attempted to be multithreaded.
Feel free to add any changes to the code that may help the network learn better :)


The MNIST Dataset used for this project can be downloaded here: [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

<img width="510" alt="paipic" src="https://github.com/AreOlsen/Plain-Neural-Network/assets/58704301/eaff9bd8-4ed0-4eb0-a867-408890094887">
