# Handwritten Digits Recognition

![Digit recognition page](https://github.com/cau777/ai_playground/blob/master/docs/screenshots/digits_page.png)

### Training

This model is trained with the [MNIST database of handwritten digits](https://en.wikipedia.org/wiki/MNIST_database),
which contains 60,000 training images and 10,000 testing images of written digits. The whole process took about 16 hours
without using the GPU.

### Model structure

It's composed by 2 interspersed layers of Convolution and MaxPool operations, followed by 2 Deep layers.
ReLu is used throughout the model as the activation function. Also, 2 Dropout layers are placed to prevent overfitting.

### Outcome

The model has a tested accuracy of 90% (not very high because of the Dropout layers) and sometimes fails with 7 and 9.
However, the performance in the online playground is very satisfying.