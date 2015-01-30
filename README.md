# NeuroBricks
A theano warpper for implementing, training and analysing neural nets with convenience. 

## Core Features
 - **Conveniency**

By using pre-defined layers and overiding the "+" operator, we can build deep networks within one line. Now building deep networks is as easy as stacking bricks together.
 
 - **Convertibility**

Easy to rearrange diffrent layers, be it pretrained or not, into new models. It allows you to build a network with different kind of layers, or reuse trained layers from some other models. It also brings convenience to form ensumbles.

 - **Separability**

We make training methods completely separated to the models. With this separability, you can train a network with any accessible training methods. 
 
We also ensures that diffent kinds of training mechanisms get independent to each other. This allows you to combine different tricks together to form very complicated training procedures, like combining Dropout, Feedback Alignment, and unsupervised pretraining together, without having to define a new training class.

 - **Interactive Analysis**

A set of analyzing and visualizing methods are built into the definition of the models. So you are going to have a lot of analysing methods at hand _right after_ your model is created. Most analysing methods allow interactive data updating, you can see how the weights/activations are changing during training epoches.
 
## Installation
This part is not finished at the moment. But you can do the following to allow for the import:

 - Add the following line to your .bashrc file:

    export PYTHONPATH=/your-installation-path/NeuroBricks:$PYTHONPATH


