## Development Objectives

Currently Talos yields state-of-the-art results (e.g. Iris dataset 100% and Wisconsin Breast Cancer dataset 99.4%) across a range of prediction tasks in a semi-automatic manner, while providing the simplest available method for hyperparameter optimization with Keras.

The development goals and current work include:

- to expose 100% of Keras functionality to the user
- to create robust synthesis of generality and performance in a single metric 
- to use the synthesis score as a metric for optimizing the hyperparameter optimization process
- to allow on-to-go model ensembling
- to allow *reverse inheritance* of the optimization capacity i.e. models that build models that...

These goals are currently being met by systematically building simple, easy-to-understand building blocks that solve manageable parts of the challenge.
