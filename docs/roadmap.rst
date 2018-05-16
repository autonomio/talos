Development Objectives
======================

The development goals and current work include:

- to expose 100% of Keras functionality to the user
- to create robust synthesis of generality and performance in a single metric 
- to use the synthesis score as a metric for optimizing the hyperparameter optimization process
- to leverage gradient based optimization approaches
- to allow on-the-go model ensembling
- to allow ensembling of model ensembles
- to allow *reverse inheritance* of the optimization capacity i.e. models that build models that...

These goals are currently being met by systematically building simple, easy-to-understand building blocks that solve manageable parts of the challenge.

Immediate Development
---------------------

- Show false negatives / false positives  / border cases
- Robust cross-validation
- Generalization measure
- Prediction task agnostic performance measure
- Adding layer generation for other than Dense layers
- Current custom f1 implementation into Keras metric
- Add support for fastai?
