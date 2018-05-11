## Notes on Usage 

- Models need to have a model.fit() object and model in the return statement

- The model needs to be inside a function (which is passed to the talos.Scan()

## Options

In addition to the parameter, there are several options that can be set within the Scan() call. These values will effect the actual scan, as opposed to anything that change for each permutation.

#### val_split

The validation split that will be used for the experiment. By default .3 to validation data.

#### shuffle

If the data should be shuffle before validation split is performed. By default True.

#### search_method

Three modes are offered: 'random', 'linear', and 'reverse'. Random picks randomly one permutation and then removes it from the search grid. Linear starts from the beginning of the grid, and reverse from the end.

#### reduction_method

There is currently one reduction algorithm available 'spear'. It is based on an approach where depending on the 'reduction_interval' and 'reduction_window' poorly performing parameters are dropped from the scan. If you would like to see a specific algorithm implemented, please create an issue for it.

#### reduction_interval

The number of rounds / permutation attempts after which the reduction method will be applied. The 'reduction_method' must be set to other than None for this to take effect.

#### reduction_window

The number of rounds / permutation attempts for looking back when applying the reduction_method. For continuous optimization, this should be less than reduction_interval or the same.

#### grid_downsampling

Takes in a float value based on which a fraction of the total parameter grid will be picked randomly.

#### early_stopping

Provides a callback functionality where once val_loss (validation loss) is no longer dropping, based on the setting, the round will be terminated. Results for the round will be still recorded before moving on to the next permutation. Accepts a string values 'moderate' and 'strict', or a list with two int values (min_delta, patience). Where min_delta indicates the threshhold for change where the round will be flagged for termination (e.g. 0 means that val_loss is not changing) and patience indicates the number of epochs counting from the flag being raised before the round is actual terminated.

#### dataset_name

This information is used for the master log and naming the experiment results round results .csv file.

#### experiment_no

This will be appended to the round results .csv file and together with the dataset_name form a unique handler for the experiment.  

#### talos_log_name

The path to the master log file where a log entry is created for every single scan event together with meta-information such as what type of prediction challenge it is, how the data is transformed (e.g. one-hot encoded). This data can be useful for training models for the purpose of optimizing models. That's right, models that make models.

By default talos.log is in the present working directory. It's better to change this to something where it has persistence.

#### debug

Useful when you don't want records to be made in to the master log (./talos.log)
