def test_distribute():

    import talos
    from numpy import loadtxt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from talos import DistributeScan

    dataset = loadtxt(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        delimiter=",",
    )

    x = dataset[:, 0:8]
    y = dataset[:, 8]

    def diabetes(x_train, y_train, x_val, y_val, params):

        # replace the hyperparameter inputs with references to params dictionary
        model = Sequential()
        model.add(
            Dense(params["first_neuron"], input_dim=8, activation=params["activation"])
        )
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        # make sure history object is returned by model.fit()
        out = model.fit(
            x=x,
            y=y,
            validation_data=[x_val, y_val],
            epochs=4,
            batch_size=params["batch_size"],
            verbose=0,
        )

        # modify the output model
        return out, model

    p = {
        "first_neuron": [12, 24, 48],
        "activation": ["relu", "elu"],
        "batch_size": [10, 20, 40],
    }

    exp_name = "diabetes_exp"

    import os

    filepath = os.path.abspath(
        __file__
    )  # mention the current filepath so that it can be distributed in multiple machines.
    print(filepath)
    t = DistributeScan(
        x=x,
        y=y,
        params=p,
        model=diabetes,
        experiment_name=exp_name,
        file_path=filepath,
        destination_path="./five.py",
    )
    t.distributed_run(run_central_node=True, show_results=True)
