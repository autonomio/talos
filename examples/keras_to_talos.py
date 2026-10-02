"""Compare one Keras fit with a native Talos sweep on bundled Iris data.

Install the maintained generation for this example::

    python -m pip install 'talos[keras,torch] @ git+https://github.com/autonomio/talos.git@master'

Choose the backend before running this file:
``KERAS_BACKEND=torch python examples/keras_to_talos.py``.
Importing the module neither loads Keras nor trains a model. The validation
split guides model selection; its accuracy is not a final test-set estimate.
"""

from pathlib import Path

backend = 'keras'


def prepare_data():
    """Return caller-owned 120/30 splits, scaling from training rows only."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=17,
    )
    scaler = StandardScaler().fit(x_train)
    return {
        'x_train': scaler.transform(x_train).astype('float32'),
        'y_train': y_train,
        'x_val': scaler.transform(x_val).astype('float32'),
        'y_val': y_val,
    }


def existing_model(x_train, y_train, x_val, y_val, params):
    """Fit the same network for Keras, legacy Scan and the native SFD.

    Fixed initialization makes the parameter comparison repeatable. Defaults
    preserve the epochs-only callback used in the migration example.
    """
    import keras

    keras.utils.set_random_seed(17)
    network = keras.Sequential([
        keras.layers.Input(shape=(x_train.shape[1],)),
        keras.layers.Dense(params.get('units', 8), activation='relu'),
        keras.layers.Dense(3, activation='softmax'),
    ])
    network.compile(
        optimizer=keras.optimizers.Adam(params.get('learning_rate', 0.01)),
        loss='sparse_categorical_crossentropy', metrics=['accuracy'],
    )
    history = network.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=params.get('epochs', 5), batch_size=params.get('batch_size', 16),
        verbose=0,
    )
    return history, network


def params():
    """Expose four parameter combinations to the native Talos executor."""
    return {
        'units': [8, 16], 'learning_rate': [0.01, 0.03],
        'epochs': [5], 'batch_size': [16],
    }


def prep(data, round_params):
    """Use the caller's prepared split for every trial."""
    return data


def model(data, round_params):
    """Keep the existing five-argument fitting callback unchanged."""
    return existing_model(
        data['x_train'], data['y_train'], data['x_val'], data['y_val'],
        round_params,
    )


def verify_results(result, recovered, my_splits):
    """Check durable trial rows and metrics against actual model predictions."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, log_loss

    pd.testing.assert_frame_equal(result.data, recovered.data[result.data.columns])
    assert result.round_history == recovered.round_history
    assert len(result.data) == 4
    assert result.data['_trial_id'].nunique() == 4
    assert set(zip(result.data['units'], result.data['learning_rate'])) == {
        (8, 0.01), (8, 0.03), (16, 0.01), (16, 0.03),
    }
    for model_id, row in result.data.iterrows():
        live = result.predict(my_splits['x_val'], model_id=model_id)
        restored = recovered.predict(my_splits['x_val'], model_id=model_id)
        np.testing.assert_allclose(restored, live, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            log_loss(my_splits['y_val'], restored, labels=[0, 1, 2]),
            row['val_loss'], rtol=1e-5, atol=1e-6,
        )
        np.testing.assert_allclose(
            accuracy_score(my_splits['y_val'], restored.argmax(axis=1)),
            row['val_accuracy'], rtol=1e-6, atol=1e-7,
        )


def main(output_dir):
    """Run the paired example and preserve the sweep for later recovery."""
    import json

    import numpy as np
    import talos

    my_splits = prepare_data()
    history, network = existing_model(
        my_splits['x_train'], my_splits['y_train'],
        my_splits['x_val'], my_splits['y_val'],
        {'units': 8, 'learning_rate': 0.01, 'epochs': 5, 'batch_size': 16},
    )
    baseline_predictions = network.predict(my_splits['x_val'], verbose=0)

    result = talos.run(
        __file__, data=my_splits, experiment_dir=output_dir / 'sweep',
        objective={'metric': 'val_loss', 'direction': 'min'}, seed=17,
        save_models=True, retain_models=True, prep_each_round=False,
        progress_bar=False,
    )
    recovered = talos.RunResult.load(result.run_dir)
    predictions = recovered.predict(my_splits['x_val'])
    verify_results(result, recovered, my_splits)
    best_index = result.data['val_loss'].idxmin()
    np.testing.assert_allclose(
        predictions, result.predict(my_splits['x_val'], model_id=best_index),
        rtol=1e-6, atol=1e-7,
    )

    np.savez(output_dir / 'iris-splits.npz', **my_splits)
    np.save(output_dir / 'baseline-predictions.npy', baseline_predictions)
    np.save(output_dir / 'best-predictions.npy', predictions)
    summary = {
        'dataset': 'scikit-learn bundled Iris', 'split_seed': 17,
        'training_rows': len(my_splits['x_train']),
        'validation_rows': len(my_splits['x_val']),
        'model_initialization_seed': 17,
        'baseline_val_loss': history.history['val_loss'][-1],
        'baseline_val_accuracy': history.history['val_accuracy'][-1],
        'trial_count': len(result.data), 'run_dir': str(result.run_dir),
        'objective': result.objective, 'best_model_id': int(best_index),
        'best_val_loss': float(result.data.loc[best_index, 'val_loss']),
        'best_val_accuracy': float(result.data.loc[best_index, 'val_accuracy']),
        'restored_prediction_shape': list(predictions.shape),
    }
    (output_dir / 'comparison-results.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))
    print(result.data[['units', 'learning_rate', 'val_loss', 'val_accuracy']])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('keras-talos-comparison'))
    arguments = parser.parse_args()
    main(arguments.output_dir.resolve())
