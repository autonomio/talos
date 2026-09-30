import pandas as pd


def rescale_meanzero(x):
    frame = pd.DataFrame(x).copy()
    for column in frame.select_dtypes(include='number'):
        scale = frame[column].std()
        frame[column] = (frame[column] - frame[column].mean()) / (1 if scale == 0 else scale)
    return frame.to_numpy()
