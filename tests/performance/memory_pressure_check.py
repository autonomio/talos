if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import os

    print('\n Memory Pressure Test Starts...\n')

    for i in os.listdir():
        if 'mprofile_' in i:
            df = pd.read_csv(i, sep=' ', error_bad_lines=False)

    df.columns = ['null', 'memory', 'time']
    df.drop('null', 1, inplace=True)

    std_limit = 5
    highest_limit = 800

    std = np.std(np.array(df.memory.values[1500:]))
    highest = df.memory.max()

    if std > std_limit:
        raise Exception('MEMORY TEST FAILED: Standard deviation of memory pressure is %d which is above the %d limit' % (std, std_limit))

    if highest > highest_limit:
        raise Exception('MEMORY TEST FAILED: Max memory is %d which is above the %d limit' % (highest, highest_limit))

    print("\n Memory Pressure Test Passed \n")
