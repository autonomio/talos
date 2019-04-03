import talos as ta


def test_random_methods():

    '''Tests all the available random methods
    in reducers/sample_reducer.py that are invoked
    that are invoked through Scan(random_method)'''

    print('Start testing random methods...')

    random_methods = ['sobol',
                      'quantum',
                      'halton',
                      'korobov_matrix',
                      'latin_sudoku',
                      'latin_matrix',
                      'latin_improved',
                      'uniform_mersenne',
                      'uniform_crypto',
                      'ambience']

    for method in random_methods:
        ta.templates.pipelines.titanic(random_method=method)

    return "Finished testing random methods!"
