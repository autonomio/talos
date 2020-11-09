def test_random_methods():

    '''Tests all the available random methods
    in reducers/sample_reducer.py that are invoked
    that are invoked through Scan(random_method)'''

    print('\n >>> start Random Methods... \n')

    import talos

    random_methods = ['sobol',
                      'quantum',
                      'halton',
                      'korobov_matrix',
                      'latin_sudoku',
                      'latin_matrix',
                      'latin_improved',
                      'uniform_mersenne',
                      'uniform_crypto',
                      'ambience'
                      ]

    for method in random_methods:
        talos.templates.pipelines.titanic(random_method=method)

    print('finish Random Methods \n')
