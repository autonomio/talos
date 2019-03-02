import talos as ta


def test_auto_scan():

    '''Tests the object from Params()'''

    print('Start auto Scan()...')

    x, y = ta.templates.datasets.breast_cancer()
    x = x[:50]
    y = y[:50]

    p = ta.Params().params

    for key in p.keys():
        p[key] = [p[key][0]]

    ta.Scan(x, y, p, ta.KerasModel().model,
            permutation_filter=lambda p: p['batch_size'] < 150,)

    return "Finished testing auto Scan()"
