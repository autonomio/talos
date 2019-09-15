def test_lr_normalizer():
    '''Test learning rate normalizer to confirm an invalid type is
    recognized and throws TalosModelError.'''

    from talos.model.normalizers import lr_normalizer
    from talos.utils.exceptions import TalosModelError

    print('Testing lr_normalizer() and invalid optimizer type...')

    # Using string as proxy for any invalid class
    # (ex., tensorflow-sourced optimizer)
    bad_optimizer = 'test'

    try:
        lr_normalizer(1, bad_optimizer)
    except TalosModelError:
        print('Invalid model optimizer caught successfully!')
    else:
        print('Invalid (string) model optimizer type not caught.')
