def local_strategy(self):

    try:
        import importlib
        importlib.reload(gamify)
    except NameError:
        from gamify import gamify

    self = gamify(self)

    return self
