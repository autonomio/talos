def local_strategy(self):

    try:
        import importlib
        importlib.reload(talos_strategy)
    except NameError:
        from talos_strategy import talos_strategy

    self = talos_strategy(self)

    return self
