def local_strategy(self):

    try:
        import importlib
        importlib.reload(talos_strategy)
        self = talos_strategy(self)
    except NameError:
        try:
            from talos_strategy import talos_strategy
            self = talos_strategy(self)
        except ImportError:
            print("No talos_strategy.py found in pwd. Nothing is done.")

    return self
