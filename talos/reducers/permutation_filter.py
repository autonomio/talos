def permutation_filter(self, ls, final_grid_size, virtual_grid_size):

    '''Handles the filtering for ta.Scan(... permutation_filter= ...)'''

    from ..parameters.round_params import create_params_dict

    # handle the filtering with the current params grid

    def fn(i):

        params_dict = create_params_dict(self, i)
        fn = self.main_self.permutation_filter(params_dict)

        return fn

    grid_indices = list(filter(fn, range(len(self.param_grid))))
    self.param_grid = self.param_grid[grid_indices]
    final_expanded_grid_size = final_grid_size

    while len(self.param_grid) < final_grid_size and final_expanded_grid_size < virtual_grid_size:
        final_expanded_grid_size *= 2

        if final_expanded_grid_size > virtual_grid_size:
            final_expanded_grid_size = virtual_grid_size

        self.param_grid = self._create_param_grid(ls,
                                                  final_expanded_grid_size,
                                                  virtual_grid_size)

        grid_indices = list(filter(fn, range(len(self.param_grid))))
        self.param_grid = self.param_grid[grid_indices]

    self.param_grid = self.param_grid[:final_grid_size]

    return self
