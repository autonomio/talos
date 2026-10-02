"""Column aliases for result tables without changing live parameter names."""


_CATEGORY_PREFIX = '__talos_category__:'


def log_frame(result):
    """Typed parameter categories in a portable Polars view, without Arrow."""
    import numpy as np
    import polars as pl

    from .serialization import dumps

    def family(value):
        if isinstance(value, np.generic):
            native = value.item()
            kind = family(native) if not isinstance(native, np.generic) else 'object'
            return kind + ':numpy:' + str(value.dtype) if kind != 'object' else 'object'
        if isinstance(value, bool):
            return 'bool'
        if isinstance(value, (int, float)):
            return 'number:' + type(value).__name__
        if isinstance(value, str):
            return 'string'
        return 'object'

    encoded_columns = set()
    for name, column in result.parameter_columns.items():
        values = result.domain.values_for(name) if hasattr(result, 'domain') else []
        values = list(values) + [record['params'][name] for record in result._records if name in record['params']]
        families = {family(value) for value in values}
        if len(families) > 1 or 'object' in families:
            encoded_columns.add(column)
    rows = []
    for record in result._records:
        row = result._record_row(record)
        for column, value in row.items():
            if column in encoded_columns:
                row[column] = _CATEGORY_PREFIX + dumps(value)
            else:
                if isinstance(value, np.generic):
                    value = value.item()
                row[column] = value if value is None or isinstance(value, (str, bool, int, float)) else dumps(value)
        rows.append(row)
    return pl.from_dicts(rows, strict=False, infer_schema_length=None) if rows else pl.DataFrame(), encoded_columns


class LogQueueView:
    def __init__(self, queue, parameter_columns, encoded_columns=()):
        self.queue = queue
        self.columns = parameter_columns
        self.encoded_columns = set(encoded_columns)
        self.original = {column: name for name, column in parameter_columns.items()}

    def __getattr__(self, name):
        return getattr(self.queue, name)

    @property
    def domain_keys(self):
        return [self.columns.get(name, name) for name in self.queue.domain_keys]

    def distribution(self, param=None):
        if param is None:
            return {name: self.distribution(name) for name in self.domain_keys}
        original = self.original.get(param, param)
        values = self.queue.distribution(original)
        if param in self.encoded_columns:
            from .serialization import dumps
            count = next(iter(values.values()), 0)
            return {_CATEGORY_PREFIX + dumps(value): count for value in self.queue._domain.values_for(original)}
        return values

    def resolve_log_value(self, param, value):
        original = self.original.get(param, param)
        if param in self.encoded_columns and isinstance(value, str) and value.startswith(_CATEGORY_PREFIX):
            from .serialization import dumps
            encoded = value[len(_CATEGORY_PREFIX):]
            for candidate in self.queue._domain.values_for(original):
                if dumps(candidate) == encoded:
                    return candidate
            import json

            from .serialization import decode
            return decode(json.loads(encoded))
        import numpy as np

        from .param_domain import values_equal
        candidates = self.queue._domain.values_for(original)
        matching = [candidate for candidate in candidates if values_equal(candidate, value)]
        if not matching:
            matching = [candidate for candidate in candidates
                        if isinstance(candidate, np.generic) and values_equal(candidate.item(), value)]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            raise ValueError(f'Ambiguous typed parameter category for {original!r}')
        return self.queue.resolve_log_value(original, value)

    def remove_is(self, param, value):
        return self.queue.remove_is(self.original.get(param, param), self.resolve_log_value(param, value))

    def remove_ge(self, param, threshold):
        return self.queue.remove_ge(self.original.get(param, param), threshold)

    def remove_le(self, param, threshold):
        return self.queue.remove_le(self.original.get(param, param), threshold)

    def keep_is(self, param, value):
        return self.queue.keep_is(self.original.get(param, param), self.resolve_log_value(param, value))

    def keep_between(self, param, lower, upper):
        return self.queue.keep_between(self.original.get(param, param), lower, upper)

    def inject_value(self, param, value):
        return self.queue.inject_value(self.original.get(param, param), value)

    def inject(self, combo, prioritize=False):
        return self.queue.inject({self.original.get(key, key): value for key, value in combo.items()}, prioritize=prioritize)

    def set_filter(self, key, condition, *, filter_type=None, filter_params=None):
        if filter_params is not None:
            from talos.experiment.reducer.filter_types import FILTER_BUILDERS
            filter_params = dict(filter_params)
            if 'param' in filter_params:
                filter_params['param'] = self.original.get(filter_params['param'], filter_params['param'])
            if 'params' in filter_params:
                filter_params['params'] = [self.original.get(name, name) for name in filter_params['params']]
            condition = FILTER_BUILDERS[filter_type](filter_params)
        return self.queue.set_filter(key, condition, filter_type=filter_type, filter_params=filter_params)
