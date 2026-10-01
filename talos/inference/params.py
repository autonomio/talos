"""Recover original trial parameters independently of metric-column aliases."""


def trial_params(result, row, keys=None):
    identifier = row.get('_trial_id', row.get('id'))
    for record in getattr(result, '_records', []):
        recorded = record.get('row', {})
        recorded_id = recorded.get('_trial_id', recorded.get('id'))
        if identifier is not None and str(recorded_id) == str(identifier) and 'params' in record:
            return dict(record['params'])
    if keys is None:
        domains = getattr(result, 'params', {})
        keys = domains.keys() if isinstance(domains, dict) else getattr(result, '_param_dict_keys', [])
    aliases = getattr(result, 'parameter_columns', {})
    return {key: row[aliases.get(key, key)] for key in keys if aliases.get(key, key) in row}
