from talos.scalers.linear_scaler import LinearScaler, inverse_transform


class LogRegScaler(LinearScaler):
    """General linear scaling with caller rules; compatible public scaler name."""
