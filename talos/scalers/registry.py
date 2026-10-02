"""Resolve named scalers without acquiring caller data."""

from talos.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from talos.scalers.linear_scaler import LinearScaler
from talos.scalers.logreg_scaler import LogRegScaler
from talos.scalers.rank_gauss_scaler import RankGaussScaler
from talos.scalers.robust_scaler import RobustScaler

SCALER_REGISTRY: dict[str, type] = {
    'linear': LinearScaler, 'logreg': LogRegScaler,
    'robust': RobustScaler,
    'rank_gauss': RankGaussScaler,
    'causal_rolling_robust': CausalRollingRobustScaler,
}
