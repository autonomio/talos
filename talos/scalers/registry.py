from talos.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from talos.scalers.linear_scaler import LinearScaler
from talos.scalers.logreg_scaler import LogRegScaler
from talos.scalers.robust_scaler import RobustScaler
from talos.scalers.rank_gauss_scaler import RankGaussScaler

SCALER_REGISTRY: dict[str, type] = {
    'linear': LinearScaler,
    'logreg': LogRegScaler,
    'robust': RobustScaler,
    'rank_gauss': RankGaussScaler,
    'causal_rolling_robust': CausalRollingRobustScaler,
}
