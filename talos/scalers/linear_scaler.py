import re
from typing import cast

import polars as pl

DEFAULT_SCALING_RULES = {r'.*': 'standard'}


def build_rules(
    overrides: dict[str, list[str]] | None = None,
    base_rules: dict[str, str] | None = None,
) -> dict[str, str]:

    """
    Build scaling rules by combining base rules and user overrides.

    Args:
        overrides: User-specified rules in sklearn style, e.g.
            {'standard': ['measurement'], 'log_standard': ['count']}.
        base_rules: Regex-based rules to start with.
    """

    rules = {}

    if overrides:
        for rule, cols in overrides.items():
            for col in cols:
                rules[fr'^{col}$'] = rule

    rules.update(base_rules or DEFAULT_SCALING_RULES)
    return rules


def get_scaling_rule(col: str, rules: dict[str, str], default: str = 'none') -> str:

    """
    Find the matching scaling rule for a column name.

    Args:
        col: Column name.
        rules: Regex-to-rule mapping.
        default: Rule to use if no pattern matches.

    Returns:
        The scaling rule name.
    """

    for pattern, rule in rules.items():
        if re.match(pattern, col):
            return rule

    return default


class LinearScaler:
    def __init__(
        self,
        x_train: pl.DataFrame,
        rules: dict[str, str] | None = None,
        default: str = 'standard',
    ) -> None:

        """
        Linear transformation utility for scaling features.

        Args:
            x_train: Training DataFrame.
            rules: Regex-to-rule mapping.
            default: Fallback scaling rule.
        """

        super().__init__()

        self.rules = rules or DEFAULT_SCALING_RULES
        self.default = default
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

        for col in x_train.columns:
            if not x_train[col].dtype.is_numeric():
                continue
            rule = get_scaling_rule(col, self.rules, self.default)

            if rule == "log_standard":
                mean = x_train.select(pl.col(col).log1p().mean()).item()
                std = x_train.select(pl.col(col).log1p().std(ddof=0)).item()
            elif rule == "standard":
                mean = x_train[col].mean()
                std = x_train[col].std(ddof=0)
            else:
                continue

            self.means[col] = cast(float, mean)
            self.stds[col] = float(std) if std is not None and std != 0 else 1.0

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        """
        Apply linear scaling transformation.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """

        exprs: list[pl.Expr] = []
        for col in df.columns:
            if col not in self.means and get_scaling_rule(col, self.rules, self.default) in ('standard', 'log_standard'):
                continue
            rule = get_scaling_rule(col, self.rules, self.default)

            if rule == 'standard':
                exprs.append(((pl.col(col) - self.means[col]) / self.stds[col]).alias(col))

            elif rule == 'log_standard':
                exprs.append(((pl.col(col).log1p() - self.means[col]) / self.stds[col]).alias(col))

            elif rule == 'divide_100':
                exprs.append((pl.col(col) / 100).alias(col))

            elif rule == 'none':
                exprs.append(pl.col(col).alias(col))

        return df.with_columns(exprs)


def inverse_transform(df: pl.DataFrame, scaler: LinearScaler) -> pl.DataFrame:

    """
    Apply inverse scaling transformation.

    Args:
        df: DataFrame to inverse transform.
        scaler: LinearScaler instance with fitted parameters.

    Returns:
        DataFrame in original scale.
    """

    exprs: list[pl.Expr] = []
    for col in df.columns:
        if col not in scaler.means and get_scaling_rule(col, scaler.rules, scaler.default) in ('standard', 'log_standard'):
            continue
        rule = get_scaling_rule(col, scaler.rules, scaler.default)

        if rule == 'standard':
            exprs.append((pl.col(col) * scaler.stds[col] + scaler.means[col]).alias(col))

        elif rule == 'log_standard':
            exprs.append(((pl.col(col) * scaler.stds[col] + scaler.means[col]).exp() - 1).alias(col))

        elif rule == 'divide_100':
            exprs.append((pl.col(col) * 100).alias(col))

        elif rule == 'none':
            exprs.append(pl.col(col).alias(col))

    return df.with_columns(exprs)
