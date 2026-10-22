"""Dtype encoding and aggregation helpers for extracted indicator data.

Provides non-continuous dtype encoding (binary, ordinal, categorical -> numeric)
and Polars aggregation expression builders used by the pipeline's stage 2 logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import polars as pl

from grassmoth.public import (
    ComputedExtractionSpec,
    ObservationInstant,
    SummaryOperator,
    WindowExpression,
    WindowOperator,
)
from grassmoth_server.actions.prepare_data.rows import support_window_tick_frame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars._typing import FillNullStrategy

    from grassmoth_server.actions.prepare_data.context import MeasurementContext


def _compile_computed_rule_expr(window_expr: WindowExpression) -> pl.Expr:
    """Lower the retained window grammar without parsing or reconstruction."""
    return window_expr.fold(
        literal=pl.lit,
        column=pl.col,
        collection=lambda values: pl.lit(list(values)),
        operation=_compile_window_operation,
    )


def _compile_window_operation(operator: WindowOperator, args: tuple[pl.Expr, ...]) -> pl.Expr:
    match operator:
        case WindowOperator.ADD:
            return args[0] + args[1]
        case WindowOperator.SUBTRACT:
            return args[0] - args[1]
        case WindowOperator.MULTIPLY:
            return args[0] * args[1]
        case WindowOperator.DIVIDE:
            return args[0] / args[1]
        case WindowOperator.MODULO:
            return args[0] % args[1]
        case WindowOperator.POWER:
            return args[0].pow(args[1])
        case WindowOperator.NEGATE:
            return -args[0]
        case WindowOperator.POSITIVE:
            return args[0]
        case WindowOperator.NOT:
            return ~args[0].fill_null(False)
        case WindowOperator.AND | WindowOperator.OR:
            result = args[0]
            for arg in args[1:]:
                result = result & arg if operator == WindowOperator.AND else result | arg
            return result
        case WindowOperator.EQ:
            return args[0] == args[1]
        case WindowOperator.NE:
            return args[0] != args[1]
        case WindowOperator.LT:
            return args[0] < args[1]
        case WindowOperator.LE:
            return args[0] <= args[1]
        case WindowOperator.GT:
            return args[0] > args[1]
        case WindowOperator.GE:
            return args[0] >= args[1]
        case WindowOperator.IN:
            return args[0].is_in(args[1])
        case WindowOperator.NOT_IN:
            return ~args[0].is_in(args[1])
        case WindowOperator.IS:
            return args[0].is_null()
        case WindowOperator.IS_NOT:
            return args[0].is_not_null()
        case WindowOperator.IF:
            return pl.when(args[0]).then(args[1]).otherwise(args[2])
        case WindowOperator.CONTAINS:
            return (
                args[0]
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .str.contains(args[1].str.to_lowercase(), literal=True)
            )
        case WindowOperator.CONTAINS_ANY:
            return (
                args[0]
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .str.contains_any(args[1].list.eval(pl.element().str.to_lowercase()))
            )
        case WindowOperator.ABS:
            return args[0].abs()
        case WindowOperator.ALL:
            return (
                pl.when(args[0].is_not_null().any())
                .then(args[0].fill_null(False).all())
                .otherwise(None)
            )
        case WindowOperator.ANY:
            return args[0].fill_null(False).any()
        case WindowOperator.COALESCE:
            return pl.coalesce(args)
        case WindowOperator.COUNT_NON_NULL:
            return args[0].is_not_null().cast(pl.Int64).sum()
        case WindowOperator.COUNT_TRUE:
            return args[0].fill_null(False).cast(pl.Int64).sum()
        case WindowOperator.FIRST:
            return args[0].drop_nulls().first()
        case WindowOperator.LAST:
            return args[0].drop_nulls().last()
        case WindowOperator.LOWER:
            return args[0].cast(pl.Utf8, strict=False).str.to_lowercase()
        case WindowOperator.MAX:
            return args[0].max()
        case WindowOperator.MEAN:
            return args[0].mean()
        case WindowOperator.MIN:
            return args[0].min()
        case WindowOperator.STD:
            return args[0].std()
        case WindowOperator.SUM:
            return pl.when(args[0].is_not_null().any()).then(args[0].sum()).otherwise(None)
        case _:
            assert_never(operator)


def _build_agg_expr(operator: SummaryOperator, col_name: str = "value") -> pl.Expr:
    """Lower the owned summary vocabulary to a Polars reduction."""
    col = pl.col(col_name)
    match operator:
        case SummaryOperator.MEAN:
            expr = col.mean()
        case SummaryOperator.SUM:
            expr = pl.when(col.is_not_null().any()).then(col.sum()).otherwise(None)
        case SummaryOperator.STD:
            expr = col.std()
        case SummaryOperator.LAST:
            expr = col.drop_nulls().last()
        case SummaryOperator.FIRST:
            expr = col.drop_nulls().first()
        case SummaryOperator.COUNT:
            expr = col.drop_nulls().count()
        case _:
            assert_never(operator)
    return expr.alias("value")


def compute_indicators(
    raw_df: pl.DataFrame,
    measurement_structure: MeasurementContext,
    time_col: str,
) -> pl.DataFrame:
    """Compute indicator values directly via Polars aggregation.

    For variables with a computed extraction recipe, applies a deterministic
    support-window computation grouped by each indicator's effective
    observation window (explicit observation_window or fallback model_clock).
    Direct single-column aggregations are supported, along with computed_rule
    expressions that deterministically derive one scalar per support window
    from one or more source columns. Non-numeric direct columns are supported
    for point aggregations (`first`/`last`), and ordinal direct columns are
    converted to their declared integer codes before emission.

    Args:
        raw_df: Raw wide-format DataFrame with actual column names.
        measurement_structure: Owned variable definitions, fallback clock and source span.
        time_col: Name of the datetime column in raw_df.

    Returns:
        Long-format DataFrame with columns: indicator (Utf8), value (Utf8),
        timestamp (Utf8). Matches the schema produced by the semantic path.
    """
    start, end = measurement_structure.source.start, measurement_structure.source.end
    output_schema = {"indicator_id": pl.Utf8, "value": pl.Utf8, "timestamp": pl.Utf8}
    computed = tuple(
        (ind, extraction)
        for ind in measurement_structure.indicators
        if isinstance(extraction := ind.extraction, ComputedExtractionSpec)
    )
    if not computed:
        return pl.DataFrame(schema=output_schema)

    df = ObservationInstant.normalize_column(raw_df, time_col).sort(time_col)

    frames: list[pl.DataFrame] = []
    for ind, extraction in computed:
        agg_name = ind.observation.aggregation
        measurement_dtype = ind.observation.measurement_dtype
        observation_window = measurement_structure.window(ind).polars_interval
        tick_frame = support_window_tick_frame(
            df, observation_window, time_col, start=start, end=end
        )
        selected = (
            df.with_columns(pl.col(time_col).dt.truncate(observation_window).alias("__tick__"))
            .join(tick_frame, on="__tick__", how="semi")
            .drop("__tick__")
        )
        source_columns = list(extraction.source_columns)
        computed_rule = extraction.computed_rule
        fill_null = extraction.fill_null

        missing_source_cols = [column for column in source_columns if column not in df.columns]
        if missing_source_cols:
            continue

        if computed_rule:
            prepared = _prepare_computed_rule_frame(
                selected,
                time_col=time_col,
                source_columns=source_columns,
                observation_window=observation_window,
            )
            prepared = _with_dense_support_rows(prepared, tick_frame)
            expr = _missing_window_guard(
                _compile_computed_rule_expr(computed_rule),
            )
            agg_df = prepared.group_by("__tick__", maintain_order=True).agg(expr)
        else:
            source_col = source_columns[0]
            prepared = _prepare_computed_indicator_frame(
                selected,
                time_col=time_col,
                source_col=source_col,
                observation_window=observation_window,
                measurement_dtype=measurement_dtype,
                ordinal_levels=ind.observation.ordinal_levels,
            )
            prepared = _with_dense_support_rows(prepared, tick_frame)

            expr = _missing_window_guard(_build_agg_expr(agg_name, "__value__"))
            agg_df = prepared.group_by("__tick__", maintain_order=True).agg(expr)

        if fill_null is not None:
            agg_df = agg_df.sort("__tick__").with_columns(
                fill_null_expression(pl.col("value"), fill_null, limit=extraction.fill_null_limit)
            )
        agg_df = agg_df.select(
            pl.lit(ind.observation.id).alias("indicator_id"),
            pl.col("value").cast(pl.Utf8).alias("value"),
            pl.col("__tick__").dt.to_string("%Y-%m-%dT%H:%M:%S").alias("timestamp"),
        )
        frames.append(agg_df)

    if not frames:
        return pl.DataFrame(schema=output_schema)

    return pl.concat(frames, how="vertical").sort("timestamp", "indicator_id")


def fill_null_expression(
    value: pl.Expr, fill_null: FillNullStrategy | float, *, limit: int | None = None
) -> pl.Expr:
    """Execute a flat observation declaration with Polars' native null-filling semantics."""
    if isinstance(fill_null, str):
        return value.fill_null(strategy=fill_null, limit=limit)
    return value.fill_null(value=fill_null)


def _missing_window_guard(expr: pl.Expr) -> pl.Expr:
    """Represent empty windows as null before any across-window fill_null operation."""
    has_records = pl.col("__observed_row__").fill_null(False).any()
    return pl.when(has_records).then(expr).otherwise(None).alias("value")


def _prepare_computed_indicator_frame(
    df: pl.DataFrame,
    *,
    time_col: str,
    source_col: str,
    observation_window: str,
    measurement_dtype: str,
    ordinal_levels: Sequence[str] | None,
) -> pl.DataFrame:
    """Prepare a computed indicator's source values for deterministic aggregation."""
    value_expr = _computed_value_expr(
        source_col,
        measurement_dtype=measurement_dtype,
        ordinal_levels=ordinal_levels,
    ).alias("__value__")
    return df.select(
        pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
        value_expr,
    )


def _with_dense_support_rows(prepared: pl.DataFrame, tick_frame: pl.DataFrame) -> pl.DataFrame:
    """Add null-valued placeholder rows for support windows with no raw rows."""
    if tick_frame.is_empty():
        return prepared.with_columns(pl.lit(True).alias("__observed_row__"))
    observed = prepared.with_columns(pl.lit(True).alias("__observed_row__"))
    return tick_frame.join(observed, on="__tick__", how="left", maintain_order="left_right")


def _prepare_computed_rule_frame(
    df: pl.DataFrame,
    *,
    time_col: str,
    source_columns: list[str],
    observation_window: str,
) -> pl.DataFrame:
    """Prepare source columns for a deterministic support-window computed rule."""
    return df.select(
        pl.col(time_col).dt.truncate(observation_window).alias("__tick__"),
        *[pl.col(column) for column in source_columns],
    )


def _computed_value_expr(
    source_col: str,
    *,
    measurement_dtype: str,
    ordinal_levels: Sequence[str] | None,
) -> pl.Expr:
    """Build the deterministic source-value expression for a computed indicator."""
    source = pl.col(source_col)
    if measurement_dtype == "ordinal":
        max_code = len(ordinal_levels or []) - 1
        label_map = {
            normalize_level_label(level): idx for idx, level in enumerate(ordinal_levels or [])
        }
        return source.map_elements(
            lambda value, _max=max_code, _label_map=label_map: _coerce_ordinal_code(
                value, _label_map, _max
            ),
            return_dtype=pl.Int64,
        )
    return source


def _coerce_ordinal_code(
    value: object,
    label_map: dict[str, int],
    max_code: int,
) -> int | None:
    """Normalize an ordinal source value to its canonical integer code."""
    if value is None or isinstance(value, bool):
        return None

    code: int | None = None
    if isinstance(value, int):
        code = value
    elif isinstance(value, float):
        if value.is_integer():
            code = int(value)
    elif isinstance(value, str):
        normalized = normalize_level_label(value)
        if not normalized:
            return None
        numeric = pl.Series([normalized]).cast(pl.Float64, strict=False).item()
        if numeric is None:
            code = label_map.get(normalized)
        elif numeric.is_integer():
            code = int(numeric)

    if code is None:
        return None
    if code < 0:
        return None
    if max_code >= 0 and code > max_code:
        return None
    return code


_BINARY_TRUE = {"true", "yes", "1", "1.0", "t", "y"}
_BINARY_FALSE = {"false", "no", "0", "0.0", "f", "n"}


def _encode_non_continuous(
    df: pl.DataFrame,
    dtype_lookup: dict[str, str],
    ordinal_levels_lookup: Mapping[str, Sequence[str]] | None = None,
    categorical_levels_lookup: Mapping[str, Sequence[str]] | None = None,
) -> pl.DataFrame:
    """Encode non-continuous indicator values to numeric before Float64 cast.

    - binary: map true/false/yes/no/1/0 → 1.0/0.0
    - ordinal/categorical: encode using the declared, stable codebook
    - continuous/count: no-op (already numeric)

    Modifies the 'value' column in-place per indicator partition.
    """
    if not dtype_lookup:
        return df

    ordinal_levels_lookup = ordinal_levels_lookup or {}
    categorical_levels_lookup = categorical_levels_lookup or {}

    non_continuous = {
        name: dtype
        for name, dtype in dtype_lookup.items()
        if dtype in ("binary", "ordinal", "categorical")
    }
    if not non_continuous:
        return df

    # Ensure value is Utf8 for string matching
    if df.schema.get("value") != pl.Utf8:
        df = df.with_columns(pl.col("value").cast(pl.Utf8, strict=False))

    frames = []
    remaining_mask = pl.lit(True)

    for name, dtype in non_continuous.items():
        indicator_mask = pl.col("indicator_id") == name
        subset = df.filter(indicator_mask)
        if subset.is_empty():
            continue

        remaining_mask = remaining_mask & ~indicator_mask

        if dtype == "binary":
            subset = subset.with_columns(
                pl.col("value")
                .str.to_lowercase()
                .map_elements(
                    lambda v: 1.0 if v in _BINARY_TRUE else (0.0 if v in _BINARY_FALSE else None),
                    return_dtype=pl.Float64,
                )
                .alias("value")
            )
        else:
            levels = (ordinal_levels_lookup if dtype == "ordinal" else categorical_levels_lookup)[
                name
            ]
            label_map = {normalize_level_label(label): index for index, label in enumerate(levels)}
            max_code = len(levels) - 1
            subset = subset.with_columns(
                pl.col("value")
                .map_elements(
                    lambda value, _labels=label_map, _maximum=max_code: _coerce_ordinal_code(
                        value, _labels, _maximum
                    ),
                    return_dtype=pl.Int64,
                )
                .alias("value")
            )

        # Cast value back to Utf8 for consistency with remaining data
        subset = subset.with_columns(pl.col("value").cast(pl.Utf8, strict=False))
        frames.append(subset)

    if not frames:
        return df

    remaining = df.filter(remaining_mask)
    return pl.concat([remaining, *frames], how="vertical")


def normalize_level_label(label: str) -> str:
    """Match declared level labels independently of case and surrounding whitespace."""
    return label.strip().lower()
