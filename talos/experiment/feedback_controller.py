"""Collect feedback and audit parameter-queue interventions."""

import json
import logging
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from talos.experiment.msq import MSQ
from talos.experiment.param_search.search_strategy import SearchStrategy
from talos.experiment.reducer.filter_types import FILTER_BUILDERS
from talos.experiment.reducer.pruning_strategy import ACTION_SUGGEST, PruningStrategy
from talos.experiment.serialization import dumps

logger = logging.getLogger(__name__)

_SOURCE_ERRORS = (
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    OSError,
    json.JSONDecodeError,
)


def _apply_set_filter(msq: MSQ, intervention: dict[str, Any]) -> None:

    '''Validate and apply a set_filter intervention to the MSQ.'''

    filter_type = intervention.get('filter_type')
    filter_params = intervention.get('filter_params')

    if filter_type not in FILTER_BUILDERS:
        supported = ', '.join(sorted(FILTER_BUILDERS.keys()))
        raise ValueError(
            f"FeedbackController Unknown filter_type '{filter_type}'. Supported types: {supported}"
        )

    if not isinstance(filter_params, dict):
        raise ValueError(
            f"FeedbackController filter_params must be a dict, got {type(filter_params).__name__}"
        )
    filter_params = cast(dict[str, Any], filter_params)

    try:
        condition = FILTER_BUILDERS[filter_type](filter_params)
    except KeyError as e:
        raise ValueError(
            f"FeedbackController filter_params for '{filter_type}' missing required key: {e}"
        ) from e

    msq.set_filter(
        intervention['key'],
        condition,
        filter_type=filter_type,
        filter_params=filter_params,
    )


class FeedbackController:

    '''Orchestrate feedback from all enabled sources and dispatch interventions to the MSQ.

    Sources are checked in order: pruning strategies, intra callback,
    intervention file. Each source is isolated — failures in one
    source do not block others.
    '''

    def __init__(self,
                 *,
                 feedback_interval: int = 100,
                 pruning_strategies: list[PruningStrategy] | None = None,
                 intra_callback: Callable[[Any, MSQ], None] | None = None,
                 intervention_path: str | Path | None = None,
                 audit_log_path: str | Path | None = None) -> None:

        '''
        Initialize the FeedbackController.

        Args:
            feedback_interval (int): Trigger feedback every N rounds
            pruning_strategies (list[PruningStrategy] | None): Reducers that
                analyze logs and return intervention dicts
            intra_callback (Callable | None): Python callback receiving
                (log, msq) with direct MSQ access
            intervention_path (str | Path | None): Path to JSON file with
                intervention dicts, polled by modification time
            audit_log_path (str | Path | None): Path to JSONL audit trail file

        '''

        super().__init__()

        if feedback_interval < 1:
            raise ValueError(
                f"FeedbackController feedback_interval must be >= 1, got {feedback_interval}"
            )
        self._feedback_interval = feedback_interval
        self._pruning_strategies = pruning_strategies or []
        self._intra_callback = intra_callback
        self._intervention_path = Path(intervention_path) if intervention_path else None
        self._audit_log_path = Path(audit_log_path) if audit_log_path else None
        self._intervention_last_mtime: float = 0.0
        self._trigger_count: int = 0

    def should_trigger(self, current_round: int) -> bool:

        '''Check whether feedback should fire at this round.'''

        return current_round > 0 and current_round % self._feedback_interval == 0

    def trigger(self,
                log: Any,
                msq: MSQ,
                strategy: SearchStrategy,
                current_round: int) -> list[dict[str, Any]]:

        '''
        Execute a feedback cycle across all enabled sources.

        Collects interventions from each source, applies them to the MSQ,
        updates the search strategy, and writes an audit trail entry.
        Suggestion interventions (action='suggest') are logged in the
        audit trail but not dispatched or returned.

        Args:
            log (pl.DataFrame): Polars experiment log for analysis
            msq (MSQ): Mutable search queue to apply interventions to
            strategy (SearchStrategy): Search strategy to notify of changes
            current_round (int): Current experiment round number

        Returns:
            list[dict[str, Any]]: Applied interventions (excludes suggestions)

        '''

        all_interventions: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        sources = [
            ('pruning_strategies', lambda: self._collect_from_pruning(log, msq), True),
            ('intra_callback', lambda: self._collect_from_intra(log, msq), False),
            ('intervention_file', lambda: self._collect_from_file(), True),
        ]

        for name, collector, needs_dispatch in sources:
            self._run_source(
                name, collector, needs_dispatch,
                msq, all_interventions, errors,
            )

        applied = [i for i in all_interventions if i.get('action') != ACTION_SUGGEST]
        suggestions = [i for i in all_interventions if i.get('action') == ACTION_SUGGEST]

        strategy.update_from_feedback(log, applied)

        self._write_audit_entry(
            current_round, applied, errors, msq, suggestions=suggestions,
        )
        self._trigger_count += 1

        return applied

    def _run_source(self,
                    name: str,
                    collector: Callable[[], list[dict[str, Any]]],
                    needs_dispatch: bool,
                    msq: MSQ,
                    all_interventions: list[dict[str, Any]],
                    errors: list[dict[str, Any]]) -> None:

        '''
        Execute a single feedback source with error isolation.

        Args:
            name (str): Source name for logging and audit
            collector (Callable): Function returning intervention dicts
            needs_dispatch (bool): Whether to apply interventions via dispatch.
                False for intra callback which applies directly
            msq (MSQ): Mutable search queue for dispatching
            all_interventions (list): Accumulator for all interventions
            errors (list): Accumulator for errors

        '''

        try:
            interventions = collector()
            if needs_dispatch:
                for intervention in interventions:
                    if intervention.get('action') != ACTION_SUGGEST:
                        self._apply_intervention(msq, intervention)
            all_interventions.extend(interventions)
        except _SOURCE_ERRORS as e:
            logger.warning('%s failed: %s', name, e)
            errors.append({
                'source': name,
                'error': str(e),
                'traceback': traceback.format_exc(),
            })

    def _collect_from_pruning(self,
                              log: Any,
                              msq: MSQ) -> list[dict[str, Any]]:

        '''
        Collect interventions from all active pruning strategies.

        Each strategy is called in isolation. A failure in one strategy
        is logged but does not prevent others from executing.

        '''

        interventions: list[dict[str, Any]] = []

        for ps in self._pruning_strategies:
            if not ps.active:
                continue
            try:
                result = ps.analyze_and_intervene(log, msq)
                for item in result:
                    if 'source' not in item:
                        item['source'] = ps.__class__.__name__
                    interventions.append(item)
            except _SOURCE_ERRORS as e:
                logger.warning(
                    'PruningStrategy %s failed: %s',
                    ps.__class__.__name__, e,
                )

        return interventions

    def _collect_from_intra(self,
                            log: Any,
                            msq: MSQ) -> list[dict[str, Any]]:

        '''
        Collect interventions from the intra-experiment callback.

        The callback has direct MSQ access and applies interventions
        itself. We track what changed by comparing MSQ intervention_log
        length before and after the call. If the callback crashes
        mid-way, we still capture whatever it applied before failing.

        '''

        if self._intra_callback is None:
            return []

        log_len_before = len(msq.intervention_log)
        try:
            self._intra_callback(log, msq)
        except _SOURCE_ERRORS as e:
            logger.warning('Intra callback failed: %s', e)

        new_entries = msq.intervention_log[log_len_before:]
        return [{'source': 'intra_callback', **entry} for entry in new_entries]

    def _collect_from_file(self) -> list[dict[str, Any]]:

        '''
        Poll the intervention file and return new interventions if changed.

        Checks file modification time. If unchanged since last poll,
        returns empty list. If changed, reads and parses the JSON array.

        '''

        if self._intervention_path is None:
            return []

        if not self._intervention_path.exists():
            return []

        current_mtime = self._intervention_path.stat().st_mtime
        if current_mtime <= self._intervention_last_mtime:
            return []

        self._intervention_last_mtime = current_mtime
        content = self._intervention_path.read_text()
        interventions = json.loads(content)

        if not isinstance(interventions, list):
            raise ValueError('FeedbackController Intervention file must contain a JSON array')
        interventions = cast(list[Any], interventions)

        for item in interventions:
            if 'source' not in item:
                item['source'] = 'intervention_file'

        return interventions

    @staticmethod
    def _apply_intervention(msq: MSQ,
                            intervention: dict[str, Any]) -> None:

        '''
        Dispatch a single intervention dict to the appropriate MSQ method.

        Args:
            msq (MSQ): The mutable search queue to modify
            intervention (dict[str, Any]): Intervention dict with 'op' key

        '''

        param = intervention.get('param')
        if param is not None:
            for name in ('value', 'threshold', 'lower', 'upper'):
                if name in intervention:
                    intervention[name] = msq.resolve_log_value(param, intervention[name])
            if 'values' in intervention:
                intervention['values'] = [msq.resolve_log_value(param, value) for value in intervention['values']]
        if 'combo' in intervention:
            intervention['combo'] = {key: msq.resolve_log_value(key, value)
                                     for key, value in intervention['combo'].items()}
        if 'filter_params' in intervention:
            settings = dict(intervention['filter_params'])
            param = settings.get('param')
            if param is not None:
                for name in ('value', 'threshold', 'lower', 'upper'):
                    if name in settings:
                        settings[name] = msq.resolve_log_value(param, settings[name])
                if 'values' in settings:
                    settings['values'] = [msq.resolve_log_value(param, value) for value in settings['values']]
            intervention['filter_params'] = settings

        dispatch: dict[str, Callable[[dict[str, Any]], Any]] = {
            'remove_is': lambda i: msq.remove_is(i['param'], i['value']),
            'remove_ge': lambda i: msq.remove_ge(i['param'], i['threshold']),
            'remove_le': lambda i: msq.remove_le(i['param'], i['threshold']),
            'keep_is': lambda i: msq.keep_is(i['param'], i['value']),
            'keep_between': lambda i: msq.keep_between(i['param'], i['lower'], i['upper']),
            'inject': lambda i: msq.inject(i['combo'], prioritize=i.get('prioritize', False)),
            'inject_value': lambda i: msq.inject_value(i['param'], i['value']),
            'trim': lambda i: msq.trim(i['target_count']),
            'set_filter': lambda i: _apply_set_filter(msq, i),
            'clear_filter': lambda i: msq.clear_filter(i['key']),
        }

        op = intervention['op']
        handler = dispatch.get(op)
        if handler is None:
            raise ValueError(f"FeedbackController Unknown intervention op: '{op}'")
        _ = handler(intervention)

    def _write_audit_entry(self,
                           current_round: int,
                           interventions: list[dict[str, Any]],
                           errors: list[dict[str, Any]],
                           msq: MSQ,
                           *,
                           suggestions: list[dict[str, Any]] | None = None) -> None:

        '''Append a single JSONL entry to the audit log file.'''

        if self._audit_log_path is None:
            return

        entry = {
            'round': current_round,
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'trigger_number': self._trigger_count,
            'interventions': interventions,
            'suggestions': suggestions or [],
            'errors': errors,
            'msq_state_after': {
                'remaining_count': msq.remaining_count(),
                'yielded_count': msq.yielded_count,
                'priority_queue_size': msq.priority_queue_size,
            },
        }

        try:
            with self._audit_log_path.open('a') as f:
                _ = f.write(dumps(entry) + '\n')
        except OSError as e:
            logger.warning('Failed to write audit log: %s', e)

    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'trigger_count': self._trigger_count,
            'intervention_last_mtime': self._intervention_last_mtime,
        }

    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        required = ('trigger_count', 'intervention_last_mtime')
        for key in required:
            if key not in state:
                raise ValueError(f"Invalid FeedbackController state: missing required key '{key}'.")

        self._trigger_count = state['trigger_count']
        self._intervention_last_mtime = state['intervention_last_mtime']
