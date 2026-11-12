"""Explicit, validated parameter proposals; never infer edits from answer prose."""
import json
import re

from hybrid_settings import validate

EVIDENCE = ('measured simulation result', 'calculated model reference', 'untested trial')
BLOCK = re.compile(r'^```panel-settings[ \t]*\n(.*?)^```[ \t]*$', re.MULTILINE | re.DOTALL)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate proposal key: '+key)
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError('Nonfinite number in proposal: '+value)


def validate_changes(changes, baseline, mode):
    if not isinstance(changes, dict) or not changes:
        raise ValueError('The proposal must contain a nonempty changes object.')
    if any(not isinstance(value, (int, float, str, list)) for value in changes.values()):
        raise ValueError('Unsupported parameter value type.')
    baseline = validate(baseline)
    candidate = validate({**baseline, **changes})
    changes = {key: candidate[key] for key in changes if candidate[key] != baseline[key]}
    if not changes:
        raise ValueError('The proposed values already match the panel.')
    # Import lazily: the CLI context adapter also imports model calculations.
    from codex_assistant import tuning_context
    context = tuning_context(candidate, mode)
    unused = set(changes).intersection(context['inactive_parameters'])
    if unused:
        raise ValueError('These controls are inactive in this mode: '+', '.join(sorted(unused)))
    if context['error']:
        raise ValueError(context['error'])
    return changes


def parse_proposal(answer, baseline, mode):
    """Return display text, an optional bound proposal, and an optional error."""
    matches = list(BLOCK.finditer(answer))
    if not matches:
        error = 'Incomplete suggested-values block; ask for a new proposal.' if '```panel-settings' in answer else None
        return answer, None, error
    display = BLOCK.sub('', answer).strip()
    try:
        if len(matches) != 1:
            raise ValueError('Only one suggested-values block is allowed per answer.')
        payload = json.loads(matches[0].group(1), object_pairs_hook=_unique_object,
                             parse_constant=_invalid_number)
        if not isinstance(payload, dict) or set(payload) != {'changes', 'evidence', 'reason'}:
            raise ValueError('Suggested values need exactly changes, evidence and reason.')
        if payload['evidence'] not in EVIDENCE:
            raise ValueError('Unknown evidence category.')
        if not isinstance(payload['reason'], str) or not payload['reason'].strip():
            raise ValueError('Suggested values need a reason.')
        changes = validate_changes(payload['changes'], baseline, mode)
        return display, dict(changes=changes, evidence=payload['evidence'], reason=payload['reason'],
                             base_settings=validate(baseline), mode=mode), None
    except (ValueError, TypeError) as exc:
        return display, None, str(exc)


def prepare_application(proposal, current, mode):
    if mode != proposal['mode'] or validate(current) != proposal['base_settings']:
        raise ValueError('Settings or output mode changed since this answer. Ask for a fresh proposal.')
    return validate_changes(proposal['changes'], current, mode)
