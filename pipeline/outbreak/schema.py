"""Check documents against the Outbreak contracts in schemas/*.schema.json, without extra packages.

The schemas are standard JSON Schema (draft 2020-12), so any validator can read them. This checker
supports the subset they use: type, enum, const, required, properties, additionalProperties, items,
minItems, maxItems, minimum, maximum, exclusiveMinimum, pattern, anyOf and $ref (to "#/$defs/…" or a
sibling schema file), plus annotations. Any other keyword raises SchemaError, so a schema never
silently checks less than it says.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent / 'schemas'
ANNOTATIONS = {'$schema', '$id', '$comment', '$defs', 'title', 'description', 'examples', 'default'}
KEYWORDS = {'type', 'enum', 'const', 'required', 'properties', 'additionalProperties', 'items', 'minItems', 'maxItems',
            'minimum', 'maximum', 'exclusiveMinimum', 'pattern', 'anyOf', '$ref'}
_cache: dict[str, dict] = {}


class SchemaError(Exception):
    """The schema itself uses something this checker can't check."""


def load(name: str) -> dict:
    """A schema by name ('outbreak_geo.v1') or file name ('outbreak_geo.v1.schema.json')."""
    file = name if name.endswith('.json') else f'{name}.schema.json'
    if file not in _cache:
        _cache[file] = json.loads((SCHEMA_DIR / file).read_text(encoding='utf-8'))
    return _cache[file]


def _is(value, t: str) -> bool:
    if t == 'integer':
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    if t == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if t == 'boolean':
        return isinstance(value, bool)
    return isinstance(value, {'object': dict, 'array': list, 'string': str, 'null': type(None)}[t])


def _resolve(ref: str, root: dict):
    file, _, pointer = ref.partition('#')
    doc = load(file) if file else root
    node = doc
    for part in [p for p in pointer.split('/') if p]:
        node = node[part]
    return node, doc


def _same(a, b) -> bool:
    """JSON equality: true is not 1."""
    return type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b


def errors(value, sch: dict, root: dict, path: str = '$') -> list[str]:
    unknown = set(sch) - ANNOTATIONS - KEYWORDS
    if unknown:
        raise SchemaError(f'unsupported schema keywords {sorted(unknown)} at {path}')
    out = []
    if '$ref' in sch:
        target, doc = _resolve(sch['$ref'], root)
        out += errors(value, target, doc, path)
    if 'type' in sch:
        types = sch['type'] if isinstance(sch['type'], list) else [sch['type']]
        if not any(_is(value, t) for t in types):
            return out + [f'{path}: expected {" or ".join(types)}, got {type(value).__name__}']
    if 'const' in sch and not _same(value, sch['const']):
        out.append(f'{path}: must be {sch["const"]!r}')
    if 'enum' in sch and not any(_same(value, e) for e in sch['enum']):
        out.append(f'{path}: {value!r} is not one of {sch["enum"]}')
    if 'anyOf' in sch and all(errors(value, s, root, path) for s in sch['anyOf']):
        out.append(f'{path}: matches none of the allowed forms')
    if isinstance(value, dict):
        for key in sch.get('required', []):
            if key not in value:
                out.append(f'{path}: missing "{key}"')
        props = sch.get('properties', {})
        extra = sch.get('additionalProperties', True)
        for key, v in value.items():
            if key in props:
                out += errors(v, props[key], root, f'{path}.{key}')
            elif extra is False:
                out.append(f'{path}: unexpected "{key}"')
            elif isinstance(extra, dict):
                out += errors(v, extra, root, f'{path}.{key}')
    if isinstance(value, list):
        if 'minItems' in sch and len(value) < sch['minItems']:
            out.append(f'{path}: needs at least {sch["minItems"]} items')
        if 'maxItems' in sch and len(value) > sch['maxItems']:
            out.append(f'{path}: allows at most {sch["maxItems"]} items')
        if 'items' in sch:
            for i, v in enumerate(value):
                out += errors(v, sch['items'], root, f'{path}[{i}]')
    if _is(value, 'number'):
        if 'minimum' in sch and value < sch['minimum']:
            out.append(f'{path}: below {sch["minimum"]}')
        if 'maximum' in sch and value > sch['maximum']:
            out.append(f'{path}: above {sch["maximum"]}')
        if 'exclusiveMinimum' in sch and value <= sch['exclusiveMinimum']:
            out.append(f'{path}: must be above {sch["exclusiveMinimum"]}')
    if isinstance(value, str) and 'pattern' in sch and not re.search(sch['pattern'], value):
        out.append(f'{path}: does not match {sch["pattern"]}')
    return out


def validate(document, name: str) -> list[str]:
    """Every problem with the document, as 'path: message' strings (empty when it conforms)."""
    root = load(name)
    return errors(document, root, root)


def check(document, name: str):
    problems = validate(document, name)
    if problems:
        more = f' (and {len(problems) - 10} more)' if len(problems) > 10 else ''
        raise ValueError(f'{name}: ' + '; '.join(problems[:10]) + more)
