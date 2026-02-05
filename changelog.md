# Changelog
To help users and AI agents update.

## Pending version

- Fixed: token usage extraction no longer drops zero values (e.g. `input_tokens: 0`). Previously, the code used `x or y`, which treats 0 as falsy and skipped setting the attribute; extraction now uses explicit None checks so 0 is preserved.
- Fixed: ExperimentRunner now keys scores by `metric.id` instead of `metric.name` when sending to the server, so summaries and result tables (which look up by id) display correctly.
- Fixed: WithTracing now filters out keys starting with `_` in serialized span input/output even when values are objects (e.g. SQLAlchemy models). Previously, the default `_*` ignore pattern was only applied to nested dicts; objects were serialized via `__dict__` without stripping internal attributes like `_sa_instance_state`, so traces could contain large ORM dumps. The object serialiser now omits `_*` keys when converting any dict or object to the serialized form.
- Fixed: token usage was not extracted on spans when the traced function returned raw JSON response text (e.g. `response.text`) instead of a dict or response object; extraction now tries parsing the result as JSON when it is a string.

## v0.7.2

