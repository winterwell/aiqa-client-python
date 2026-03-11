# Changelog
To help users and AI agents update.

## Pending version

- Added: Trace cache writes and cache reads are now recorded on spans. Token usage extraction sets `gen_ai.usage.cache_read.input_tokens` and `gen_ai.usage.cache_creation.input_tokens` when the usage payload includes cache_read_input_tokens / CacheReadInputTokens and cache_creation_input_tokens / CacheWriteInputTokens (e.g. Bedrock). Input tokens reported to the span include cached tokens per convention.
- Added: Experiment runner can filter which examples are run via `example_filter` in `run_some_examples()` and `get_examples_for_dataset()`. Pass a server-side query string (e.g. `"id:example-id"` or `"tags:tag1,tag2"`); it is sent as the `q` query parameter to the examples API.
- Added: Experiment runner can rerun experiments: `run_some_examples()` now accepts `rerun_examples` (re-run all fetched examples even if they have results) and `rerun_examples_with_missing_scores` (re-run examples that have a result but are missing a score for at least one metric).
- Fixed: token usage extraction no longer drops zero values (e.g. `input_tokens: 0`). Previously, the code used `x or y`, which treats 0 as falsy and skipped setting the attribute; extraction now uses explicit None checks so 0 is preserved.
- Fixed: ExperimentRunner now keys scores by `metric.id` instead of `metric.name` when sending to the server, so summaries and result tables (which look up by id) display correctly.
- Fixed: WithTracing now filters out keys starting with `_` in serialized span input/output even when values are objects (e.g. SQLAlchemy models). Previously, the default `_*` ignore pattern was only applied to nested dicts; objects were serialized via `__dict__` without stripping internal attributes like `_sa_instance_state`, so traces could contain large ORM dumps. The object serialiser now omits `_*` keys when converting any dict or object to the serialized form.
- Fixed: token usage was not extracted on spans when the traced function returned raw JSON response text (e.g. `response.text`) instead of a dict or response object; extraction now tries parsing the result as JSON when it is a string.

## v0.7.2

