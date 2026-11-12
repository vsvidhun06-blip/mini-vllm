# Historical provenance record

SOURCE_PROVENANCE_2026-09-18.json records the earlier local source/archive state.
It uses schema source_provenance/1 and does not describe the current code.
It is preserved unchanged for traceability, outside the active runtime path.

For a new experiment, build its source archive and run the current
scripts/eval/repair/source_provenance.py with --archive pointing to that archive.
The current validator requires archive-member provenance; this historical record
must not be copied into docs/eval/SOURCE_PROVENANCE.json to authorize a new run.
