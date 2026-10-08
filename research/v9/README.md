# Contextual source binding — protocol and inventory only

V9 studies whether published financial evidence can be tied to the right issuer, series, contract or maturity with its source, dates, units and accounting context intact. The actual manifest schema forbids the optional ownership fields our earlier V8 assumption expected; document metadata is retrieval provenance, not a guaranteed ownership contract. Rejected V8 and V7 are unchanged.

`SOURCE_INVENTORY_BOUNDS.json` pins all 11 units, 78 entities and 69 document entries, including same-digest metadata, a fixed 4096-character source prefix and every pipe-run first line. Thirty-eight entities have exact metadata clues. The inventory has not run a contextual binder, so guaranteed context-safe informative coverage is zero and its only general upper bound is 78.

`PROTOCOL.json` fixes family-specific provenance requirements, exact source blocks, conflict refusal, full roster accounting and adversarial cases. Shared monetary-policy context is reported separately from maturity-specific facts. Missing issuer, period, units, category or table headers must remain a source hole; source IDs, fuzzy names and hardcoded entity aliases cannot rescue it. There is no coverage threshold tuned to obtain success.

Root and independent precode review are required before any binder implementation. No numeric features, predictions, model calls or submissions exist here. Eventual source feasibility alone cannot establish financial quality.

Independent review verified the inventory and accepted its zero-to-78 planning bounds. A binder is **not authorized**: executable financial-table, source-introduction/local-issuer, family-qualifier, code-boundary and context-packing predicates must be frozen first. The inventory includes inline/navigation pipe fragments and an oversized flattened filing; those are refusal cases, not financial tables. See `INDEPENDENT_PRECODE_REVIEW_TRACK3.md`.
