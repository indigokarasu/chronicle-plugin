"""
Chronicle — Configuration reference (§27).

Defaults that mirror the build spec's YAML. ChronicleCore reads a flat config
dict (typically loaded from ~/.hermes/config.yaml under `memory:`) merged over
these defaults. Nothing here touches the network; every value has a safe local
default so the system is fully functional on Hermes hooks alone (I18).
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Callable

# Env override for the embedding model (§27 embeddings.model). Deliberately wins
# over file config: eval/CI must be able to pin the deterministic offline embedder
# on a host whose config says `auto`, without editing anyone's config.yaml.
EMBED_MODEL_ENV = "CHRONICLE_EMBED_MODEL"

# Support gates selectable via retrieval.abstain_gate — §18.4
ABSTAIN_GATES = ("score", "overlap", "focus")

# Dormant flags: declared in DEFAULTS but never read by engine code. Boot warns
# once per process when a flag is actually "live" — see is_enabled below. This is
# NOT a default-value diff: two of these (hyde, expand_synonyms, decompose) are
# *already* True in DEFAULTS, so a stock boot will warn for them exactly once —
# that is the honest report, since the shipped config claims those features run
# and nothing implements them (§u1 audit). Each entry:
#   (config path, is_enabled(current_val) -> bool, reason comment)
DORMANT: list[tuple[str, Callable[[Any], bool], str]] = [
    ("retrieval.query_understanding.hyde", lambda v: bool(v),
     "Hypothetical Document Embeddings — rewrite loop not yet implemented"),
    ("retrieval.query_understanding.expand_synonyms", lambda v: bool(v),
     "Query synonym expansion — synonym detection not yet wired"),
    ("retrieval.query_understanding.decompose", lambda v: bool(v),
     "Query decomposition — multi-clause strategy not yet implemented"),
]

# Flags already warned about in this process. Config() boots repeatedly (every
# LME query, every ChronicleCore() call) — without this, the loop below would
# re-log the same warning on every single boot. Module-level (not per-Config)
# so it survives across the many short-lived Config instances in one run.
# Tests may `.clear()` this to re-arm warnings for a fresh assertion.
_DORMANT_WARNED: set = set()

# Trust ceiling C(level) — §10.3
TRUST_CEILING = {0: 0.40, 1: 0.60, 2: 0.75, 3: 0.90, 4: 1.00}
INFERENCE_TRUST = 2  # inference ceiling = C(2) = 0.75 (§9.4, §10.3)

# Confidence base per source_type — §27 confidence.base
CONFIDENCE_BASE = {
    "user_direct": 0.85,
    "session_transcript": 0.70,
    "agent_memory_write": 0.85,
    "ocas_journal": 0.70,
    "tool_output": 0.50,
    "web_retrieval": 0.40,
    "inference": 0.60,
    "rescue_extraction": 0.70,
    "delegation": 0.60,
}

DEFAULTS: dict[str, Any] = {
    "provider": "chronicle",
    "store": "sqlite",
    "db_path": "~/.hermes/commons/db/chronicle/chronicle.db",
    "git_repo": "~/.hermes/commons/db/chronicle/git",
    "git_remote": None,
    # Default = auto: detect a running local OpenAI-compatible server (base_url
    # null → LM Studio :1234, Ollama :11434, llama.cpp :8080, or
    # $CHRONICLE_EMBED_BASE_URL) and use whatever embedding model IT serves — no
    # model id is hardcoded. If none is reachable the engine runs DEGRADED (no
    # vectors written, embeds queued for retry, §24.4) — it never silently hashes.
    # Canonical recommendation (locked 2026-07-31): nomic-embed-text, served
    # locally — Ollama (:11434) or llama.cpp --embedding (:8080, ~350MB RSS, runs
    # on 1-2 vCPU self-hosted boxes). No remote-API tier: memory excerpts never
    # leave the host. Hosts too weak for nomic run DEGRADED (FTS + queued embeds).
    # Set model: hashing to force the offline embedder (eval/CI only), or a
    # specific id to pin one. $CHRONICLE_EMBED_MODEL overrides this value.
    # max_input_tokens: the model's real context window, in tokens; clamped
    # [256, 32768]. overflow: "truncate" (default, boundary-aware first slice)
    # | "chunk_mean" (split into cap-sized chunks, embed each, L2-normalize and
    # mean, then re-normalize). Both exist so an oversized excerpt is clamped
    # BEFORE it ever reaches the model — never sent over-cap, never raises for
    # length (§27; the nemotron->nomic 2048-token overflow incident).
    # task_prefixes: "auto" (default: enabled iff model name contains "nomic") |
    # true (always prepend "search_query: " to queries, "search_document: " to
    # documents) | false (never prepend). Hashing mode never prefixes. Nomic-
    # embed-text is an asymmetric model trained on task prefixes; prepending
    # improves both query and document embeddings (E1).
    # doc2query (E2, §24.4): at write time, generate the questions an item can
    # answer and embed those alongside its content vector as `query_proxy_vectors`
    # rows (kind='query_proxy' role, but stored keyed by the parent's own belief
    # kind so a hit resolves straight back to it -- see engine/doc2query.py).
    # `beliefs` covers facts/notes/episodes/procedures/references (on by default);
    # `excerpts` covers raw observed spans, off by default -- Tier-1 template
    # generation is strong for structured beliefs but only a "simple transform"
    # for free text, so the volume/quality tradeoff favors leaving it off until a
    # host model (H1/H2) can generate better excerpt questions.
    # §H2.4: `excerpts` is now FUNCTIONAL rather than inert. Its rows are keyed
    # by event_id under kind='observed'; before H2 retrieval resolved them
    # against the belief tier (_table_of_kind's "facts" default), found no row
    # and dropped every one. They now resolve through the RAW channel
    # (RetrievalEngine._observed_proxies -> retrieve_raw), where an event id
    # means something. The default stays OFF -- the flag now does what it says,
    # which is the precondition for measuring whether it is worth enabling, not
    # a reason to enable it.
    "embeddings": {"model": "auto", "dimensions": 768, "base_url": None, "api_key": None,
                   "exclude_session_prefixes": [], "max_input_tokens": 2048, "overflow": "truncate",
                   "task_prefixes": "auto",
                   "doc2query": {"beliefs": True, "excerpts": False}},
    "vector_index": {"backend": "bruteforce", "bruteforce_ceiling": 100000},

    # §15.8 (issue #5): the declarative users/agents access topology. can_read's
    # ceiling comes from here -- per-memory read_acl narrows within it but a
    # runtime grant() can never widen past it (see engine/access.py Topology).
    #   users: explicit user roster (optional; agents' user/users also populate it)
    #   agents: [{id, user|users, reads: [principal_id,...]?, sandbox: bool?}]
    #     - no `reads` declared -> falls through to default_cross_agent_read for
    #       same-user peers; cross-user is NEVER implicit regardless
    #     - `reads` declared -> authoritative ceiling for this agent (narrows
    #       same-user default; the ONLY way to grant an explicit cross-user edge)
    #     - sandbox: true -> absolute veto on inbound reads to this agent's data,
    #       even from same-user siblings under default_cross_agent_read: allow
    "principals": {
        "deployment": "shared_core",          # shared_core | per_agent_isolated
        "default_cross_agent_read": "allow",
        "users": [],
        "agents": [],
        "encryption": {"restricted_partition_keys": True},
    },

    "sources": {
        "hermes_hooks": {"enabled": True},
        "ocas_journals": {"enabled": "auto", "paths": ["~/.hermes/commons/journals/"]},
    },
    "federation": {
        "mode": "dynamic",
        "discover": ["hermes_plugins", "mcp", "skills", "config"],
        "rebind_on_change": True,
        "precedence": ["config_pin", "most_specific", "most_recent"],
        "cache_ttl": "24h",
        "pins": {},
        "provider_trust": {"default": 2},
        # Local SQLite databases this deployment declares, e.g.
        #   [{"name": "somedb", "path": "/abs/path.db", "read_only": True}]
        # Nothing about a particular database is hard-coded anywhere in engine/:
        # the name is the capability, the path is opened mode=ro, and the schema
        # is introspected. Optional per-entry "read_acl" (§15) defaults to
        # user_agents. Empty by default — a Chronicle with no declared DBs has
        # no federated channel to run (I18).
        # The federate_sweep job (§14, g4) reads the same declarations; entries
        # may additionally carry:
        #   {table, id_column, content_columns: [...], name_column?, capability?}
        # `name` is the provider id in pointers/watermarks, `content_columns` are
        # what the cached projection holds, and the optional `name_column` is the
        # only thing that can propose an identity link — for review, never applied.
        "local_dbs": [],
    },
    "outputs": {"ocas_signal_emit": {"enabled": "auto", "sink": "~/.hermes/commons/signals/"}},

    "extraction": {
        "version": "extractor-v1",
        # backend: heuristic (default — deterministic, offline, replayable) | llm.
        # llm calls an OpenAI-compatible chat endpoint per excerpt and falls back
        # to the heuristic on ANY failure, so capture never depends on a model.
        # The write path stays LLM-free unless explicitly opted in (I9, §16).
        "backend": "heuristic",
        "llm": {"base_url": None, "model": None, "api_key": None, "timeout": 30},
        "multi_hypothesis_threshold": 0.6,
        "signal_confidence_min": 0.7,
        "granularities": ["atomic", "entity", "session_summary"],
        "self_consistency": {"passes": 1, "vote": True},
        "promote_on_read": True,
        "reextract": {"mode": "eager", "read_budget_per_query": 2},
    },
    # Host-model piggyback (§H1). The host agent already runs an LLM; when this
    # is on, Chronicle may attach ONE compact enrichment request (≤400 chars) to
    # a turn the host is paying for anyway, and parse a fenced JSON block out of
    # the next reply. OFF by default and inert at defaults: with piggyback false
    # nothing is enqueued, nothing is attached, nothing is parsed, and the
    # heuristic write path is byte-for-byte unchanged (tests/test_host_model.py
    # proves this by diffing a full store dump against the pre-H1 tree).
    #   piggyback         — the master switch. False = every H1 path is dead code.
    #   max_pending       — queue cap; a 33rd enqueue oldest-expires. [1, 256]
    #   max_request_chars — rendered-request ceiling. Clamped to ≤400; config can
    #                       only make requests SMALLER, never bigger.
    #   max_reply_chars   — a fenced block above this is dropped unparsed.
    #
    # §H2 drains all three request kinds into real consumers. Two of them need
    # knobs of their own:
    #
    #   doc2query — a host's questions are merged with the Tier-1 templates
    #     under doc2query.MERGE_RULE ("host_first_template_fill": host questions
    #     take the leading slots, templates fill the rest of the <=4 budget) and
    #     written through the reducer's own delete-then-write proxy path. The
    #     merge rule is a code constant, not a config knob: it is a correctness
    #     contract with the volume bound, not a preference.
    #
    #   rerank_hints — a rerank reply arrives a TURN LATE and cannot reorder its
    #     own query, so it is persisted as query->evidence relevance hints and
    #     applied when a similar query recurs (RetrievalEngine._hint_scores).
    #     enabled      — read-side switch. On, but the store is empty unless
    #                    piggyback is on and a host answered, so "on" costs one
    #                    indexed lookup against an empty table.
    #     weight       — channel weight of a top-ranked hint, clamped [0, 2].
    #                    1.0 (= fts_weight + vector_weight) makes one leading
    #                    hint worth about a candidate topping BOTH channels.
    #                    0 is the off-switch. A hint can only re-weight a
    #                    candidate retrieval already found; it never adds one.
    #     similarity   — Jaccard floor on distinctive-token overlap for a hint
    #                    filed under a DIFFERENT query to apply at all; below it
    #                    the hint is ignored, at or above it the weight is
    #                    scaled by the overlap. Exact signature matches skip it.
    #     ttl_days     — hard expiry stamped on each row; the applied weight also
    #                    decays linearly to zero across that window.
    #     max_entries  — whole-table row cap, oldest-first eviction.
    #     max_per_query— how many beliefs ONE verdict may hint at.
    "host_model": {"piggyback": False, "max_pending": 32,
                   "max_request_chars": 400, "max_reply_chars": 4000,
                   "rerank_hints": {"enabled": True, "weight": 1.0, "similarity": 0.6,
                                    "ttl_days": 30, "max_entries": 200, "max_per_query": 8}},

    "derivation": {
        "enabled": True,
        "materialize": "high_value",
        "max_depth": 2,
        "max_fanout": 32,
        "confidence": {"aggregate": "min", "rule_factor": 0.9, "ceiling": 0.75},
        "default_status": {"user": "draft", "agent": "active"},
        "auto_disable_precision_below": 0.6,
    },
    "retrieval": {
        "fts_weight": 0.4, "vector_weight": 0.6, "rrf_k": 60, "overfetch": 4,
        "default_limit": 10, "max_limit": 50, "miss_threshold": 0.15,
        # Support gate for abstention (I8, §18.4). Only the threshold belonging to
        # the active gate is read. The dial (scripts/sweep_abstain.py, LongMemEval
        # _abs set): 0.78 → 21/30 abstained but 57/100 answerable REFUSED; 0.5 →
        # 3/30 abstained, 8/100 refused. No lexical signal separates "unanswerable"
        # from "answerable" (the _abs haystacks are on-topic, they just omit the
        # asked fact). Default is PERMISSIVE: in production the reading agent gets
        # a second abstention chance over get_context, so a refusal loop costs more
        # than a checkable wrong answer. Raise toward 0.78 where fabrication is
        # the greater harm.
        "abstain_gate": "focus", "score_threshold": 0.0148,
        "focus_coverage": 0.5, "overlap_min_tokens": 1,
        # Geometric abstention (E10): if set, abstain when the best candidate's
        # cosine distance exceeds this threshold (distance = 1 - similarity).
        # None (default) disables; when enabled, off-topic queries abstain with
        # reason "no sufficiently close memory".
        "abstain_distance": None,
        # Drafts (A16): when false (default), drafts are rendered with a [DRAFT]
        # tag in search()/get_context() but excluded from answer()'s confident
        # path — a draft is not yet verified and should not be presented as a
        # confident answer. When true, drafts flow through answer() unmodified
        # (the [DRAFT] tag still renders, so a reader can tell).
        "include_drafts": False,
        # Temporal channel (§18.6): rerank raw-tier survivors when the query names
        # an absolute date/month/year. 0 disables; clamped to [0, 2] in code.
        "temporal_boost": 0.5, "graph_weight": 0.25,
        # §L9 E8: greedy-MMR trade-off for search()'s Tier-1 candidate
        # selection -- next = argmax lam*relevance - (1-lam)*max_similarity_
        # to_already_selected. 0.7 favors relevance, spending the rest on
        # diversity so near-duplicate hits stop crowding out distinct
        # evidence. Clamped to [0, 1]. No embedder / no query vector ->
        # unaffected, today's plain score-order top-N.
        "mmr_lambda": 0.7,
        "prefetch_budget": 1200,
        "predictive_prefetch": True,
        # Embedding reranker (E3): after FTS+vector+graph fusion, the top
        # rerank_top_k candidates are re-scored by cosine(query embedding,
        # candidate embedding) blended with their fusion score --
        # blend*cosine + (1-blend)*normalized_fusion -- and re-ordered before
        # packing. The fusion term is min-max normalized to [0,1] across the
        # re-scored set FIRST: raw RRF scores live near 0.002-0.025, so an
        # un-normalized blend is a pure cosine re-sort (see
        # RetrievalEngine._rerank). A candidate with no stored vector keeps its
        # normalized fusion score, unpenalized. No query embedding (embedder
        # absent/degraded) is a complete no-op: fusion order passes through
        # exactly as before. blend=0 is also an exact no-op.
        #
        # rerank_blend: 0 IS THE OFF-SWITCH -- the only one. (A prior build
        # carried a `reranker_version` flag alongside this one, documented as
        # vestigial since no engine code read it; ladder-9 F4a removed it
        # outright rather than continue documenting a dead knob. There is no
        # "identity" mode to fall back to -- rerank_blend: 0 is the complete
        # off-switch.)
        "rerank_blend": 0.5, "rerank_top_k": 50,
        "raw_tier": {"enabled": True, "span_index": True, "session_index": True},
        "read_and_answer": {"enabled": True, "confidence_gate": 0.55,
                            "read_budget_tokens": 4000, "max_hops": 2,
                            "apply_derivation_rules": True},
        "query_understanding": {"decompose": True, "expand_synonyms": True, "hyde": True},
        # Federated query channel (§g3, federation.local_dbs). OFF by default:
        # it reads databases outside Chronicle's own store, so it is opt-in.
        "federated_channel": False,
        # E9 (§18.2): classify each query by nearest prototype centroid (a fixed
        # built-in phrase bank per question kind, embedded once per process) and
        # route get_context's evidence assembly accordingly. On by default; a
        # missing/degraded embedder always falls back to the factual/default
        # route, so this is a no-op wherever no vector channel exists.
        "query_routing": True,
        # How far a non-factual centroid must BEAT the factual one before the
        # query leaves the default route. A bare argmax over four prototype
        # centroids over-routes badly (45/60 real questions -> "aggregation",
        # costing ctx_eval@4000 3.4 points); genuine route queries win by
        # +0.22..+0.55, so 0.20 discards the ambiguous middle without touching
        # confident classifications. 0.0 restores the plain argmax.
        "query_routing_margin": 0.20,
        # PER-KIND OVERRIDE of that margin (F5). 0.20 was calibrated on the
        # built-in acceptance set, whose phrasings are near-duplicates of the
        # prototype bank itself; real questions do not clear it. Measured over
        # all 250 stratified LongMemEval questions with real nomic embeddings
        # (F3 §4a, `f3dump/routes_all.jsonl`): the shipped distribution is
        # factual 240 / aggregation 10 / preference 0 / temporal 0 -- both the
        # preference and temporal routes are dead code at 0.20.
        #
        # `preference` is nonetheless the MOST separable kind in that corpus:
        # it is the argmax on 15/15 single-session-preference questions and the
        # only type whose preference-minus-factual margin is positive for every
        # instance (min +0.025, median +0.065, max +0.138). The threshold below
        # is read off the measured sweep, not chosen by taste:
        #
        #   margin | pref route fires on | of which the 15 targets | collateral
        #     0.02 |                  38 |                   15/15 |         23
        #     0.05 |                  27 |                   12/15 |         15
        #     0.08 |                  11 |                    6/15 |          5
        #     0.20 |                   0 |                    0/15 |          0
        #
        # 0.05 covers 12 of 15 targets and BOTH instances where E12 precision
        # packing measurably misfired on a preference question (1d4e3b97 at
        # 0.0648, caf03d32 at 0.0615) at 15 collateral rather than 23. Every
        # other kind keeps `query_routing_margin`; a kind absent from this map
        # is byte-identical to before. Set {"preference": 0.20} to make F5's
        # routing change inert.
        "query_routing_margins": {"preference": 0.05},
        "query_routing_aggregation_limit": 40,
        "query_routing_aggregation_session_cap": 3,
        # (F5 removed `query_routing_preference_cap`: the E9 preference-belief
        # addendum it capped is gone. See engine/retrieval.py's note at
        # get_context's precision/pref_pack return.)
        # Answer-support verification (E11, ladder-9 issue #8): the minimum
        # cosine similarity between a host-generated answer and its best
        # matching evidence vector for RetrievalEngine.verify_answer to call
        # it `supported`. Read-only, host-LLM-mode hallucination check — never
        # consulted on the write path, so raising/lowering it changes nothing
        # about what gets stored.
        "support_threshold": 0.55,
    },
    "context": {"default_token_budget": 1500,
                "session_window": True,
                "session_window_max_sessions": 5,
                "session_window_max_events": 60,
                # L8: get_context's unconditional [DIRECTIVE] block, count-capped.
                # Was an unbounded-in-practice 20 (real stores reach ~110 active
                # norm notes); a judged reader given 12k tokens of context led by
                # ~20 directive lines abstained 30/30 even when the evidence was
                # present later in the same context (measured, L8 diagnosis).
                "max_directives": 5,
                # E12 precision packing (ladder-9 issue #8). When a query routes
                # factual AND retrieval has converged on ONE session,
                # get_context packs that session's best evidence item plus its
                # immediate neighbors into `precision_budget` tokens and stops.
                # Measured basis (stratified-250, real nomic, judged gpt-4o
                # reader): contexts that made the reader abstain at a 12k budget
                # were answered correctly when cut to ~1k of the SAME items'
                # head — abstention tracks context volume, not evidence quality.
                "precision_packing": True,
                "precision_budget": 1500,
                # THE GATE. Minimum share of the top-5 raw candidates that
                # must come from ONE session ("retrieval converged there")
                # before get_context cuts to the precision budget. 0.60 = at
                # least 3 of the 5.
                #
                # Chosen from two measured distributions, not intuition:
                #   * the six LongMemEval `single-session-user` questions this
                #     feature exists for (real nomic, s_strat250): modal share
                #     of the top-5 is 1.00, 1.00, 1.00, 0.80, 0.40, 0.40 — four
                #     of the six at or above 0.60. (Their heads are near-tied,
                #     so a re-run shuffles which questions land at 0.40 vs
                #     0.60; the count at this threshold has been 4-5 across
                #     runs, never fewer.)
                #   * the 30 factual-route queries of the ctx_eval corpus
                #     (hashing): 0.80 once, 0.60 nine times, 0.40 or 0.20 for
                #     the other twenty — questions whose evidence really is
                #     spread across sessions.
                # At 0.60 (with the leader-agrees rule below) the gate fires on
                # 4 of the 6 and on 5 of the 30, and ctx_eval@1500 goes UP,
                # 75.9% -> 77.6%, with @4000 and @12000 unchanged: 165 of the
                # corpus's 180 contexts come back byte-identical to the pre-E12
                # tree and the 15 that change are the firing ones. 0.80 would
                # be tighter still but fires on only 3 of the 6 and misses the
                # case this exists for; below 0.60 the gate is firing on heads
                # that are mostly NOT one session, which is the definition of
                # the ambiguity it must refuse (at 0.60 WITHOUT the
                # leader-agrees rule it fires on 10 of the 30 and ctx_eval
                # drops to 72.4/79.3/82.8).
                "precision_concentration": 0.60,
                # SECONDARY tightening dial: extra lead ((s0 - s1) / s0) the
                # leading candidate must hold over the runner-up, on top of
                # concentration. Default 0.0 — no extra requirement — and that
                # default is itself a measurement, not laziness. On those same
                # six questions the leader's relative margin is 0.004-0.155
                # (0.010, 0.013, 0.045, 0.051, 0.150 in the run that set these
                # numbers), while the crowded ctx_eval queries run up to
                # 0.458: margin does not separate the two
                # populations in either direction. An earlier build of this
                # feature gated on margin ALONE and was either inert (0.50 —
                # fired on none of the six) or destructive (0.30 -> -1.7
                # ctx_eval points, 0.20 -> -8.6, 0.10 -> -12.1). Raise it to
                # tighten the gate on a corpus where it does separate; 1.0
                # makes the feature inert.
                "precision_margin": 0.0,
                # F5 preference packing. On the E9 `preference` route,
                # get_context packs the LEADING message of each ranked excerpt
                # (the user's own turn) across every session first, defers the
                # assistant halves, and cuts to `preference_budget` tokens.
                #
                # Measured basis (F3, stratified-250, real nomic): 73-91% of a
                # packed preference context is assistant prose, and only the
                # user's half of an excerpt can carry a preference. Across the
                # six probed gold sessions the whole of what the user said
                # about themselves is 870-1 419 chars (5-8% of the session's
                # text), so ALL of it fits where today 2 of 7-9 excerpts fit
                # whole. Re-measured after the change: the packed contexts are
                # 94-95% user text.
                "preference_packing": True,
                # 3 000 rather than the 1 500 precision packing uses, and the
                # difference is arithmetic, not taste. The F3 design proposed
                # 1 500 on the estimate that a 6 000-char budget holds "25-30
                # user heads ... across all ~10 sessions the raw fill would
                # have touched", i.e. ~2.5 heads per session. Counted on the
                # six probed haystacks the median session has 6 user turns
                # totalling 823-1 310 chars, so 6 000 chars holds 4.6-7.3
                # sessions COMPLETE, not 10 -- and the ones past the cut get a
                # header and nothing else.
                #
                # That matters more than it looks, because `retrieve_raw`'s
                # ordering of near-tied candidates is not stable run to run
                # (measured on v560 as well as here: two sequential probes of
                # the same instance put the answer session at header rank 4 and
                # then 5). At 6 000 chars that reordering decides whether the
                # answer session gets 7 of its user turns or none of them --
                # measured, both outcomes, same instance. At 12 000 it holds
                # 9.2-14.6 sessions complete, which covers the whole group list
                # the raw fill produces, so the ordering stops deciding.
                # Still a quarter of the 48 000 chars a 12k-token caller would
                # otherwise get. Raise it further only with a judged reader run.
                "preference_budget": 3000,
                "weights": {"relevance": 0.4, "recency": 0.25, "salience": 0.25, "pinned": 0.1}},
    "capture": {"max_excerpt_chars": 4000,          # per-chunk cap, clamped [500, 16000] (§12.1)
                "sync_turn": {"mode": "observe_only"},
                "precompress": {"budget_ms": 400},
                "agent_memory_write": {"salience": "high", "confidence_discount": 0}},
    "reaper": {"enabled": True, "schedule": "*/5 * * * *", "idle_threshold": "20m",
               "reap_threshold": "45m", "startup_recovery": True},
    "confidence": {"base": CONFIDENCE_BASE, "trust_ceiling": TRUST_CEILING},
    "calibration": {"min_obs": 50, "refit_every": 100},
    "forgetting": {"confirm_critical": True,
                   "raw_retention": {"keep_verbatim_days": 365, "then": "gist"}},
    "salience": {"decay_multipliers": {"pinned": 0, "high": 0.25, "normal": 1.0, "incidental": 4.0}},
    "representation": {"canonicalize": {"enabled": True, "similarity_threshold": 0.8,
                                        "auto_apply_domain": ["agent"]}},
    "behavior_change": {"risk_tier_default": "low", "high_risk_requires_review": True},
    "epistemic": {"redundant_window": "48h", "forgot_window": "30d"},
    "domains": {
        "user": {"auto_decay": False, "contradiction_policy": "flag_for_review"},
        "agent": {"auto_decay": True, "decay_days": 90, "contradiction_policy": "newer_wins"},
        "general": {"auto_decay": True, "decay_days": 30, "contradiction_policy": "refetch"},
    },
    "curation": {"mode": "event_driven", "sweep_schedule": "0 * * * *",
                 "identity_threshold": 0.85, "consolidate_min_facts": 50,
                 # E5 near-duplicate merge floor: at or above this cosine, with
                 # the same subject, a write MERGES into the existing item
                 # instead of storing a second copy.
                 "dup_similarity": 0.95,
                 # Ladder 9 E4 (§issue-8): cosine floor for "this write looks like
                 # an update of that belief" (nearest same-subject neighbor, or the
                 # global-store neighbor when no same-subject candidate exists).
                 # High on purpose -- a false positive links two unrelated facts
                 # into a misleading "history"; missing a real update only means a
                 # reader sees two separate facts instead of a dated chain, which
                 # is exactly today's behavior. Strictly BELOW dup_similarity:
                 # the two form a ladder over the same cosine (0.82 supersede
                 # candidate, 0.95 merge), so a near-identical re-assertion is
                 # absorbed by E5 before E4 ever calls it a supersession.
                 "supersede_similarity": 0.82,
                 # §E6: neighbor-cosine floor between consecutive observed-event
                 # embeddings within one session. Absolute, not a rolling
                 # baseline -- the simplest rule that is still correct, and it
                 # mirrors identity.split_below's fixed-floor shape rather than
                 # tracking a moving average that a slow topic drift could ride
                 # under. Below this, session_summarize opens a new episode.
                 # Hashing-mode vectors for genuinely unrelated excerpts land
                 # near-orthogonal (~0.0-0.15 cosine, no shared vocabulary);
                 # same-topic paraphrases share tokens and sit well above it.
                 # 0.35 is a conservative middle that a real sentence embedder
                 # (nomic et al.) also respects -- unrelated topics score below
                 # it, ordinary within-topic variation does not.
                 "topic_shift_threshold": 0.35},

    # Identity evidence (§E7, issue #8). Similarity produces CANDIDATES only —
    # nothing here ever merges or splits an entity; identity is adjudicated,
    # never inferred. split_below: a new mention whose cosine to its entity's
    # running centroid falls below this is queued as a possible split (one id
    # carrying two subjects). merge_above: two entity centroids above this are
    # queued as a possible merge (two ids carrying one subject).
    # merge_scan_limit BOUNDS the pairwise check — the entity just written is
    # compared against at most this many most-recently-updated centroids (the
    # working set), never against every entity in the store. Inert without an
    # embedder: no vector reaches the check, so no state and no candidates.
    "identity": {"enabled": True, "split_below": 0.30, "merge_above": 0.90,
                 "merge_scan_limit": 50},
    "consolidation": {"enable_parametric": False},
    "health": {"schedule": "0 4 * * *",
               "ghost_fact": {"confidence_min": 0.8, "age_days": 14},
               "consistency_sweep": {"enabled": True}, "self_heal": {"tier1_auto": True}},
    "learning": {"max_active_deltas": 8, "max_delta_magnitude": 0.15,
                 "mutable_dimensions": ["rrf_weights", "context_weights", "decay_multipliers",
                                        "reranker", "calibration", "read_confidence_gate",
                                        "derivation_rule_enable"]},
    "consent": {"default_scope": ["*"], "enforce_purpose": True},
    "security": {"encrypt_at_rest": True},
    "git": {"enabled": True, "max_commit_rows": 1000, "max_lag_minutes": 30,
            "snapshot_interval": "0 * * * *"},
    "tier_triggers": {"write_lock_contention": 0.20, "vector_count": 5000000, "sqlite_max_gb": 20},

    # Context-engine slot (§27 context:)
    "context_engine": {
        "engine": "chronicle",
        "keep_weights": {"relevance": 0.35, "recency": 0.20, "salience": 0.20,
                         "criticality": 0.20, "redundancy_vs_store": 0.30},
        "never_evict": "directives",
        "should_compress": {"on_memory_pressure": True, "on_focus_shift": True},
        # Two-watermark hysteresis (§R2): HIGH is when should_compress() decides
        # a pass is due (fraction of the model's context window); LOW is the
        # target fraction compress() evicts DOWN TO. Using two different points
        # instead of one is the hysteresis -- a single cutoff either re-triggers
        # a pass on every call sitting right at the edge, or (the previous bug)
        # doesn't bound the compressed size at all. Both are fractions of
        # context_length, so the actual token count scales with whatever model
        # is configured instead of a number picked for no particular window.
        "high_watermark_percent": 0.75,
        "low_watermark_percent": 0.55,
        "reinject": {"enabled": True},
        "standalone_fallback": "heuristic",
        # §R7: cap (tokens) on the rolling, no-model checkpoint digest of
        # everything compression has folded out this session; oldest lines
        # drop first once a refresh would push it over this.
        "checkpoint_digest_max_tokens": 300,
    },
}


def check_abstain_gate(name: str) -> str:
    """Reject an unknown retrieval.abstain_gate loudly (§18.4).

    A typo here would silently disable abstention, which is the one failure the
    gate exists to prevent — so it is a hard error, not a fallback.
    """
    if name not in ABSTAIN_GATES:
        raise ValueError("retrieval.abstain_gate must be one of {} (got {!r})".format(", ".join(ABSTAIN_GATES), name))
    return name


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    """Thin dotted-path accessor over a merged config dict."""

    def __init__(self, overrides: dict[str, Any] | None = None):
        self._d = _deep_merge(DEFAULTS, overrides or {})
        env_model = (os.environ.get(EMBED_MODEL_ENV) or "").strip()
        if env_model:
            if not isinstance(self._d.get("embeddings"), dict):
                self._d["embeddings"] = {}
            self._d["embeddings"]["model"] = env_model
        check_abstain_gate(self._d["retrieval"].get("abstain_gate"))
        self._check_dormant_flags()

    def _check_dormant_flags(self):
        """Warn once per process for each DORMANT flag that is currently live.

        `is_enabled` decides liveness directly from the value, not from a diff
        against some remembered default — that was the sonnet-1 bug: comparing
        against a stored default_val of True made `hyde=False` trip `!= True`
        and warn "is enabled" anyway (§u1 review triage).
        """
        log = logging.getLogger("chronicle.config")
        for path, is_enabled, reason in DORMANT:
            if path in _DORMANT_WARNED:
                continue
            if is_enabled(self.get(path)):
                log.warning("dormant config flag %s is enabled: %s", path, reason)
                _DORMANT_WARNED.add(path)

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._d
