# Project guide: weights_as_tokens_attention

Canonical, tool-neutral project guide. Claude Code imports it via `CLAUDE.md`; Codex and other AGENTS.md-aware agents read it directly.

Exploratory research code: optimize for correctness, clarity for a future reader, and reproducibility of runs. Skip production hardening and premature abstraction.

## Conventions

- Python: type hints on public functions; docstrings explain *why*.
- Every stochastic entry point takes an explicit seed.
- A run is reproducible from a single config file in the repo.


## Analysis: mechanisms only, no speculation

When interpreting numerical results or architecture, report only what is derivable from code and data. Do not hypothesize, predict, or propose next steps — those are the user's purview.

## Mechanistic reasoning — derive, don't pattern-match

Bad runs look superficially alike (within-task degradation plus forgetting); pattern-matching past diagnoses yields plausible but wrong interpretations. Before naming a mechanism, answer in order, no skipping:

1. What does the architecture do at this step? Name the tensors, operations, and gradient paths.
2. What does the data show? Values, relative magnitudes across conditions, direction of change.
3. What forward computation or gradient path produces that signal?
4. Does the mechanism explain direction AND magnitude? Contradicting even one row of data means wrong, not "mostly right."
5. What single observation would falsify it? Look for it in the existing data; if found, restart at 1.

Anti-patterns:

- **Idioms as mechanisms.** "Rich-get-richer", "recency basin" are summaries. A mechanism names tensor + operation + gradient path.
- **Parameter sharing conflated with gradient flow.** A parameter can be shared (one tensor) yet receive zero gradient on a batch because upstream gating zeros the chain rule. State which is meant.
- **Train-time and test-time conflated.** A failure observed at eval often roots in how something was built at train. State which time the mechanism operates at.
- **Analogy as conclusion.** "Same failure mode as X" without re-deriving the gradient path on the new data is hand-waving. Analogy is a hypothesis to test.

## Delegate mechanistic interpretation

Delegate to a context-fresh agent — do not reason inline — when: new benchmark/baseline results arrive (always: tabulate, then delegate interpretation); a result is unexpectedly good or bad; patterns differ across conditions or tasks; the claim is load-bearing for the next iteration; the data contradicts your first-pass model; or the user pushes back on a mechanism (re-derive from scratch, don't patch the wording). Do not delegate file-navigation or syntax questions.

The delegation prompt must supply: the observed pattern with full numbers; pointers to `wata/model.py`, `wata/training.py`, and the relevant dev plan's symbol table; ALL the data, not a curated subset. It must ask, per observation: which forward computation/gradient path produces the value, whether the candidate mechanism explains direction AND magnitude, and what observation would falsify it. It must return: one sentence of intuition, the mechanism grounded in tensors/operations/gradient terms, observations explained, and observations not yet explained (never papered over).

## Writing rules

- **Mechanistic claims ground to code** — symbol name or file:line. When unsure: claim → distinguishing test → implementation, in that order.
- **Symbol grounding.** Every new symbol is defined on first use, in the same sentence, by its meaning and its architectural referent (module, tensor, axis, or step). One symbol, one meaning per document.
- **Cross-references by named concept, never by number.** Section numbers (`§7.7`, `Section 4`) and hypothesis labels (`H1`) go stale on every reorder and hide the claim behind a lookup. Point to the named concept ("the slab-anchored head design in `class_conditional_router.md`", "the cross-channel LayerNorm degeneracy above"). A reference may stay only if it names the concept, the target is the canonical definition too load-bearing to inline, and no `§N` appears — even as a trailing parenthetical.
- **Code self-documents.** Docstrings and comments never cite dev-plan filenames, hypothesis labels, or section numbers. State what the function does, its invariants, the math it implements. Design rationale lives in dev plans; each must be readable without the other.



When the user is learning or exploring a concept, explain at the level of mechanism: name the components, the operations, and the causal chain that produce the behavior — not just the rule, the analogy, or the API surface. Analogies and summary labels may introduce a mechanism, never substitute for it.                                                                                                                                                                                                                                                                     

# Critically review design decisions                                                                                                                                                                                                                                                     

Don't assume the user-requested approach is the only or best one. Make sure the goal is clear; where a better option exists, propose it.                                                                                                                                                
Critically review the user's design decisions and offer another perspective if it hasn't been considered yet in the dialogue.
Interview the user on design choices. Challenge the user's understanding of what the right approach is, and challenge your own.                                                                                                                                                         
Planning documents always list the alternative approaches considered, with pros/cons for each, alongside the chosen approach.   