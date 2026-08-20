# Pre-upload checklist (Zenodo preprint)

## 1. Repository URL — RESOLVED for the Zenodo build

The Zenodo build prints the real URL
(`https://github.com/Y1ssh/perrow-gradient-interference`), defined as `\repourl`
in `paper/venues/zenodo/main.tex` and consumed by the shared
`paper/sections/reproducibility.tex`.

**Still open for TMLR:** `paper/venues/tmlr/main.tex` defines `\repourl` as
`[[ TODO BEFORE SUBMISSION: anonymized repository URL ]]`. That build is
double-blind, so the value must be an anonymous mirror, not the GitHub URL. The
gate does not fail on it because the correct value is not knowable from inside
the repo.

## Gate status

Run `python tests/check_paper_claims.py`; it prints the count. At the last
check it was 139/139 across six families (A: recomputed values, B: sigma
provenance, C: structure and figure paths, D: number ownership, E: archive
metadata and README, F: measured prose ceilings). The count is deliberately not
restated as a fixed number elsewhere, because it grows.

Note: the gate reads `paper/venues/zenodo/main.aux`, a build product this
archive excludes. On a fresh copy it therefore runs one fewer check and says so
out loud, naming the compile command. Build the paper first for the full run.

The gate is mutation-tested. Across this work 13 deliberate defects were
injected one at a time and each was caught, then reverted: a wrong table value,
a retracted framing, a dangling reference, an unscoped superlative baked into a
figure title, a stale section label in the spine, the spine reverted to bare
numbers, a drifted metadata title, a non-preprint publication type, a LICENSE
holder not matching the paper's author, a license disagreeing with LICENSE, a
placeholder or checksum-invalid ORCID, a README script path that does not resolve, and a
README headline dropping the honest denominator. A gate never observed to fail
is not evidence.

## Automated (the gate owns these -- do not hand-check)

- [x] Every printed value for an owned quantity recomputed from committed JSONs
      at printed precision. No expected values are hardcoded in the gate; they
      are derived from `results/**` at run time.
- [x] sigma provenance pinned to `analysis/estimate_kl.py` (sqrt(2) x 0.018),
      so removing the constant fails the gate rather than silently orphaning it.
- [x] 12 retracted framings absent from `.tex` AND from `analysis/`,
      `measurement/`, `figures/` source (figure titles are baked into images).
- [x] No dangling `\ref`; no descending Figure/Section/Table reference pairs.
- [x] Abstract within the 250-word cap (243 rendered).
- [x] No em-dashes in prose, unicode or LaTeX `---`.
- [x] Every shipped figure has a generator.
- [x] Owned numbers single-sited across files.

## Verified this pass, by hand

- [x] Nine-layer spine written and every section audited against it.
- [x] Limitations promoted to its own numbered section (S6) with four
      subheadings; co-location, loss-only scope and single-seed sweep stated.
- [x] Control discrepancy quarantined, not explained away: threshold curve given,
      inability to resolve it stated, per-row re-run named as the fix.
- [x] Table 2 generated from `analysis/stats.py` output, all 16 cells verified.
- [x] Section 4.2 split: decoupling (L2) and mechanism (L3/L4) are now separate
      subsections; 12 cross-references repointed to `sec:mechanism`.
- [x] Rounding defect fixed: AdamW active-row opposed-norm fraction is 0.708
      from the arrays, not the 0.709 obtained by averaging rounded appendix
      values.

## Owner actions -- BLOCKING, cannot be done from here

1. **ORCID (Zenodo).** DONE, 2026-08-18. Both `CITATION.cff` and
   `.zenodo.json` carry the real identifier `0009-0003-1009-0716`; the
   checksum validates and gate family E asserts it is not a placeholder.
2. **Anonymous mirror URL (TMLR only).** `paper/venues/tmlr/main.tex` still
   defines `\repourl` as
   `[[ TODO BEFORE SUBMISSION: anonymized repository URL ]]`. The Zenodo build
   needs nothing here; it already prints the real URL.
3. **Git commit.** `.git/config` is sandbox-protected, so no commit was made
   from here. Review `git status`, set identity, commit, push. `.gitattributes`
   is in place and line endings are normalized.
4. **GPT-2 bibliography entry.** The author list is flagged
   `UNVERIFIED AUTHOR LIST` in `paper/references.bib` because it could not be
   confirmed from a reachable source. It renders correctly as an OpenAI
   technical report, which is the canonical form for this citation; confirm the
   author order against the PDF if you want the flag removed.

## Owner actions -- recommended

1. **GPU passes.** See `docs/GPU_RUN_BRIEF.md`. Two runs (~1.5 h and ~2 min)
   close spine gaps 1 and 2. Optional; the paper is coherent without them, and
   both outcomes are pre-registered in the brief.
2. **Zone.Identifier cruft.** 347 `*Zone.Identifier` files exist in the tree
   (158 of them doubly or triply nested, e.g. `README.md:Zone.Identifier:Zone.Identifier`).
   None are tracked and `.gitignore` covers them; `clean_repo.sh --apply`
   sweeps them locally if you want a tidy checkout.

## Known-and-stated, not defects

These are in the paper on purpose, in Limitations:

- Measurement/loss timing gap at 124M (spine gap 1).
- No CE-only per-row baseline at 124M (gap 2).
- Control threshold-sensitivity, unresolvable from committed data (gap 3).
- Sweep is single-seed (gap 4); untying check is one 500-step run (gap 5).
- Outcome is validation loss only, no downstream task (gap 6).
