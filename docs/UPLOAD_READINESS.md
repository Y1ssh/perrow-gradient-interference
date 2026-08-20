# Upload readiness -- Zenodo preprint

Re-verified independently against the current build, not against my own notes from the
previous round. Gate 355/355; both venues 31 pp, 0 undefined references.

## Round-2 review: 16 fixed in text, 3 open by measurement, 1 accepted, 1 disclosed, 1 owner action

The reviewer's actionable set is 22 items: A1-A8, B, C2r, and D1-D12. Sixteen are fixed
in the manuscript, and I verified each by probing the current sources and the rendered
PDF rather than trusting my notes from the previous round. Two apparent failures in that sweep turned out to be my probe strings and not
defects: the surviving `3.985` is the headline degradation figure (correct there, since
the anchor passage uses matched seeds), and the second "weak residual" is the figure
caption's own sentence rather than a duplicate.

Six are not fixed, and they are not all open in the same way. Three need a measurement
this archive does not contain (D7, D9, D10). One is accepted as unverifiable (D3: arXiv
returns no venue for Gerontopoulos). One is disclosed in the paper but unexplained (D8:
the control-cliff shape, where neither a chance nor a shared-span model fits). One is
yours to confirm (D12: the repository being public). All 22 carry a row in
`docs/REVIEW_ROUND2.csv` with the status quoted here.

D8 is **not** closed. I first closed it with a shared-span null, but that null was
tested on one moment of the distribution; over all four committed bins it fails as
badly as the chance null the reviewer used. The passage now states the shape is
unexplained and names the measurement that would settle it. D8 joins the open list.

## Ready

| item | state |
|------|-------|
| no placeholders in the Zenodo build | OK -- TMLR's `[[ TODO ]]` / `XXXXXXXXXX` are correct for double-blind and are not in this build |
| repository URL | OK -- real, `\anonurl` retired |
| preprint mode, real author block | OK |
| TMLR build still anonymous | OK -- furniture separates cleanly |
| CITATION.cff | OK -- `type` is not `software`; title and author match `main.tex` |
| .zenodo.json | OK -- license agrees with LICENSE |
| LICENSE copyright | OK -- `Yash Madelwar` |
| claim gate | OK -- 355/355 on the shipped tree |
| both PDFs | OK -- 31 pp, 0 undefined |
| ORCID | OK -- supplied as `0009-0003-1009-0716` in both metadata files; supply it only if you want it on the permanent record |

D12 (repo public) is one of the two blockers below.

## Blocking, and only you can do these

Git identity is sandbox-protected here, so every commit is yours to make.

1. **117 uncommitted paths.** Includes the entire `paper/venues/` tree, `CITATION.cff`,
   `.zenodo.json`, `tests/check_paper_claims.py`, `PAPER_SPINE.md`, the three extracted
   section files, and five deletions (the retired flat `paper/main.tex` and its style
   files). `git add -A` then commit; nothing in the list is cruft, and no
   `Zone.Identifier` files remain in the status.
2. **Confirm the repo is public** before the DOI points at it.

## Open by measurement, disclosed in the paper

| id | item | why it stays open |
|----|------|-------------------|
| D3 | Gerontopoulos venue | arXiv returns no `journal_ref`; I will not assert NeurIPS unverified |
| D8 | control-cliff shape unexplained | neither a chance nor a shared-span model fits all four bins; the run committed no per-row cosines |
| D7 | CE-vs-L1 active-row control, one run | needs a GPU pass; stated in Limitations |
| D9 | active-row median trajectory unplotted | phase-A snapshots hold no per-row arrays |
| D10 | exact target mask vs norm-ratio proxy | needs a measurement pass |

The highest-value remaining measurement is still the projection-path recomputation
(R2-A2's remedy 1): it would bound the embedding-path contribution to the active-row
statistics, which is currently the paper's largest acknowledged gap.

## Provenance of the G_nce value (asked at the round-5 pre-upload check)

An earlier note carried a lower G_nce figure than the paper's. Both numbers are
real and both are in the archive; they are different run sets.

- The paper reports the five-seed mean over `results/phase_b/gnce_seed*.json`:
  4.045625, 4.037813, 4.033125, 4.002187, 4.001875, mean **4.024125**.
- The lower figure is the single seed-42 run in
  `results/phase_b_50M_repeated/gnce_seed42.json`, **3.954687**.

Both used the same 499,990,528-token budget; the repeated set drew from the
smaller 50M pool. The paper's number is the five-seed mean, which is the one
Table 4 and `analysis/stats_results.json` agree on. Nothing needs changing --
this entry exists so the question has a written answer.
