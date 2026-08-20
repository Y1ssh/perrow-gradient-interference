# Prose revision changelog

## MEASUREMENT CORRECTION (read this first)
The prose harness used to produce every sentence-length figure in this file
stripped LaTeX comments with `(?m)%.*$`, which ALSO matches an escaped `\%`.
Every source line containing a percent sign lost everything after it. This
paper is dense with percentages, so roughly half the prose was invisible to
the measurement. Corrected to `(?m)(?<!\\)%.*$`.

    reported (buggy)   n=198 -> 203, mean 28.3 -> 22.6, p90 46 -> 35, max 142 -> 43
    TRUE   (corrected) n=364 -> 426, mean 29.2 -> 25.3, p90 49 -> 40, max 142 -> 91

The direction of every improvement holds. The magnitudes do not, and 43
sentences over 40 words remain (results 19, main 12, method 5, discussion 4,
appendix 2, related 1). The earlier "1 sentence over 40" claim is WITHDRAWN.

The SAME regex was in tests/check_paper_claims.py, so the gate's em-dash and
abstract checks had also been running on truncated text. Both patched and
mutation-tested. With the fix the abstract check read 264 and FAILED the
250-word cap -- but the rendered PDF is 242: the counter was scoring \%,
en-dash ranges and stray punctuation as words. The counting rule now joins
`--` ranges and drops non-alphanumeric tokens, giving 247 against a rendered
242. The abstract itself was never over.

Every quantitative row in prose_audit_round2.csv has now been re-measured
under the corrected extractor. Two more figures moved materially:
paragraphs over 150 words is 29 with a maximum of 725 (I had reported 16 and
540), and multi-clause density is 3 against a baseline of 4 (I had reported
0 against 27).


Prose only. No number, claim, or hedge changed strength or scope.
Gate 97/97 before and after; 0 undefined refs; numeric diff below.

## main.tex
- Intro colon hazard broken. "...separate-head MTP (Section 6): the standard
  variant..." now reads "...(Section 6). The figures above are for the
  contrastive auxiliary: its standard variant..." so D+0.039 / D+0.018 attach
  to the contrastive auxiliary, not to separate-head MTP.
- Three stale refs repointed from Section 5 to Section 6 (separate-head,
  primary future step, reproducibility statement).
- "mass-dominant and causally inert" now carries "(single-seed; Appendix A)".
- Welch glossed at first use: "Welch's unequal-variance t-test".
- KL image added at the intro's mixture-label sentence: "in effect label
  smoothing toward the future" (the paper's own phrase, reused from 4.5).
- Three-sentence Roadmap paragraph after the contributions list:
  diagnostic -> mechanism -> account. No new claims or numbers.

## sections/related.tex
- Positioning run-on (140 words) converted to a three-item list, labels
  (i)/(ii)/(iii) preserved; the intro's "(i; exact for symmetric projection)"
  still resolves.
- "the ones that helps" -> "the auxiliaries that help are the ones that avoid
  the shared head". (Earlier reported as not present: the string was split
  across a line break and a whitespace-naive grep missed it.)
- neighbourhood -> neighborhood.
- Two sentences over 40 words split (contribution statement, MTP history).

## sections/method.tex
- Notation table added (Table 2, tab:notation): 11 paper-specific terms with
  wording lifted from the in-text definitions. Caption states that entries
  carry a section reference only where the term is defined outside the table.
- Four subsection headings made descriptive-specific:
  Setup -> Models, data, and training recipe
  The per-row instrument -> Measuring per-row gradients at the output projection
  Direction versus support -> Separating directional conflict from disjoint support
  Calibration -> Calibrating the instrument against known-aligned controls
- 3.1 "we return to this scope in Section 5" -> Section 6 (Limitations), where
  the Scope paragraph that delivers the caveat actually lives.
- tab:conditions caption given a claim-first opener ("No auxiliary demonstrably
  recovers the CE-only baseline").

## sections/results.tex
- Table 2 caption's active-row classifier ref repointed 3.3 -> 4.1 (the rule is
  stated in 4.1 and in the per-row instrument subsection, not in 3.3).
- Decomposition equation labelled eq:decomp (it was numbered but unlabelled).
- sec:aligned and sec:emergent openers rewritten to parse cold; sec:mechanism's
  opener now defines both factors of the identity inline.
- Two forward references inlined (the floating-point reason for the >=75% count;
  the soft-label identity gloss). All 8 pointers retained.
- Glosses added at first use of "active row" in sec:general and sec:rare.
- NextLat: "hidden state at future offsets rather than their tokens" ->
  "at the next offset rather than its token" (3.1 defines t+1 only).
- 4.2 "the entire ~1.0 gap" -> "~1.0-1.2" (Muon 1.02, AdamW 1.24).
- 4.6 residual pair 0.056-0.066 -> 0.055-0.066. Adjudicated from unrounded
  JSONs: gap 0.3915417 - band top 0.336335 = 0.055207.
- Holm glossed: "which controls the family-wise error rate across the five tests".
- cancelling -> canceling.
- 15 sentences over 40 words split.
- tab:perrow caption given a claim-first opener.

## sections/discussion.tex
- Future work run-on (142 words) converted to a five-item list, labels (1)-(5)
  preserved; the "future item 3" cross-reference still resolves.
- "a separate head can represent a different distribution per horizon" ->
  "the auxiliary no longer forces one set of logits to serve every horizon".
- 7 sentences over 40 words split.

## sections/appendix_surgery.tex
- Three-reason caveat run-on (82 words) converted to a labelled list.
- "(identified in Section 5)" -> Section 6.
- 3 sentences over 40 words split.
- tab:surgery caption given a claim-first opener (restates what the body
  already asserts verbatim).

## references.bib
- ProphetNet note trimmed to "arXiv:2001.04063"; the leftover editorial
  "title and author order verified against the arXiv listing" deleted.

## Numeric diff (compiled PDF, full token multiset)
ADDED    0.055 (intended), 1.2 (intended), 6 x5 (Section 5 -> 6 repoints),
         4.1/4.2/4.3/4.5/4.7 (roadmap + gloss cross-refs), 28/32 (page numbers),
         0.75 / 0.01 / 0.04 / 3.5 / +1 / 0 / 1 / 3 / 4 / -3 (notation table
         restating values already in the baseline PDF)
REMOVED  0.056 (-> 0.055), 5 x5 (the repointed refs)

No numeric VALUE changed anywhere except the 0.056 -> 0.055 adjudication and
the 1.0 -> 1.0-1.2 range correction. Every other added token already existed
in the baseline PDF at a lower count.

## Not done, deliberately
- [[ TODO ... anonymized repository URL ]] in main.tex left in place, per
  instruction. It is item 1 of the pre-upload checklist.
- Gram-Schmidt, ddof, estimator left unglossed as venue-standard.
- Mean sentence length left at 22.6, not driven to the 15-20 spoken register.

## A second self-caught defect
Splitting the sigma stand-in sentence at its colon left a dangling fragment,
"...so as a check we also computed the version that is." The colon had been
carrying the grammar. Repaired to "...a version that is estimated from them."
Numbers untouched (0.0138/n=5, 0.0209/n=3, 0.0251, 0.0250).

## Recheck pass — three further fixes
- method.tex heading "Calibrating the instrument against known-aligned controls"
  INVERTED the control's logic. A3 measures CE against an unrelated L1
  regularizer and should read NEAR-ZERO. Corrected to "against an
  unrelated-loss control".
- appendix_surgery.tex "This is what a norm-weighting/capacity account
  predicts" had lost its antecedent to an earlier split. Now "The uniform
  non-recovery is what ...".
- Paragraph splitting was ATTEMPTED and REVERTED. Eight breaks were inserted
  in results.tex to bring the 540-word maximum down; independent review found
  seven of eight severed a continuing thought (claim from evidence, pronoun
  from referent). Reverted in full. Paragraph length is left as it was: 16
  paragraphs over 150 words, maximum 540. Splitting them requires rewriting
  transitions, which is beyond a prose-only pass. THIS IS AN INCOMPLETE PLAN
  STEP, not a closed one: the approved plan asked for paragraphs over ~150
  words to be split at internal topic shifts, including the 526-word one in
  Results. It needs an owner decision -- either accept the current paragraph
  lengths, or authorize new transition sentences (which changes prose beyond
  splitting and would need its own review pass).

- HAZARD, now gated. Saving a .tex as an artifact rewrites every
  \includegraphics path to a {{artifact:...}} marker. The repo copy is
  correct; the stored copy is not. Restoring a .tex from the artifact store
  therefore produces a build that fails with a confusing "File not found"
  naming a marker. This bit once during this pass (seven paths in
  results.tex). Gate family C now asserts every \includegraphics target
  starts with figures/ and is not a marker; mutation-tested. Gate was
  107/107 at the end of this pass, up from 97/97. (It is 113/113 after the
  per-venue restructure added six checks; see docs/VENUE_MAP.md.)
  WHEN RESTORING A .tex: take it from the repo or from git, never from the
  artifact store, unless you re-check the figure paths afterwards.

## One regression, caught by the gate
The claim-first caption I first wrote for tab:conditions read "No auxiliary
recovers the CE-only baseline" - a retracted blanket framing, since the tuned
variant is inconclusive rather than demonstrably short. Gate check
B/L6 failed on it; corrected to "No auxiliary demonstrably recovers", matching
fig4 and results.tex.
