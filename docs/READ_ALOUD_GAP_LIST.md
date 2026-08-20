# What is missing, measured against the Explainer Engine

Read-only audit of the manuscript; the gate and the review register were updated to
close an accounting error this audit found. Gate 355/355; both venues 31 pp, 0 undefined references.

Every row below was measured on the current sources or the rendered PDF, not recalled.
Every number in this document was re-measured against the sources or the rendered PDF
after it was written, and every per-section breakdown checked to sum to its stated
total: the sentence and paragraph statistics and their per-file breakdowns, the
never-write and bare-"this" counts, the `\ref` target tallies, the `\item` and emphasis
counts, the abstract's rendered word count, the Zenodo description length and its
jargon counts, the page counts, the gate total, the uncommitted-path count, and the
review-item accounting. That pass caught three errors: a bare-"this" total of 10 over a
breakdown summing to 9, cross-reference targets characterized rather than classified,
an item accounting of "21 items, 15 fixed" when the register holds 22 rows with 16
FIXED, a headline that labelled six unlike items "5 open by measurement" when the
register gives three statuses, an emphasis denominator quoted from a rendered-section
figure this session could not reproduce (Results is 5,539 words of source prose), and
two documents shipping 111 and 112 for the same blocker. Gate families C10 and C11 now
derive the headline from the register and require the shared counts to agree.
Priority A blocks upload. B and C are quality gaps that do not block a preprint. D is
mechanical.

## Can you upload?

**Yes, once the two A-priority items that are yours are done.** Nothing in the manuscript
is wrong. The paper's own claim gate passes at 355/355, every number traces to a committed
JSON, and 16 of the external reviewer's 22 actionable items are fixed in text, with
the other 6 disclosed (3 open by measurement, 1 accepted as unverifiable, 1 disclosed but unexplained, 1 owner action). What is missing is not correctness -- it is reach: the paper is written for one
audience, and the Engine asks for a text that works across a span.

## A -- blocks upload

| item | measured | what to do |
|---|---|---|
| 117 uncommitted paths | `git status --porcelain` returns 117 lines; every `paper/venues/` file is still untracked, along with `CITATION.cff`, `.zenodo.json`, `tests/check_paper_claims.py`, `PAPER_SPINE.md` | `git add -A` and commit. Git identity is sandbox-protected here, so this is yours |
| repo public | not verifiable from inside the sandbox | confirm before the DOI points at it |
| Zenodo landing description | 1,414 chars, the paper abstract verbatim: "cosine" 4x, "gradient" 4x, MTP / KL / Muon / AdamW unglossed, opening on "task gradients" and "auxiliary-loss training" | this is the only text every visitor reads before deciding to open the PDF, and it is the single highest-leverage fix on this list. It is a metadata field, so nothing in the paper changes |

## B -- reach gaps, do not block a preprint

| item | measured | why the Engine flags it |
|---|---|---|
| no plain-language on-ramp | abstract is 256 rendered words, entirely technical; the introduction opens on the instrument rather than the question | Q asks every section for a jargon-free plain answer -- "someone who stops here should be correct, just less equipped" |
| 31 paragraphs over 150 words, 17 over 250, longest 639 | Results holds 639 / 456 / 451 / 431; Discussion 422 | G1: a paragraph is one claim. Two earlier attempts to split these were reverted after independent review found 7 of 8 breaks severed a continuing thought, so this needs sentence-level work, not bulk splitting |
| 54 sentences over 40 words, p90 42, longest 83 | intro 10, results 24, method 8, discussion 5, related 3, appendix 4 | M replaces the cap with "one idea, spine early" -- each needs reading aloud individually |

## C -- teaching moves absent

| item | measured | Engine rule |
|---|---|---|
| term debt across sections | "active row" appears with no local gloss in intro, related and appendix_surgery; "norm-profile cosine" in abstract and appendix; "opposed-norm fraction" in appendix | Law 2 -- a reader landing on Appendix A by search meets three undefined terms |
| the reader's wrong model is never named as theirs | "conflict" appears 29 times, but no "most readers assume" / "reasonable guess" / "tempting" construction exists anywhere | J2 -- correct information layers on top of a wrong model unless the wrong one is named and dismantled |
| one analogy, and it is expert-only | "analog" once in the intro; label smoothing 3x; no everyday image anywhere | I6 -- the source must be more familiar than the target for everyone in the span |

## D -- mechanical

| item | measured |
|---|---|
| never-write hits | "just" 2x (results), "trivially" 1x (method) |
| bare "this" as sentence subject | 9 sites: intro 1, related 1, method 2, results 3, discussion 2 |

## Verified as passing

- **R never-write list**: clean except the 3 hits above. No "as discussed above", no "it is important to note", no "delve/tapestry/landscape/realm", no "obviously", no whole-sentence bolding.
- **Law 2 back-references**: zero "as we saw earlier" / "recall from" constructions. 139 cross-references, classified by target prefix: 81 `sec:`, 29 `fig:`, 14 `app:`, 10 `tab:`, 5 `eq:`. The 81 section references are the point worth stating -- each names the numbered section it points at (for example "the active-row control of Section~\ref{sec:calib}"), so the reader gets a label to jump to rather than an unresolvable "see earlier".
- **L2 list discipline**: 17 `\item` total across four sections, all genuinely parallel enumerations; reasoning is in prose.
- **L3 emphasis**: bold is rare and load-bearing (Results carries 20 `\textbf` across 5,539 words of source prose; intro 18 across 1,427, method 9 across 2,361, discussion 6 across 2,038); `\emph` carries term introduction.
- **G6 skim test**: Results paragraph openers read as a coherent sequence. Two apparent failures were extraction artifacts -- a sentence continuing after a display equation, and a `\textbf{}` stripped by the math filter.
- **T navigation**: the Roadmap paragraph names what each section establishes.
- **S accuracy**: no un-traced numbers; every value in the gate recomputes from a committed JSON.
