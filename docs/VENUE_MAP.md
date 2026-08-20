# VENUE_MAP.md — what changes between venues, and what must not

Governing reference for the per-venue source layout. Written before any edit,
from file/line inspection; every claim below cites where it was verified.

Companion to `PAPER_SPINE.md` (which governs *content*). This document governs
*venue furniture only*. If the two ever conflict, the spine wins: no venue
requirement justifies moving a claim.

---

## 0. The one-line summary

Only three things are genuinely venue-specific: **the style file and its
options**, **the identity block**, and **the bibliography style**. Everything a
reader would call "the paper" — every claim, number, hedge, caption, table, the
notation table, the roadmap, the section order — is venue-independent and is
stored exactly once.

---

## 1. Venue-SPECIFIC (differs per venue)

### 1.1 Style file and mode

| site | current (TMLR) | Zenodo preprint |
|---|---|---|
| `main.tex:9` | `\usepackage{tmlr}` | `\usepackage[preprint]{tmlr}` |
| `main.tex:271` | `\bibliographystyle{tmlr}` | unchanged (tmlr.bst is a formatting choice, not a venue claim) |

`tmlr.sty` already ships the option we need. Verified in the style source:

- `tmlr.sty:34-37` — `\newif\if@preprint\@preprintfalse` / `\DeclareOption{preprint}{ \@preprinttrue \@acceptedtrue }`.
  So `preprint` implies `accepted`.
- `tmlr.sty:110-117` — under `\if@accepted` + `\if@preprint`, the running head is
  set empty (`\lhead{}`); the `\else` branch at :117 is what prints
  `Under review as submission to TMLR`.
- `tmlr.sty:125-132` — the same pair of conditionals selects the title block.
  `preprint` takes :127 (`\@startauthor \@author \@endauthor`, i.e. the REAL
  author); the `\else` at :132 is what prints
  `Anonymous authors \\ Paper under double-blind review`.
- The non-preprint `accepted` branch prints
  `{\bf Reviewed on OpenReview:} \openreview` — which is why `preprint`, not
  `accepted`, is the correct option for Zenodo. We are not published in TMLR and
  must not imply we are.

**Verified by prototype compile:** with `[preprint]`, all four pieces of review
furniture are absent from the rendered PDF — `Under review as submission to
TMLR`, `Anonymous authors`, `Paper under double-blind review`, and
`Reviewed on OpenReview`. 28 pages, exit 0.

### 1.2 Identity block

| site | current | Zenodo |
|---|---|---|
| `main.tex:21` | comment: "Single author: fill in identifying details for camera-ready." | retire the comment |
| `main.tex:22-23` | `\author{\name Author Name \email author@institution.edu \\ \addr Institution}` | real name, email, affiliation |
| `main.tex:27` | `\def\openreview{\url{...forum?id=XXXXXXXXXX}}` | delete — unused under `preprint` |
| `main.tex:29-31` | three comments describing double-blind anonymization policy | retire |
| `main.tex:32` | `\newcommand{\anonurl}{\texttt{[[ TODO BEFORE SUBMISSION: anonymized repository URL ]]}}` | retire the macro; print the real URL |
| `main.tex:267` | the single use of `\anonurl{}`, inside the Reproducibility Statement | inline the real URL |

`\openreview` is **defined at :27 and never used** anywhere in the sources —
confirmed by grep across `main.tex` and all of `sections/`. Under `preprint` the
style never expands it either, so deleting it is safe.

`\anonurl` is used at **exactly one site**, `main.tex:267`.

### 1.3 Header comments naming the venue

`main.tex:3`, `:4`, `:6` describe the file as "TMLR submission source" and
explain how to swap the class. These are LaTeX comments — they never render —
but they are wrong for a shared tree and are updated.

**Note (re-measured).** A grep for reviewer-facing terms (`reviewer`,
`rebuttal`, `camera-ready`, `supplementary`, `page limit`, `checklist`) across
both venue `main.tex` files and all eight shared section files returns **two
matches, both of the word "camera-ready", both in `venues/tmlr/main.tex` LaTeX
comments** (lines 9 and 22). Zero matches in any shared section; zero in the
Zenodo venue file. No rendered sentence anywhere addresses a reviewer or assumes
a review process, which is why the venue switch is a furniture change rather than
a rewrite.

An earlier revision of this file claimed the matches were at `main.tex:3`, `:4`
and `:6`. That was wrong in both directions -- those three lines contain none of
the terms, and lines 9 and 22 do. The claim was written from assumption rather
than from grep output and is corrected here.

---

## 2. Venue-INDEPENDENT (stored once, never duplicated)

These live at `paper/` root and are shared by every venue:

- `sections/related.tex`, `method.tex`, `results.tex`, `discussion.tex`,
  `appendix_surgery.tex` — all prose, all claims, all numbers
- `references.bib` — 14 verified entries
- `figures/` — 10 PDF figures, referenced as `figures/figN_*.pdf`
- The abstract and the Reproducibility Statement body (both inside `main.tex`,
  which is why the venue `main.tex` is thin but not empty)

Also venue-independent, and explicitly NOT to be touched by this work:

- Every numeric claim. The numeric-token diff must show no movement beyond the
  DOI, the real URL, and page furniture.
- Every hedge and scope qualifier. "demonstrably", "within noise",
  "leading account", "single-seed", the measurement-window clauses.
- `PAPER_SPINE.md`'s nine layers L0–L8 and their owned-numbers lists.
- The gate checks in `tests/check_paper_claims.py` (113 at the time of writing;
  the count grows as checks are added, so treat the number as measured, not fixed).
- The notation table (`tab:notation`, Method), the Roadmap paragraph (intro),
  the 13 descriptive subsection headings, all 14 captions.
- Section order and every `\ref` target.

---

## 3. The layout, and why it is shaped this way

```
paper/
  sections/        <- shared, one copy
  figures/         <- shared, one copy
  references.bib   <- shared, one copy
  venues/
    tmlr/          main.tex + tmlr.sty tmlr.bst fancyhdr.sty
    zenodo/        main.tex + tmlr.sty tmlr.bst fancyhdr.sty
```

Three constraints were established by prototype, not assumed:

1. **Style files must sit beside the venue `main.tex`.** Tectonic resolves
   `\usepackage` and `\input` relative to the `.tex` being compiled, not the
   invocation directory. A venue folder with the style files one level up fails
   with `! LaTeX Error: File 'tmlr.sty' not found.` — compiling from `paper/`
   root does not fix it. The style files ARE the venue-specific part, so
   co-locating them is also the honest layout.
2. **Shared content is reached with `../../`.** `\input{../../sections/...}`,
   `\bibliography{../../references}`.
3. **`\graphicspath{{../../}}` must come AFTER `\usepackage{graphicx}`.**
   Placed before it, the compile dies with `Undefined control sequence` at the
   `\graphicspath` line. This is the failure that killed the first two prototype
   attempts.

Figure paths inside the shared sections stay **unchanged** (`figures/fig1_*.pdf`)
— `\graphicspath` does the redirection. This matters because the gate has a
family-C check asserting every `\includegraphics` target starts with `figures/`
and is not an artifact marker; rewriting the paths per venue would break it.

---

## 4. What Zenodo needs that TMLR never asked for

TMLR takes a PDF and handles metadata itself. A DOI record does not.

| need | status | why |
|---|---|---|
| `CITATION.cff` | MISSING | machine-readable citation; GitHub renders it, Zenodo reads it |
| `.zenodo.json` | MISSING | controls the DOI record's type, license, creators |
| `LICENSE` | PRESENT (MIT) | must agree with both files above |
| ORCID | SUPPLIED (`0009-0003-1009-0716`) | on both the CITATION.cff author and the .zenodo.json creator |
| Front-matter status note | MISSING | a DOI reader has no venue context and no way to know this is not peer reviewed |
| Archive manifest | MISSING | 62M tree, 42M of it `results/`; what ships must be stated |

---

## 5. Correction to my own earlier claim

An earlier draft of this plan asserted that `README.md:50,53` carried a stale
`480M / 20M` token split "corrected by §3.1 to a 500M pool". **That was wrong
and is withdrawn.** `method.tex:15-17` states "$499{,}990{,}528$ tokens
processed ... The training slice is $480$M unique tokens, so this is $1.04$
epochs" and `method.tex:31-32` states "The token pool is the first $500$M tokens
... of which the final $20$M are held out for validation". The README agrees
exactly. I made the assertion after two greps returned empty output, filling the
gap from assumption instead of reading the passage. **Do not "fix" the README
token table.**

The README's genuine defect is narrower and was verified: `README.md:10-12`
gives the Headline finding as the full-vocabulary median cosine `≈ +1` with
"only ~0.3% of rows have cosine < 0", and never mentions the ≈8% active-row
denominator, the 0.53 active-row median, or the 3–4% opposed figure that the
paper now leads with.

---

## 6. The prose bar carries over

The three read-aloud criteria are a standing acceptance bar, not a completed
pass. The restructure must not drop them, and three items are still open:
43 prose sentences over 40 words, Method 3.5's inline-statistics clause chain,
and 29 paragraphs over 150 words (max 725 — the item the earlier prose pass did
not move). All measurements use the corrected extractor
`(?m)(?<!\\)%.*$`; the buggy `(?m)%.*$` also matched escaped `\%` and hid
roughly half the prose from every measurement in this session, including inside
the gate itself.
