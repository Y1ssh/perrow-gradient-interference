# Placeholder audit: is anything still unfilled?

Two scans. The first walks 119 text files for 21 placeholder patterns: INSERT, TODO,
FIXME, XXX, TBD, PLACEHOLDER, anonymi[sz]ed, Anonymous, `0000-0000`, ORCID digit
patterns, example.com, `YOUR_`, `<lowercase>`, FILL, bracketed
`[insert|fill|add|update ...]`, lorem ipsum, `???`, a zeroed DOI, `{{`, N/A, unknown --
21 entries, the count checked against the scanner's own pattern set rather than asserted.
It excludes `results/`, `.git`, binaries, LaTeX build products, **and `paper/venues/`**.

The second scan covers `paper/venues/` directly, since that is where the manuscript
source lives: the same 21 patterns over every text file there (`main.tex`, `tmlr.sty`,
`fancyhdr.sty`, `tmlr.bst`, `README.md`) and over the extracted text of both rendered
31-page PDFs. The first rows of the table below report it.

**Verdict: nothing unfilled reaches the Zenodo record.** No owner TODO remains: the ORCID is
supplied as 0009-0003-1009-0716. Five real defects were found and fixed in this pass, all in
the repository furniture rather than the paper.

| item | state | evidence |
|---|---|---|
| `paper/venues/zenodo/main.tex` + main.pdf | clean | all 21 patterns run over the file directly and over the rendered PDF's full extracted text: zero hits in either. `\repourl` resolves to `https://github.com/Y1ssh/perrow-gradient-interference`; author block reads Yash Madelwar / ymadelwa@asu.edu / Independent Researcher; 0 undefined references |
| `paper/venues/zenodo/tmlr.sty` | inert | carries "Anonymous authors / Paper under double-blind review" inside the `\else` branch of `\@maketitle`. `main.tex` loads the class as `\usepackage[preprint]{tmlr}`, so that branch never executes -- confirmed by the rendered PDF, which contains neither string |
| `paper/venues/tmlr/main.pdf` | double-blind furniture only | all 21 patterns over the rendered text return exactly three: `TODO`, `anonymized` (both from the anonymised repository URL in the Reproducibility Statement) and `Anonymous` (the title block). None of the other eighteen appears |
| `paper/venues/tmlr/main.tex` | placeholder, correct | `\repourl` is `[[ TODO BEFORE SUBMISSION: anonymized repository URL ]]`. That build is double-blind, so a real URL there would break anonymity. Not shipped as the Zenodo record; SHIP_CHECKLIST now names it as a TMLR-only camera-ready action |
| `CITATION.cff` | clean | the ORCID is supplied and live: `0009-0003-1009-0716`, checksum-valid. Gate family E already fails on a *live* `0000-0000-0000-0000`, and it is commented, so it cannot reach the DOI record. Every other field is real |
| `.zenodo.json` | clean | no placeholder token; `orcid` is a real, checksum-valid identifier on the creator record |
| `LICENSE` | clean | `Copyright (c) 2026 Yash Madelwar` (the corrupted `(X)Y ash` was fixed earlier) |
| `setup.sh` -- git identity | FIXED this pass | the hardcoded `yash@research.local` was gone, but three documents still described it as present. `README.md`, `REPRO_NOTES.md` and `clean_repo.sh` now describe the current state, and the file's only remaining `example.com` is a commented instruction to the cloning user |
| `setup.sh` -- results dirs | FIXED this pass | it created only `phase_a`--`phase_d`, so a fresh clone's first norm-support run had nowhere to write. It now creates `phase_a`--`phase_e`, `norm_support` and `checkpoints`. The phantom `phase_f` it once created is gone, and nothing references it |
| `clean_repo.sh` | FIXED this pass | it grepped for the removed identity and printed .gitignore lines that are all already present, so it reported phantom work. It now checks for a *live* `git config user.*` line and verifies .gitignore coverage. Dry run reports: no live identity, no phase_f, all five patterns present |
| `REPRO_NOTES.md` | FIXED this pass | three hygiene items were written as outstanding when all three are done. Restated in the past tense with what was actually changed |
| dates | consistent | `CITATION.cff` date-released and `.zenodo.json` publication_date agree, and match the deposit date. The PDF itself prints no date: `\month`/`\year` are set to 08/2026 in `main.tex`, but the preprint style does not typeset them, so the deposit date lives only in the two metadata files. Both were 2026-08-11 when first written and were moved forward as the deposit slipped; see the round-5 section for the current value |
| `README.md` "YOUR cuda" | not a placeholder | an instruction to the reader installing torch |
| checklists and audit records | expected | `PRE_UPLOAD_CHECKLIST`, `VENUE_MAP`, `SHIP_CHECKLIST`, `PROSE_REVISION_CHANGELOG`, `UPLOAD_READINESS` contain the words "placeholder", "TODO", "anonymized" *because they document them*. `tests/check_paper_claims.py` contains the literal patterns because it greps for them |
| figure PNGs | false positives | `XXX`, `???`, `{{` occur inside compressed image bytes in 10 PNGs |

## The five fixes

`setup.sh` created `results/phase_a` through `phase_d` and nothing else, while the shipped
scripts write to `results/norm_support` and `results/checkpoints` as well. On a fresh
clone the first norm-support run would have failed on a missing directory. That is the one
finding here with a functional consequence.

The other four were documentation describing defects that had already been fixed:
`README.md`, `REPRO_NOTES.md` and `clean_repo.sh` all still told the reader to remove a
hardcoded git identity that is not there, and `clean_repo.sh` additionally printed five
`.gitignore` additions that are all already present. A reader following those instructions
would have gone looking for work that does not exist.

`docs/SHIP_CHECKLIST.md` listed the anonymous author block as a camera-ready action
without saying which venue it applies to. It now says TMLR only, and names the
`\repourl` placeholder alongside it, so the one legitimate placeholder in the tree is
recorded where someone preparing a camera-ready will see it.

## Gated

Gate family C13 (7 checks) runs all 21 patterns over `venues/zenodo/main.tex` and over the
extracted text of both rendered PDFs on every gate run. For the Zenodo build it requires
zero hits, preprint mode, and a live `\repourl`, so the anonymous branch of `tmlr.sty`
cannot reach the page. For the TMLR build it requires the anonymous title block to be
present -- that build must stay double-blind -- and allows exactly three tokens there
(`TODO`, `anonymized`, `Anonymous`, all from the anonymised repository URL and the review
header), failing on any of the other eighteen. Seven mutations injected: dropping preprint
mode, restoring a placeholder `\repourl`, adding a TBD, an `N/A`, an `XXX` and a
`<lowercase>` token, and rebuilding the TMLR PDF in preprint mode to confirm the anonymity
check fires. The `N/A` mutation was missed on first attempt, because at that point C13
enforced only fourteen of the twenty-one patterns; all twenty-one are now enforced and it
is caught.

Gate family C12 (5 checks) fails if `setup.sh` omits a results directory the shipped
scripts write to, references a nonexistent phase, or carries a live `git config user.*`
line, and if `README.md` or `clean_repo.sh` describes the removed git identity as present.
Five mutations injected, all five caught.

Gate 355/355. Both venues rebuild at 31 pages with 0 undefined references, and a numeric
token diff over both rendered PDFs is empty in each direction: no claim moved.

## Correction to this document's own first version

It said "21 placeholder patterns" over a list of 20 (the `unknown` pattern was missing),
and gave the excluded set as "`results/`, `.git`, and binaries" when `paper/venues/` --
the manuscript source -- was excluded from that walk too. The check I ran to confirm the
count was a regex over an empty string, so it compared the pattern set against itself and
never read the list. Both are fixed above: the enumeration is 21 entries verified against
the scanner's own keys, the venues tree now has its own scan reported in the table, and C13
runs all 21 patterns on every gate invocation, with the gate total in this document itself
now checked against the live count by C11, over both PDFs, rather than relying on this
note. Two further errors in that first version, both found the same way: it described C13
as running "the placeholder set" when the check enforced only 14 of the 21 patterns, and
it claimed the venues scan covered both rendered PDFs when the walk excluded `.pdf` files
and only the Zenodo PDF had been extracted. The TMLR PDF is now scanned, and its row is
in the table above.

## Re-scan, round 5 (first run 2026-08-15; deposit dates refreshed to 2026-08-20)

Re-run against the current tree after the round-4 typography pass and the venue
restructure. 113 text files, the same 21 patterns, plus two checks the pattern
scan structurally cannot make: empty metadata fields, and uppercase
angle-bracket tokens.

**The pattern scan found nothing new.** The figure worth quoting is
**94 hits outside those two files** -- this document and `check_paper_claims.py`
both contain the pattern list, so a whole-tree total counts the instrument along
with what it measures and moves whenever either file is edited. (At the time of
writing the whole-tree total was 245, of which 151 were in those two files.)
Every one of the 94 resolves to one of three benign classes. First, TMLR's
by-design double-blind furniture (`\repourl`'s `[[ TODO ]]`, `tmlr.sty`'s
unexecuted `\else` branch, the checklist items tracking them). Second, ordinary
code (`sys.path.insert`, `'N/A'` print fallbacks, `{{` in f-strings and BibTeX
braces). Third -- added 2026-08-18 -- the two `orcid.org/0009-0003-1009-0716`
occurrences in `CITATION.cff` and `venues/zenodo/main.tex`, and prose in the
other audit records (`docs/EDIT_MAP_ROUND2.md` uses the word *placeholder* to
describe this scan, and writes assertion forms as `<date>`, both of which the
scan then counts). These are a
*supplied* identifier caught by a pattern written to find unsupplied ones: the
`orcid_pat` regex matches `orcid.org/` followed by up to nineteen digits and
dashes, so a real ORCID trips it exactly as `orcid.org/0000-0000-0000-0000`
would. The check digit is verified separately by gate family E, which is what
distinguishes the two cases. `venues/zenodo/main.tex` returns two hits: that
ORCID and `\graphicspath{{../../}}`, which is required LaTeX syntax.

**Three defects the pattern scan could not see, all fixed:**

1. **`requirements.txt` still said `Replace <COMMIT_SHA> with the SHA you used`**
   directly above a line that already pinned `Muon@056a3c5869cf`. The
   instruction survived the resolution. A reader following it would have
   unpinned a working pin. The lowercase `<[a-z_]+>` pattern used in earlier
   rounds could not match an uppercase token; that gap is now closed by a
   case-aware check.
2. **`REPRO_NOTES.md` listed the Muon pin under "What still needs a human
   decision"**, describing `requirements.txt` as "currently ends in the bare
   `git+https://github.com/KellerJordan/Muon` (HEAD)". It does not, and had not
   for some time. Rewritten to what is actually outstanding: the SHA is inferred
   from the instance setup date, not read off the training box, so it is worth
   confirming with `pip show muon` if that box is still reachable.
3. **`CITATION.cff` and `.zenodo.json` both carried 2026-08-11**, the day the
   metadata was written rather than the deposit date. Both have been moved
   forward each time the deposit slipped a day; they now read **2026-08-20**. The lesson is that this pair needs re-checking on the
   day of deposit, not once: gate family C15 enforces that the two files agree
   with each other, but only the depositor knows the real date.

**Verified clean by direct inspection, not by pattern:** `.zenodo.json` has no
empty or null field and supplies every key Zenodo reads for a preprint deposit
(`upload_type` publication, `publication_type` preprint, MIT license, open
access, version 1.0.0, one creator, an 1,414-character description, eight
keywords, and an `isSupplementTo` link to the repository).

`requirements.txt` is **not** fully pinned, and calling it so in an earlier
draft of this section overstated it. Of six requirements, one is a git commit
pin (Muon `056a3c5869cf`), two are bounded ranges (`torch>=2.4,<2.10`,
`numpy>=1.24,<2.2`), and three are lower bounds with no ceiling (`scipy>=1.11`,
`tiktoken>=0.7`, `datasets>=2.19`). The ranges are deliberate: no result JSON
recorded a torch or CUDA version, so the file states a conservative window
rather than inventing a pin, and `REPRO_NOTES.md` names that as an open item.
What matters for reproduction is that the optimizer, the one dependency with no
PyPI versioning at all, is pinned to a commit.

The only zero-byte files in the tree are two `.gitkeep` markers, which is their
purpose.

**Correct by design, unchanged:** TMLR's `\repourl` placeholder (double-blind).
The ORCID is no longer outstanding: it is supplied as `0009-0003-1009-0716` in
both metadata files, and its checksum validates.

**Gate family C15 (25 checks)** covers the class: the two metadata files must
agree on the release date, `.zenodo.json` must have no empty field and must
supply each required key, no `<TOKEN>` may appear in an install or setup file,
and the Muon pin must be concrete and not simultaneously described as
outstanding. It also re-derives this section's own two measured claims rather than trusting
them: the 21-pattern scan is re-executed inside the gate and its substantive hit
count compared against the 94 quoted above, and the dependency breakdown is
recomputed from `requirements.txt`. Both of those were wrong in the first
draft of this section -- 138 written where the scan gave 200, and a claim of full pinning
for a file with three unbounded lower bounds -- which is the same failure the
earlier rounds kept hitting: a sentence describing a measurement, written once
and never recomputed. Sixteen mutations against this family, fifteen caught on first attempt. The one
miss is recorded in `EDIT_MAP_ROUND2.md`: mutating the stated dependency
breakdown went red only on the family-count checks, because the comparison
sat behind a conditional and a broken regex made it vanish rather than fail.
