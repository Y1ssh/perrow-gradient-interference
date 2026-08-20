# Edit map -- round 2 of external review

Finding ids in this document carry the `R2-` prefix used in `docs/REVIEW_ROUND2.csv`;
the earlier internal auditor round is namespaced `AUD-` there.

Every finding routed through the spine (`docs/PAPER_SPINE.md`, artifact
`c9b7b4eb-57cd-412c-9028-63662f774d02`) before any file was touched. The rule from the
TMLR build still holds: identify the layer that *owns* the claim, edit there, and let
the other layers reference it rather than restate it.

| id | layer | owns the claim | file edited | verdict |
|----|-------|----------------|-------------|---------|
| R2-A1 | L2 Finding: decoupling | active-row and full-vocabulary figures | `sections/results.tex` (Table 3) | FIXED |
| R2-A2 | L1 Instrument | instrument definition and its caveats | `sections/method.tex`, + limitation in `discussion.tex`, xref in `results.tex` | FIXED |
| R2-A3 | L8 Zero-parameter test | the anchor | `sections/results.tex` | FIXED |
| R2-A4 | L8 | the anchor comparator | `sections/results.tex`, `sections/method.tex` | FIXED |
| R2-A5 | L6 Falsification | the intervention ladder | `sections/discussion.tex` | FIXED |
| R2-A6 | L5 Calibration & generality | rare-token result | `sections/results.tex` (body + caption) | FIXED |
| R2-A7 | L8 | sigma | `sections/results.tex` | FIXED |
| R2-A8 | L8 | anchor arithmetic | `sections/results.tex` | FIXED |
| R2-B | L0 Setting | positioning against prior work | `sections/related.tex` (2 sites) | FIXED |
| R2-C2r | L4 Measurement: which branch | norm decomposition, Lorenz batch | `sections/results.tex` | FIXED |
| R2-D1 | L2 | the decoupling figure | `figures/make_figures.py` | FIXED |
| R2-D2 | L4 | the per-layer figure | `sections/results.tex` (caption) | FIXED |
| R2-D3 | L0 | bibliography | -- | ACCEPTED (unverifiable) |
| R2-D4 | (venue furniture) | -- | `paper/venues/zenodo/main.tex` | FIXED |
| R2-D5 | L9 Moral & scope | scope statements | `sections/discussion.tex` | FIXED |
| R2-D6 | L5 | generality claim | `sections/results.tex` | FIXED |
| R2-D11 | L2 | both aggregates | `sections/abstract.tex` | FIXED |
| R2-D8 | L9 Moral & scope | control-cliff shape | `sections/discussion.tex` | DISCLOSED, not explained |
| R2-D7 | L5 | control replication | (no edit; needs a GPU pass) | OPEN by measurement |
| R2-D9 | L2 | active-row trajectory | (no edit; no per-row arrays committed) | OPEN by measurement |
| R2-D10 | L1 Instrument | exact target mask | (no edit; needs a measurement pass) | OPEN by measurement |
| R2-D12 | (repository) | public and complete | (no edit; owner action) | OWNER ACTION |

## Why R2-A2 went to L1 rather than L2

The reviewer raised it against a statistic in Results, but the defect is in what the
instrument *measures*, which L1 owns. Editing it in Results would have left L1
asserting an identity that L2 then qualified -- the failure mode the spine exists to
prevent. The identity statement in Results now carries a cross-reference to L1
instead of a second explanation.

## What the two substantive findings had in common

Both R2-A1 and R2-A2 were paragraphs I added last round to pre-empt this reviewer's earlier
objections, and in both cases the pre-emption asserted a reconciliation the numbers
did not support. R2-A1 claimed agreement "within 0.2 points" where the identity gives a
14-point spread; R2-A2 offered a bound computed on the rows the mechanism cannot reach.
Prose that gestures at a calculation invites the reader to run it. Both are now either
a table recomputed from the arrays (R2-A1) or a stated open limitation with named
remedies (R2-A2).

## Gate

161 checks at the start of the round, 186 after the substantive repairs, 355 after the
auditor rounds. Eight of those 355 are the C7 family, which audits this table itself and
is excluded from it; the 347 below are the checks C7 counts. Rather than enumerate deltas from memory (the arithmetic slip this
document already made once), here is the current build counted at runtime by family:

| family | checks | what it asserts |
|--------|--------|-----------------|
| A | 28 | source-tree and section-file integrity |
| B | 12 | bibliography entries resolve |
| C | 42 | layout, abstract cap, figure paths, em-dashes |
| C2 | 11 | per-seed surgery losses and both delta columns |
| C3 | 20 | four-row deletion: both range pairs, row identities, per-seed sign flip |
| C4 | 16 | every cell of Table 3, recomputed from the norm-support arrays |
| C5 | 8 | balanced parentheses, one check per shared section file |
| C6 | 2 | the anchor-margin percentage floors the unrounded miss |
| C8 | 5 | the control-cliff passage claims no fitted account; ratio and ceiling recomputed |
| C9 | 12 | this document and `REVIEW_ROUND2.csv` agree on rows and statuses; this document's mutation arithmetic is self-consistent (caught plus missed equals the total, and every miss is narrated); and its typographic-mutation count agrees with `REVIEW_ROUND4.md` |
| C10 | 6 | the register covers the reviewer's whole actionable set and the readiness headline derives from it |
| C11 | 8 | counts shared across the three round-2 documents agree, and gate figures are self-consistent |
| C15 | 39 | deposit metadata no pattern scan sees: `CITATION.cff` and `.zenodo.json` agree on the release date, no empty Zenodo field, every required Zenodo key supplied, no unfilled `<TOKEN>` in `requirements.txt`/`REPRO_NOTES.md`/`setup.sh`/`README.md`, the Muon pin is concrete and not described as outstanding, the audit's substantive hit count is re-verified by re-running the 21-pattern scan inside the gate, its dependency breakdown is recomputed from `requirements.txt`, and no document asserts a release date that disagrees with the metadata |
| C18 | 5 | downloader-facing build facts: the quoted page count against the built PDF, and the build-product dependency count against the gate's own source |
| C19 | 28 | reviewer-round fixes: NextLat attribution, abstract antecedent and caveat, preprint disclosures |
| C17 | 12 | the archive manifest against a live tree walk: shipped file count, byte total, every per-directory row, and the gate check count it quotes |
| C16 | 4 | citation integrity: every `\cite` key exists in `references.bib`, every bib entry is cited, no duplicate keys, and the typeset bibliography matches the cited set |
| C14 | 7 | typographic conventions from the round-4 review: negative ranges, binary-minus spacing, one spelling of the contrastive auxiliary, parenthetical length |
| C13 | 7 | all 21 placeholder patterns over the Zenodo source and both rendered PDFs; the Zenodo build carries none and the TMLR build carries only its double-blind furniture |
| C12 | 5 | `setup.sh` creates every results directory the scripts write to, and no doc describes a fixed hygiene defect as outstanding |
| C20 | 6 | figure-tree byte parity plus comparator self-test, surgery-arm calibration, stray bold |
| D | 6 | ownership and cross-reference integrity |
| E | 55 | numeric claims against committed JSONs |
| F | 3 | measured prose ceilings |
| **total** | **347** | |

One hundred eighty distinct mutations were injected across the round: five against the C3/C4
additions, two against C5/C6, three against C8, two against the family table above, two
against the map/register agreement, three against this sentence's own arithmetic, four
against the caught-versus-total reconciliation, four against the register coverage check,
four against the status-label reconciliation, two against the cross-document count
agreement, four against the setup-directory and hygiene-documentation checks, seven
against the Zenodo and TMLR placeholder scans, three against the gate-total checks, eight against the typographic conventions, two against the cross-document mutation-count agreement, one against the breakdown-sum arithmetic at a total above forty-nine, three against the row-description bindings, four against the deposit-metadata checks, five against the audit's own tally and dependency description, three against the scan-backed hit count, four against the ledger's own arithmetic and the C15 cross-document count, four against citation integrity, three against the family table's row shape, three against the release-date agreement, two against the two checks that had silently skipped when build products were absent, five against the archive manifest's agreement with a live tree walk, and forty-eight against the downloader-facing figures a fresh copy is told to expect (the quoted page count, the prose gate total, the C7 sentence measured on its own table, this breakdown's own arithmetic, the build-product dependency count measured from the gate's own source, that count repeated in the section heading, and the narrated-miss tally, a document reclaiming a rendered PDF date line the style never
typesets, three deletions of shipped PDFs testing that the gate's own denominator
does not move with them, two source-file deletions testing that a partial archive
reports failures instead of a traceback, plus the deposit-identity set (a wrong ORCID
check digit, a reverted ORCID placeholder, the two metadata files disagreeing, and a stale
release date), the three restated-hit-count sentences in the placeholder audit, plus the title-page identity set (the address slot reverting to filler prose, the ORCID losing its hyperlink, and the title page disagreeing with the metadata files), plus the placeholder audit's class claims (a stale per-file count, a classification dropping a file, a stale per-pattern count, and a total raised for a hit nobody classified), plus the deposit-date restatement set (a sibling assertion sentence drifting, the primary one drifting, a new document starting to restate the date, and a metadata file losing it)), and forty against the round-five review fixes and the repairs to it (the NextLat attribution, the abstract's numeric antecedent under two different rewordings, two manifest rows drifting from a live walk, the self-check contradicting the rows it cites, a count sentence appended outside the parsed breakdown, and two bibliography entries reverting from their published venues to arXiv preprints).
One hundred fifty-six were caught on first attempt. Twenty-four were not, and each exposed a real weakness
rather than a typo. First, replacing this sentence's count word with a non-numeric one passed the
first version of the C9 check, which only tested that the sentence existed; the check now
parses the count and sums the breakdown. Second, a live `git config user.*` line injected
into `setup.sh` was reported missed, but the miss was in the test harness, not the check:
the search string carried the wrong indentation, so nothing was substituted and the file
was never mutated. Re-run with the exact text, C12 caught it. Third, the C13 placeholder
set was mutated with an `N/A` token while the check enforced only fourteen of the
twenty-one patterns the audit document claimed; all twenty-one are now enforced, and the
three newly-covered patterns were each mutated separately. Fourth, mutating the audit's stated
dependency breakdown went red only on the family-count checks: the breakdown comparison sat
behind `if _stated:`, so a mutation that broke the regex made the check disappear rather
than fail. Both that check and the hit-tally check now fail explicitly when their sentence
cannot be parsed, and the same defect was audited for across the family. Fifth, changing the page
count the manifest tells a downloader to expect was not caught at all: no check read that
figure, so a fresh copy could be promised a 28-page PDF against a 30-page build. C18 was
written in response and reads the count from `main.aux`, since tectonic writes compressed
object streams and counting page objects in the PDF bytes returns zero -- a probe that
would have passed while never firing. Sixth, desyncing the section heading from the
step-3 sentence was not caught: C18 read only the sentence, so the heading could contradict
it. Both are now read and compared. Seventh, under-stating the miss count passed, because the
narration check tested that at least that many ordinals appeared rather than exactly that
many -- the precise gap that let a real first-attempt miss be booked as a catch. It now
requires equality. Eighth, deleting the shipped `main.pdf` was not caught: four C13
checks and the new date-line check sat inside `os.path.exists` guards, so the gate's own
denominator fell from 293 to 289 while every document still said the total was 293 either
way. The PDFs ship, so absence is now a failure rather than a skip. Ninth, deleting all
figure PDFs dropped ten more: the generator loop iterated over `*.pdf` on disk, so the
checks it emitted depended on what happened to be there. It now iterates over the
`\includegraphics` set the sources fix, and a missing file fails. Tenth, deleting `venues/zenodo/main.tex`
did not fail -- it crashed: thirteen `open(...).read()` sites raised mid-run, killing the
gate before it reported anything, so a partial archive produced a traceback rather than a
failure list. All thirteen now read through a helper that records misses, and one check
names them. Eleventh, changing one of the three sentences that restate the
placeholder audit's hit count was not caught: the gate parsed only the headline sentence, so
a replace that silently no-op'd against a line-wrapped restatement left `93` standing beside
two corrected `89`s in a shipped deposit document. All three restatements are now collected
and required to agree. Twelfth, raising that tally by literal substitution was
not caught either: the gate bound the audit's COUNTS but not its CLASS claims, so when the
ORCID edit added a hit, the count moved and the sentence asserting all of them benign silently
extended to an item nobody had inspected. The scanner now returns per-file and per-pattern
breakdowns, and three checks bind the audit's per-file count, its per-pattern count, and the
specific sentence that classifies them -- scoped to that sentence, since these filenames
appear document-wide and a looser test passed while the classification was wrong. Thirteenth, the stale-release-date check named five files and matched one phrase, so a sibling sentence in
an already-listed file -- "deposit dates refreshed to <date>", edited by hand in the same round
as its neighbour -- drifted uncaught. It now scans the whole tree for three assertion forms,
and a second check binds the set of documents allowed to restate the deposit date, so a new
copy has to be declared rather than silently joining the drift surface. Fourteenth, the LLM-usage disclosure was
bound by matching the phrase "large language model" anywhere in the
reproducibility statement, and that phrase occurs twice -- in the paragraph
heading and in its first sentence -- so deleting one occurrence still passed.
The check now requires the paragraph heading plus the two substantive
commitments the disclosure makes: that no data was model-generated and that the
author is responsible. Fifteenth, the Table 3 rounding note said the table
displayed "one decimal" while $F$ and $a$ are shown to two, and nothing bound a
note's description to the precision of its own table; a check now reads the
body's decimal places and compares. Sixteenth, two figures introduced by the
external-review round were measured once and then quoted as prose with nothing
recomputing them -- the sign-convention gap in the Table 3 caption and the
GPU-hours in the reproducibility statement. Binding them to a live recount
immediately caught a third error: the caption's upper bound read $0.95$ against
a measured $0.952$, rounding a bound in the direction that overstates
agreement. Seventeenth, three bibliography entries were upgraded to
their published venues by hand and nothing held them there; reverting Gloeckle
to an arXiv preprint passed. A check now requires each verified entry to remain
`@inproceedings` with its booktitle intact. Eighteenth, both manifest row parsers used a
character class that cannot match a wildcard aggregate row, so three
per-directory rows added above `phase_c_350m*/` counted the same three files
twice and every check stayed green; the same blind spot also let a stale
Zone.Identifier figure sit in the exclusion table while the self-check carried
the measured one. The parsers now read wildcard rows, forbid a directory
appearing both individually and inside an aggregate, require every row to carry
a description, and cross-check the two Zone counts against the excluded
total. The same round found the upgraded bibliography
entries held to `@inproceedings` and their booktitles but not to their page
ranges; deleting Gerontopoulos's passed. Gerontopoulos ships without a page
range, so the pin covers Gloeckle (PMLR v235:15706-15734, read off the
proceedings listing) and Penedo (30811-30849, from the Crossref
proceedings-article record). Nineteenth, that sentence itself first named the
wrong entry as the uncaught deletion and credited Penedo's range to the
proceedings listing rather than Crossref; nothing bound the narration's entity
names to the bibliography, so a mutation swapping them passed. Two checks now
verify that every entry the ledger says ships without a page range really
lacks one, and that every range it quotes appears in the entry it names. Both
now read the document with whitespace normalised. Only the first needed it: its
first version used literal spaces, the sentence wraps mid-phrase, and it
silently found no names to check at all. The range check's pattern already
spanned newlines, so normalising it changes nothing today and is there so the
two read the same way. Getting that second edit in took a retry -- the first
attempt targeted the block at its old indentation and no-op'd, which is the
same silent-replace failure recorded above, caught here only because the audit
record and the gate were compared line by line. A check now reads this gate's own
source and fails if the ledger claims both narration checks normalise
whitespace while fewer than two of them do. Twentieth, the surgery-arm
calibration reverted cleanly under mutation: nothing bound Appendix A's
disclaimer to the inferences the main text draws from it. Twenty-first, the
removal of a stray bold reverted the same way, with nothing pinning it out.
Both are now checked. Twenty-second, the figure trees
themselves were unbound: `figures/` and `paper/figures/` hold the same figures,
`\graphicspath` reads the second while `make_figures.py` writes the first, and a
regenerated figure reached neither build until the copies were synced. The first
version of that check compared extracted text, which covers labels but not the
plotted data, and "the figures are identical" was asserted on that basis --
a figure whose bars moved under unchanged axis labels would have passed. It now
compares bytes with only the creation timestamp removed, which was verified
against a deliberately perturbed data series that the text comparison missed. Twenty-third, the conclusion's
single-seed marker was itself unbound: deleting the words that calibrate the
sentence passed, because the check tested only that the old over-inference was
gone. It now requires the marker to be present as well. Twenty-fourth, the figure comparator
itself was unguarded: editing it to hash one path twice reported parity forever
and passed. The first guard did not close it either -- the self-test called the
comparator correctly, so it could not see the scan loop being neutered. The two
now share one helper, which the self-test exercises on a pair of figures known
to differ, and all three ways of defanging it fail.

The C7 family exists because this
document's accounting drifted repeatedly: a total updated without its enumeration, an
enumeration that summed to 27 against a delta of 35, and a gate table updated for a new
family while the finding table and mutation count above it were left stale. The gate now fails if this table
and the live check counts disagree.
