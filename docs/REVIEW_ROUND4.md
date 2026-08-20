# Round-4 review: verification and disposition

Eleven items. **Five accepted and fixed, six rejected on measurement.** Every rejection
below is backed by a measurement against the source, the embedded font tables, or a
rendered crop of the page in question -- not by an argument that the reviewer was careless.

The pattern in the rejections is worth naming: five of the six are artifacts of PDF **text
extraction**, not of the document. A PDF stores glyph positions, and a text extractor
reconstructs a character stream from them. Where LaTeX sets an accent as a separate
glyph, or breaks a URL across a line, or uses a math font whose ToUnicode table maps the
summation slot to `P`, the extracted string is wrong while the page is right. Any review
conducted by copying text out of the PDF will see all five.

| item | verdict | evidence |
|---|---|---|
| Abstract URL has spaces and a trailing period | REJECTED — extraction artifact | The source is `\url{https://github.com/Y1ssh/perrow-gradient-interference}` inside `\repourl`, rendered by hyperref. Per-glyph box measurement on page 1 shows inter-character advances of 0.4-3.6pt with no space glyph wider than the natural typewriter tracking; the apparent gaps are where pypdfium2's extractor inserts breaks at hyperref's line-break opportunities. The visual crop shows an unbroken URL. The trailing period is the sentence's full stop, outside the link: the PDF carries a `/URI` annotation whose target is exactly `https://github.com/Y1ssh/perrow-gradient-interference`, no period. An indexing scraper reads the annotation, not the glyph run |
| Replacement characters in crossentropy, normweighted, batchrelative, improving | REJECTED — not present | Zero U+FFFD in the source and zero in the extracted PDF text. `cross-entropy` appears 13 times in the rendered text with its hyphen, `norm-weighted` 10 times. The concatenated spellings the review quotes appear 0 times in either. Whatever dropped those hyphens is in the reviewer's copy or their own extraction, not the document |
| \sum renders as the letter P; fonts not embedded | REJECTED — fonts are embedded; this is a ToUnicode mapping | All 52 fonts are subset-embedded (`XXXXXX+` prefixes, 29 FontFile objects, 108 FontDescriptors); none is referenced without embedding, so the PDF/A concern does not apply. The `\sum` glyph comes from LMMathExtension10, whose CharSet includes `/summationtext` and `/summationdisplay`; it renders correctly, as the crop shows. What the review saw is the font's ToUnicode CMap mapping the summation slot to U+0050 — a Type1 math-font convention where the extension font's code point for the large operator collides with 'P'. It affects copy-paste, not the page. Tested `cmap` + `T1 fontenc`: the extracted text was byte-identical, so it is not fixable from the source side without switching math font packages |
| Misaligned tilde: q\tilde{} should be \tilde{q} | REJECTED — already \tilde{q} | The source has six `\tilde{q}` and zero `q\tilde`. The crop shows the tilde centred over the q. The inner product is already `\langle\tilde{q},z\rangle` |
| Detached hats and asterisks: q ∗ s, KL(Pˆ 1∥qˆs) | REJECTED — already tightly bracketed | Source uses `\hat{P}_1` (6x), `\hat{q}_s` (3x), `q^{*}_s`, and `\Vert` for the KL bar. The crop shows correctly-positioned hats and subscripts. The detachment is the text extractor emitting the accent as a separate character |
| Dash between negative numbers: -0.20--0.14 | ACCEPTED and fixed, then fixed again | Four ranges in results.tex where both endpoints are negative. The review's suggested remedy, writing the range as "to", was applied first and was wrong here: every one of these ranges sits inside a "moves it from X to Y" sentence, so "from -0.20 to -0.14 to +0.18--+0.25" reads as a three-point chain and a reader can take the pre-deletion interval for the move itself. They are now intervals, `$[{-}0.20,\,{-}0.14]$ to $[+0.18,\,+0.25]$`, which removes the ambiguity in both directions and is unambiguous about what a range is. A sentence naming the intervals as the across-seed spread was added |
| Gnce in text vs G_nce in Figure 9 | ACCEPTED and fixed | The figure generator labelled bars with the literal string `G_nce`, which matplotlib rendered as a bare underscore. Labels are now `$G_{\mathrm{nce}}$`, matching the body's `$G_{\text{nce}}$`. One stray `\mathrm{G}` in results.tex normalized to the italic G used everywhere else. Figures regenerated |
| Nested parentheses in the t=48.5 sentence | ACCEPTED and fixed | Rewritten as the review suggested but the other way round: the headline statistics lead, and the rounding provenance follows as its own sentence. No value changed |
| 62-word parenthetical in the 350M passage | ACCEPTED and fixed | The result now leads (`CE-only 3.752, shared-MTP 4.129`), with the token accounting as three following sentences. Longest remaining parenthetical is 43 words |
| Recursive self-reference: Section 5 pointing at itself | REJECTED — none of the three is inside Section 5 | Three `\ref{sec:discussion}` sites exist, in intro.tex, results.tex and appendix_surgery.tex. `\label{sec:discussion}` is in discussion.tex, which contains no reference to itself. The rendered sites are pages 2, 23 and 29; Section 5 starts on page 24. Section 5 does carry the argument they point at: the symmetric-projection exactness and the head-only empirical null |
| Inconsistent spacing: -0.3186/ - 0.3355 | ACCEPTED and fixed | A real TeX defect, not an extraction artifact. `$-0.3186/-0.3355/-0.2964$` sets the second and third minus signs as binary operators, so TeX adds medium space either side. Now `${-}0.3186/{-}0.3355/{-}0.2964$`, which sets them as unary |

## The one that would have shipped

The binary-minus spacing in Appendix A is a genuine typesetting defect and the review is
the only pass that caught it. `$-0.3186/-0.3355$` asks TeX to read the second minus as
subtraction, so it sets `0.3186 / -0.3355` with operator spacing around the slash and the
sign. Wrapping each in braces, `${-}0.3186/{-}0.3355$`, sets them as unary signs and the
spacing closes up.

## What the ToUnicode finding means for Zenodo

The review's inference (missing font embedding, PDF/A risk) does not hold: every font is
subset-embedded. But the underlying observation is real and worth stating, because it
affects anyone who copies text out of the archived PDF. In Type1 math fonts the large
summation glyph occupies a code point whose ToUnicode entry resolves to U+0050. Adding
`cmap` and `T1` font encoding changes nothing -- I compiled with both and the extracted
text came back byte-identical -- because the mapping lives in the math extension font, not
in the text encoding. Fixing it would mean changing math font packages, which would reflow
every equation in the paper. Not worth doing for a copy-paste artifact.

## Gated

Gate family C14 (7 checks) fails on: an en-dash range between two negative numbers, a
"from A to B to C" three-point chain, a
slash-separated negative set without brace-wrapped signs, more than one spelling of the
contrastive auxiliary in the body, a bare `"G_nce"` label string in the figure generator,
any parenthetical over 50 words, and the return of the nested paired-test parenthetical.
Eight mutations injected, all eight caught. A ninth class, the agreement between this
document's mutation count and the edit map's, is checked by C9 after the two disagreed
(three against eight) because each had been edited from its own narrative.

Gate 355/355. Both venues rebuild at 31 pages, 0 undefined references, and the numeric
token diff over the rendered PDF is empty in both directions: 384 tokens before, 384
after, nothing added or removed. No claim moved.
