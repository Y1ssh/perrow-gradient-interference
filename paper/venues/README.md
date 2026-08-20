# paper/venues/ — per-venue build folders

Shared content lives **once**, at `paper/`:

- `sections/` — all prose, all claims, all numbers
- `figures/` — the 10 figure PDFs
- `references.bib` — the bibliography

Each venue folder holds only what genuinely differs: its own style files and a
thin `main.tex`.

## Building

    cd paper/venues/zenodo && tectonic -X compile main.tex --outdir .
    cd paper/venues/tmlr   && tectonic -X compile main.tex --outdir .

Both produce a 28-page PDF with 0 undefined references.

## The two venues

| | `tmlr/` | `zenodo/` |
|---|---|---|
| style option | `\usepackage{tmlr}` | `\usepackage[preprint]{tmlr}` |
| running head | "Under review as submission to TMLR" | none |
| title block | "Anonymous authors / Paper under double-blind review" | real author block |
| OpenReview line | n/a (submission mode) | suppressed by `preprint` |

The two `main.tex` files differ in nine hunks, all venue furniture
(verified by `diff`): the header comment block (two hunks), the style option
line, a `\graphicspath` trailing comment, the author block, `\def\month`, the
deleted `\openreview` definition, the `\repourl` definition, and one blank line. Nothing
that renders as a claim differs. All shared content is reached by `\input` and is
literally the same bytes for both.

## Three constraints, learned the hard way

1. **Style files must sit beside the venue `main.tex`.** Tectonic resolves
   `\usepackage` and `\input` relative to the `.tex` being compiled, not the
   working directory. Keeping the style files at `paper/` and compiling from
   there still fails with `! LaTeX Error: File 'tmlr.sty' not found.`
2. **Shared content is reached with `../../`** — `\input{../../sections/...}`,
   `\bibliography{../../references}`.
3. **`\graphicspath{{../../}}` must come AFTER `\usepackage{graphicx}`.** Before
   it, the compile dies with `Undefined control sequence`.

Figure paths inside `sections/` are **unchanged** (`figures/figN_*.pdf`);
`\graphicspath` does the redirection. Do not rewrite them per venue — the claim
gate asserts every `\includegraphics` target starts with `figures/`.

## Adding a venue

Copy an existing folder, swap the style files and the class/option line, and
leave every `../../` path alone. Never copy `sections/`, `figures/` or
`references.bib` into a venue folder: two copies of a claim is how a paper starts
contradicting itself.

See `docs/VENUE_MAP.md` for the full venue-specific vs venue-independent split.
