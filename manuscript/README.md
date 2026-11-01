# Manuscript: Bayesian selection of fixed and random parameters in crash frequency models

Target journal: *Transportation Research Part B: Methodological* (Elsevier `elsarticle` class, author-year references).

## Build

```bash
cd manuscript
latexmk -pdf main.tex
```

## Layout

| File | Content | Status |
|---|---|---|
| `main.tex` | Front matter, abstract, keywords, declarations | Drafted. Affiliation, e-mail and results sentences pending |
| `sections/01-introduction.tex` | Motivation, gap, contributions | Drafted. Contribution (iv) pending results |
| `sections/02-background.tex` | Random parameters crash models, specification practice, Bayesian selection | Drafted. Final literature check pending |
| `sections/03-model.tex` | RPNB model, identification results, three-state prior, posterior summaries | Drafted |
| `sections/04-computation.tex` | Pólya-Gamma and CRT augmentation, collapsed random-parameter step, sampler | Drafted. Run times pending |
| `sections/05-simulation.tex` | Design, competitors, metrics, table shells | Design drafted. Results pending |
| `sections/06-application.tex` | Data, setup, results, sensitivity | Outline only. Data set not chosen |
| `sections/07-discussion.tex` | Extensions and limitations | Drafted |
| `sections/08-conclusion.tex` | Conclusions | Drafted. Findings pending |
| `sections/appendix.tex` | Proofs and derivations | Drafted |
| `references.bib` | Bibliography | Entries with `VERIFY` in the note field need checking |
| `highlights.txt` | Elsevier highlights | Drafted |

Pending items are marked in the PDF with red `\tbd{}` boxes.

## Next steps

1. Check the derivations in Section 3.2 and Appendix A independently.
2. Implement the sampler (Box 1) and test it on DGP-A before running the full simulation grid.
3. Choose and obtain the application data (a multi-year segment panel is preferred, see Proposition 2).
4. Verify the bibliography entries tagged `VERIFY`.
