## Resubmission of an archived package

vennLasso was archived on 2023-09-21 because check issues were not corrected in time.
The issues on the CRAN check page at the time of archival were:

* ERROR (r-devel-linux-x86_64-fedora-clang, and the M1mac additional check): re-building the
  vignette failed with "Package natbib Error: Bibliography not compatible with author-year
  citations". The vignette is now an HTML vignette (rmarkdown::html_vignette) with
  citations processed by pandoc, so it needs no LaTeX at all.
* NOTE (all flavors): 'LazyData' is specified without a 'data' directory. Removed.
* NOTE (macOS flavors): installed size 27.3Mb, libs 26.8Mb. This comes from the debug symbols
  of the compiled Eigen code in CRAN's macOS builds.

This version (0.2.0) also fixes a number of defects in the fitting functions (see NEWS.md),
registers its native routines, is compatible with Eigen 5, and adds a testthat suite.

## Test environments

* local macOS Sequoia 15.7 (aarch64), R 4.5.1, Apple clang 17, with an empty user Makevars
* local build with -fsanitize=address (AddressSanitizer): the tests and all examples, including
  \donttest, ran with no report
* R-hub: linux (R-devel), windows (R-devel), macos-arm64 (R-devel), clang-asan: all passed
  (Status: OK on linux, windows and clang-asan; macos-arm64 passed with no error or warning)
* win-builder: R-release (R 4.6.2) and R-devel (2026-11-12 r90881): 1 NOTE each (see below)

## R CMD check results

Local check: 0 errors | 0 warnings | 2 notes

* checking CRAN incoming feasibility ... NOTE
  New submission. Package was archived on CRAN.
* checking HTML version of manual ... NOTE
  Skipping checking HTML validation: 'tidy' doesn't look like recent enough HTML Tidy.
  (local tool)

win-builder R-release and R-devel: 0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  New submission. Package was archived on CRAN.
  Possibly misspelled words in DESCRIPTION: Huling, et, al.
  These are the first author's surname and "et al." in the citation
  Huling et al. (2018) <doi:10.1111/biom.12769>.

R-hub (linux, windows, macos-arm64, clang-asan): 0 errors | 0 warnings.
