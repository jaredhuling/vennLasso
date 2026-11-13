#!/bin/sh
# Renders the TikZ figures of the vignette to PNG (requires a LaTeX installation with
# TikZ, pdfcrop and Ghostscript). Run from vignettes/figures-src; output in vignettes/figures.
# usage: sh render.sh <work dir for LaTeX intermediate files>
set -e
W=${1:?work directory}
mkdir -p "$W" ../figures
for f in strata_data sel_patterns_one sel_patterns; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$W" "$f.tex" > "$W/$f.out"
  pdfcrop --margins 6 "$W/$f.pdf" "$W/$f-crop.pdf" > /dev/null
  gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r150 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 \
     -sOutputFile="../figures/$f.png" "$W/$f-crop.pdf"
done
