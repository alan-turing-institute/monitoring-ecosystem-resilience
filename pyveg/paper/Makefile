paper.pdf : paper.md
	pandoc --bibliography paper.bib --filter pandoc-citeproc --toc --toc-depth=2 --number-sections -s paper.md -o paper.pdf
