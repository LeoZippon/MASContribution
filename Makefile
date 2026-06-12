all: main.pdf

main.pdf: main.tex references.tex
	/home/lzp/.conda/envs/latex/bin/tectonic --synctex --keep-logs --keep-intermediates main.tex

clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.toc main.pdf main.synctex.gz
