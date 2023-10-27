echo $(pandoc --resource-path=src:src/images src/*.md -o fda.pdf  -f markdown-implicit_figures)
echo "PDF generated for Fondamenti di Automatica"