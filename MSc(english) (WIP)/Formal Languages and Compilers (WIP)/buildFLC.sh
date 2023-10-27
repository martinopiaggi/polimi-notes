echo $(pandoc --resource-path=src:src/images src/*.md -o flc.pdf -f markdown-implicit_figures)
echo "PDF generated for Formal Languages and Compilers"