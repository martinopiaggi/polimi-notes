echo $(pandoc --resource-path=src:src/images src/*.md -o cg.pdf -f markdown-implicit_figures)
echo "PDF generated for Computer Graphics"