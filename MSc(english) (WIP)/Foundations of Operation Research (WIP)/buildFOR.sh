echo $(pandoc --resource-path=src:src/images src/*.md -o for.pdf -f markdown-implicit_figures)
echo "PDF generated for Foundations Of Operation Research"