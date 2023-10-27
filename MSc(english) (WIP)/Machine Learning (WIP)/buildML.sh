echo $(pandoc --resource-path=src:src/images src/*.md -o ml.pdf -f markdown-implicit_figures)
echo "PDF generated for Machine Learning"