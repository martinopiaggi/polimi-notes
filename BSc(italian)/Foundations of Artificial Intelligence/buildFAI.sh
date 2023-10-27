echo $(pandoc --resource-path=src:src/images src/*.md -o fai.pdf -f markdown-implicit_figures)
echo "PDF generated for Foundations of AI"