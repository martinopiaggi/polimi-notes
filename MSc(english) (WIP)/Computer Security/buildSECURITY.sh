echo $(pandoc --resource-path=src:src/images src/*.md -o Security.pdf -f markdown-implicit_figures)
echo "PDF generated for Computer Security"