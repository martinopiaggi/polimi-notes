echo $(pandoc --resource-path=src:src/images src/*.md -o ppl.pdf -f markdown-implicit_figures)
echo "PDF generated for Principles of Programming Languages"