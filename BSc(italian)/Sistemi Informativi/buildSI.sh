echo $(pandoc --resource-path=src:src/images src/*.md -o si.pdf -f markdown-implicit_figures)
echo "PDF generated for Sistemi Informativi"