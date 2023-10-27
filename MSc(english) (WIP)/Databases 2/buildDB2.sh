echo $(pandoc --resource-path=src:src/images src/*.md -o databases2.pdf -f markdown-implicit_figures)
echo "PDF generated for Databases 2"