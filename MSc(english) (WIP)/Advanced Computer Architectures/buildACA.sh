echo $(pandoc --resource-path=src:src/images src/*.md -o aca.pdf -f markdown-implicit_figures)
echo "PDF generated for Advanced Computer Infrastructures"