echo $(pandoc --resource-path=src:src/images src/*.md -o meccanica.pdf -f markdown-implicit_figures)
echo "PDF generated for Meccanica"