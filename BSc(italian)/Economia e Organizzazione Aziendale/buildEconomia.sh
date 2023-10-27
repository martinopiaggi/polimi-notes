echo $(pandoc --resource-path=src:src/images src/*.md -o eco.pdf  -f markdown-implicit_figures)
echo "PDF generated for Economia e Organizzazione Aziendale"