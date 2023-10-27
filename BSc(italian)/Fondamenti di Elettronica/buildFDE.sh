echo $(pandoc --resource-path=src:src/images src/*.md -o fde.pdf  -f markdown-implicit_figures)
echo "PDF generated for Fondamenti di Elettronica"