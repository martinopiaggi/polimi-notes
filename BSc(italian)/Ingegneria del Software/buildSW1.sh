echo $(pandoc --resource-path=src:src/images src/*.md -o sw.pdf -f markdown-implicit_figures)
echo "PDF generated for Ingegneria del Software"