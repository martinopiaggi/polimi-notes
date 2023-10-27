echo $(pandoc --resource-path=src:src/images src/*.md -o sw2.pdf -f markdown-implicit_figures)
echo "PDF generated for Software Engineering 2"