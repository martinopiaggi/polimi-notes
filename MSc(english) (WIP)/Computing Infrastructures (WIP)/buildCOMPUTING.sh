echo $(pandoc --resource-path=src:src/images src/*.md -o computing.pdf -f markdown-implicit_figures)
echo "PDF generated for Computing Infrastructures"