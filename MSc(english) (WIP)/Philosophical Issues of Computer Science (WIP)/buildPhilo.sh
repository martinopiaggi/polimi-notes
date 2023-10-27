echo $(pandoc --resource-path=src:src/images src/*.md -o "philosophical issues.pdf" -f markdown-implicit_figures)
echo "PDF generated for Philosophical Issues of Computer Science"