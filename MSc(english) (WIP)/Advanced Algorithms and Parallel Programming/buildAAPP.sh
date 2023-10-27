echo $(pandoc --resource-path=src:src/images src/*.md -o aapp.pdf -f markdown-implicit_figures)
echo "PDF generated for Advanced Algorithms and Parallel Programming"