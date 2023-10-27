echo $(pandoc --resource-path=src:src/images src/*.md -o hypermediaApps.pdf -f markdown-implicit_figures)
echo "PDF generated for Hypermedia Applications"