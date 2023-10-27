echo $(pandoc --resource-path=src:src/images src/*.md -o psi.pdf -f markdown-implicit_figures)
echo "PDF generated for Probabilitá e Statistica per l'informatica"