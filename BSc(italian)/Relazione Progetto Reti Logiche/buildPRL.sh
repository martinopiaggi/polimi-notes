echo $(pandoc --resource-path=src:src/images src/*.md -o progetto_reti_logiche.pdf -f markdown-implicit_figures)
echo "PDF generated for Relazione Progetto Reti Logiche"