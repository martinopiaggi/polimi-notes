# Polimi notes 

This is a collection of my notes for various **Computer Science and Engineering** courses that I attended. The repo includes:
- **Master's degree course notes (Work In Progress):**
    - All notes are in English
    - Some course notes are incomplete since still in progress or boring 
- **Bachelor's degree course notes:**
	- All notes are in Italian, except for Foundations of AI
    - Notes are available only for some courses of the study plan

These notes were created during periods of desperate studying, utilizing both professor-provided materials, resources from fellow students and ChatGPT. They are not intended to be perfect and may be inaccurate and incomplete. If you see any error feel free to open an issue or send a pull request.

## Main features 

- Every course note is divided in macro subjects plain markdown notes but it has also a PDF file.
- Topics are **interconnected** through **markdown links**, allowing for easy navigation and exploration of related topics directly from Github (*yeah still not enough connection .. Work In Progress*).
- **Courses marked with WIP are incomplete** as I am still working on them, studying them, or procrastinating the note refactoring.
- If you open this repository with Obsidian.md (or similar softwares) you also have a one-click graph visualization of the notes interconnections.

## Courses index

````
├── MSc(english) (WIP) 
│   ├── Advanced Algorithms and Parallel Programming
│   ├── Advanced Computer Architectures
│   ├── Advanced OS (WIP)
│   ├── Artificial Neural Networks and Deep Learning (WIP)
│   ├── Complessitá dei sistemi e delle reti (WIP)
│   ├── Computer Graphics(WIP)
│   ├── Computer Security (WIP)
│   ├── Computing Infrastructures (WIP)
│   ├── Databases 2
│   ├── Design and Implementation of Mobile Apps (WIP)
│   ├── Distributed Systems (WIP)
│   ├── Formal Languages and Compilers (WIP)
│   ├── Foundations of Operation Research (WIP)
│   ├── Hypermedia Applications (WIP)
│   ├── Machine Learning (WIP)
│   ├── Philosophical Issues of Computer Science (WIP)
│   ├── Principles of Programming Languages(WIP)
│   ├── Software Engineering 2
│   └── Videogame Design and Programming (WIP)
│
├── BSc(italian)
│   ├── Algoritmi e Principi dell'Informatica
│   ├── Analisi 2 (WIP) 
│   ├── Basi di Dati
│   ├── Economia e Organizzazione Aziendale
│   ├── Fondamenti di Automatica
│   ├── Fondamenti di Elettronica
│   ├── Foundations of Artificial Intelligence
│   ├── Ingegneria del Software
│   ├── Meccanica
│   ├── Probabilità e Statistica per l'Informatica
│   ├── Progetto di Ingegneria Informatica
│   ├── Progetto di Ingegneria del Software
│   ├── Relazione Progetto Reti Logiche
│   ├── Reti Logiche
│   └── Sistemi Informativi
````

## Generic course folder 

````
.
├── course_name.md // index of course topics
├── build_course.sh // script to build PDF file **ignore**
├── course_name.pdf // PDF generated from the md notes 
└── src 
    ├── 00.Header.md //necessary to make index in PDF **ignore**
    ├── ... .md //single topic/chapter note
    ....
    └── images 
        ├── ... .png 
        ├── ... .jpg
        ....
````
