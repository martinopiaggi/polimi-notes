# Automi a Pila

Un AP è una tupla $<Q, I, \Gamma, \delta, q_0, Z_0, F>$: 

- Q , un insieme di stati finito
- I , un alfabeto d'ingresso
- $\Gamma$ alfabeto di pila
- $\delta$ , una funzione $\delta (Q,I) \rightarrow Q$
- $q_0$ , stato iniziale (unico)
- $Z_0$ inizio pila
- F, l'insieme di stati finali


Stessa cosa degli FSA ma dotati di una pila infinita con però un inizio. Possiamo impilare quello che vogliamo, ma possiamo operare solo sulla cima. Una memoria così distruttiva.. lettura = distruzione. 

AP $\leftarrow$ PDA = Push-Down Automaton 

L'AP non è chiuso rispetto nè l'unione nè l'intersezione. Però il complemento è chiuso(procedura analoga al FSA, ma più complicata causa la possibilità, con gli AP, di poter accettare le $\epsilon$  stringhe, cioè elementi vuoti).

### La pila può fare $a^nb^nc^md^m$ e $a^n b^mc^md^n$  ma non $a^nb^nc^n$ poichè ha una memoria distruttiva.

Questo è gran sbatti ed è dovuto al funzionamento della pila. L'approccio per risolvere questo genere di problemi è impilare tutti 'i placeholders' che indicana la 'a' e poi nel momento in cui leggo le b far saltare i vari placeholders. Il problema è che se poi ho anche da contare un eventuale carattere c non riesco: ho distrutto tutta la memoria per assicurarmi che b fossero presenti nella stessa quantità delle a. Pazienza abbiamo le **Macchine di Touring.