# Strutture dati

## Array
Pila/stack , code/queue , lista doppiamente concatenata.

Complessità liste: 

- Search $T(n)=O(n)$
- Insert $T(n)=O(1)$
- Delete $T(n)=O(1)$

Quando usare array:

 - Conosco la dimensione in partenza
 - Mi serve accesso diretto di valori vicini
 - Il merge di k array ordinati di n elementi mi costa: $O(nk^2)$ : scorro tutte le teste degli array O(k), trovo il min e lo aggiungo l’elemento trovato al nuovo array. 

Quando usare liste:

- Il costo della costruzione non mi interessa (per liste ordinate)
- Voglio accedere sempre il min/max (liste ordinate), il primo/ultimo inserito (queue/stack)

## Dizionari 
Agli oggetti di un dizionario si accede tramite le chiavi, che sono numeri interi. Implementare un dizionario è molto semplice: vettore di $n$ bit in cui inserisco $1$ o meno all' $n$-esimo bit se il corrispondente numero è presente nel dizionario.

Caratteristiche:

- molto semplici
- $OR$, $AND$ etc. sui bit mi permette di unire insiemi etc.
- dizionari però **sprecano spazio se gli elementi non sono continui**.

Complessità dizionari: 

- Search $T(n)=O(1)$
- Insert $T(n)=O(1)$
- Delete $T(n)=O(1)$


## Tabelle di Hashing 

Utilizzate da Python per implementare ad esempio i dizionari. Il concetto: data una chiave $k$, so la posizione in cui si trova nel mio array tale chiave e ci accedo direttamente.  Dentro ogni cella vuota, mi serve un valore di garbage, che non deve appartenere al dominio delle mie key! Quindi il vettore iniziale lo inizializzo con i simboli di garbage.

Complessità Tabelle di Hashing:

- Search di k a tempo proporzionale alla lunghezza della lista di chiavi nella posizione f(k).
- Insert $O(1)$ (se lo slot non è già occupato, altrimenti come il search)
- Delete $O(1)$ nel caso di liste doppiamente concatenate, altrimenti se concatenate solo da un lato è proporzionale alla lunghezza della lista di chiavi nella posizione f(k).

Quando usare hashing table:

- Voglio accesso diretto a elementi sparsi o non accessibili facilmente (n numeri in un intervallo >> n, stringhe, altri dati non facilmente indicizzabili)

- Le keywords "elementi distribuiti uniformemente", "accesso diretto" e "caso medio" implicano spesso l’utilizzo di una hast table.

Due tipi di implementazioni: 

- Tabelle di hashing ad indirizzamento aperto (closed hashing)
	I dati stanno sempre dentro, la tabella è quindi 'chiusa' e non ci sono liste che 'escono' da tale tabella. Per questo motivo l'indirizzamento è aperto, infatti una chiave non è per forza indirizzata in un indirizzo specifico ma può l'indirizzamento è aperto a 'qualsiasi' cella della tabella. 
	
- Tabelle di hashing ad indirizzamento chiuso (open hashing)
	" 'open' perchè la tabella è aperta letteralmente e dal vettore 'escono' le  liste per ciascuna cella nel caso di sovrapposizione di chiavi". Di conseguenza si dice 'indirizzamento chiuso' perchè ogni chiave è indirizzata sempre e comunque ad una sola cella (chiuso in questo senso) e al più non sarà la prima della lista ma sarà in quella lista. 

### Tabelle di hashing ad indirizzamento chiuso (open):

Calcolo h(k) che mi restituisce una cella.
Collisione? $\rightarrow$ concateno: gli oggetti che vengono mappati sullo stresso slot vengono messi in una lista concatenata.

### Tabelle di hashing ad indirizzamento aperto (closed):

Calcolo $h(k)$, ovvero una funzione che data $k$ mi restituisce una cella. Collisione? $\rightarrow$ ispeziono con una **tecnica di ispezione**, cioè cerco un'altra cella con una formula. 

**Allocare per aggiungere liste è più dispendioso di semplici ispezioni**. 
Problema delle ispezioni: 


- ci potrebbe essere il clustering (raggruppamento): se continuo ad avere collisioni negli stessi punti, finisco che poi avrò continue collisioni nei posti vicini, creando un effetto a catena.
- La cancellazione in caso di indirizzamento aperto è un po' più complicata, in quanto non possiamo limitarci a mettere lo slot desiderato a $NIL$, altrimenti non riusciremmo più a trovare le chiavi inserite dopo quella cancellata. Una soluzione è quella di mettere nello slot, invece che $NIL$, un valore convenzionale: una "tombstone" .  

#### Hashing 

Esistono diverse funzioni di hashing e di ispezione.
Calcolo $h(k)$, ovvero una funzione che data $k$ mi restituisce una cella. Diversi metodi di hashing:

- Metodo della divisione
	$$h(k) = (k)mod(m)$$ 
	Facile da realizzare e veloce ma con l'accortezza di sui valori di $m$: 
		- evitare potenze di 2   
		- $m$ un numero primo non vicino ad una potenza esatta di 2. Per esempio $m$=701 ci darebbe, se n=2000, in media 3 elementi per lista concatenata.
	
- Metodo della moltiplicazione
	 $$h(k) = \lfloor m((kA)mod(1)) \rfloor$$dove $(x)mod(1)$ è la parte frazionaria di $x$ . Spesso come $m$ si prende una potenza di 2 (per 'facilitare' i conti per il calcolatore) ed è utile prendere come A il valore proposto da Knuth: 
	 $$A = \frac{\sqrt{5} -1}{2}$$

#### Tecniche di ispezione

Tecniche utilizzare in caso di collisioni. 

Tre tecniche: 

- ispezione lineare (linear probing)
	linear probing, dopo la prima collisione per evitare clustering si fa $h(k,i)=(h(k,0) + i)mod(m)$  come candidato per l' $i$-esimo inserimento. **Soffre di clustering primario**: lunghe sequenze di celle occupate consecutive, che rallentano la ricerca.

- ispezione quadratica 
	$h(k,i)=(h(k,0)+c_{1}i^2+c_{2}i)mod(m)$ per evitare clustering banale all'intorno di alcuni elementi. Non è più garantito però che la sequenza di ispezioni tocchi tutte le celle. 
	
- doppio hashing (double hashing)
	$h(k,i)=(h_1(k)+h_2(k)i )mod(m)$ dove $h_1$ e $h_2$ sono funzioni hash di supporto.  $h_2$ deve generare solo numeri dispari e che non mai $0$ (per non avere una sequenza di ispezione degenere). 


NOTA: *nessuna di queste tecniche produce le giuste permutazioni necessarie a soddisfare l'ispezione uniforme/perfetta: cioè in pratica visitare tutte le celle una ad una senza mai ripeterle... tuttavia, nella pratica si rivelano efficaci.*  


### Analisi di complessità e fattore di carico

#### Indirizzamento chiuso 
Con HashTable con indirizzamento chiuso, nel caso pessimo in cui tutti gli $n$ elementi memorizzati finiscono nello stesso slot la complessità è quella di una ricerca in una lista di $n$ elementi, cioè $O(n)$. In **media**, però, le cose non vanno così male.

Dati $m$ la dimensione della tabella ed $n$ il numero di elementi e $\alpha$ il fattore di carico, $\alpha = \frac{n}{m}$  . Siccome $0 \le n \le |U|$ avremo $0 \le \alpha \le \frac{|U|}{m}$ .

Ogni chiave ha la stessa probabilità $\frac{1}{m}$  di finire in una qualsiasi delle $m$ celle, indipendentemente dalle chiavi precedentemente inserite.  Quindi la lunghezza media di una lista è il tempo medio per cercare una chiave $k$ non presente nella lista: $\Theta (1+\alpha)$ dove $O(1)$ è il tempo per calcolare $h(k)$, che si suppone sia costante.

Quindi la complessità temporale è $O(1)$ per tutte le operazioni (INSERT, SEARCH, DELETE) . 


#### Indirizzamento aperto 


Dato $\alpha = \frac{n}{m}$ , siccome abbiamo al massimo un oggetto per slot della tabella, $n \le m$, e $0 \le a \le 1$ . 

Il numero **medio** di ispezioni necessarie per effettuare l'**inserimento** di un nuovo oggetto nella tabella è $m$ se $a = 1$ (se la tabella è piena), e non più di $\frac{1}{(1-a)}$ se $a<1$ .

Il numero **medio** di ispezioni necessarie per trovare un elemento presente in tabella è $(m+1)/2$ se $a=1$, e non più di $\frac{1}{a} log(\frac{1}{(1-a)})$ se $a<1$. 
