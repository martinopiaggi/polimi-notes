# Sorting

### Proprietà di stabilità 

Se un algoritmo di sorting è stabile allora se nell'array da ordinare ci sono più elementi con lo stesso valore, questi appariranno nell'array ordinato mantenendo il loro ordine relativo iniziale. Questa proprietà non è particolarmente interessante se ci limitiamo ad ordinare numeri. Lo diviene se stiamo ordinando dei oggetti utilizzando un valore dei loro attributi come chiave.

## Insertion sort 
- $T(n) = O(n^2)$ 
- $S(n) = O(1)$
-  array 
-  Non ricorsivo
- Concetto: Dividi l'array in 2 array: uno ordinato e l'altro non ordinato. Inizialmente quello ordinato avrà dimensione 1 ma ad ogni iterazione $k$ lo espandiamo di un elemento e facendo $k-1$ confronti posizioniamento il 'nuovo' elemento nel giusto punto dell'array già ordinato. 
- Recensione: deprecato

## Counting sort 
- $T(n) = O(n+k), k=|alfabeto|$
- $S(n) = O(k)$
- array 
- Non ricorsivo
- Concetto: si salva su un array $k$ contatori. Vengono incrementati i contatori ad ogni occorrenza dopo aver controllato una volta l’array da ordinare $O(n)$, poi partendo dall’inizio del nuovo array stampo $array[i]$ volte il valore $i$.
- Recensione: Ottimo se $k = O(n)$, altrimenti meglio mergesort o heapsort. 

## Merge sort 
- $T(n) = O(n log(n))$ 
- $S(n) = O(n)$ 
- array 
- Ricorsivo (d.e.i.)
- Concetto: divido ogni array in sottoarray e ogni sottoarray ricorsivamente ancora. Il caso base sará un semplice scambio, ogni altro caso invece sará effettuato con un confronto tra indici.
	Passi: 
	1) Controlli che $l$ < $r$ e nel caso trovi la metà, da cui trovi i sotto array
	2) continui in maniera ricorsiva a dividere fino a quando i sottoarray sono a singoli.
	3) a questo punto mergi facendo il confronto e risalendo tutte le chiamate ricorsive. Avrai sempre 2 arrays e li mergi usando 2 indici per ciascun array. Usando i due indici inserisci sempre l'elemento più piccolo tra i 2 valori puntati dagli indici. E dopo averlo inserito incrementi l'indice dell'elemento appena inserito.
- Recensione: Ottimo per la complessità temporale, ma non è in place. Infatti ad ogni divisione si istanziano 2 nuovi array ricorsivamente. 

````Java
private static void mergeSort(int[] array) {		
	int length = array.length;
	if (length <= 1) return; //base case
		
	int middle = length / 2;
	int[] leftArray = new int[middle];
	int[] rightArray = new int[length - middle];
		
		int i = 0; //left array
		int j = 0; //right array
		
	for(; i < length; i++) {
		if(i < middle) {
			leftArray[i] = array[i];
		}
		else {
			rightArray[j] = array[i];
			j++;
		}
	}
	mergeSort(leftArray);
	mergeSort(rightArray);
	merge(leftArray, rightArray, array);
}
	
//notare come è il metodo merge a fare la 'magia'

private static void merge(int[] leftArray, int[] rightArray, int[] array) {
		
	int leftSize = array.length / 2;
	int rightSize = array.length - leftSize;
	int i = 0, l = 0, r = 0; 
		
	//è proprio mentre si fa merge che si scambia
	
	while(l < leftSize && r < rightSize) {
		if(leftArray[l] < rightArray[r]) {
			array[i] = leftArray[l];
			i++;
			l++;
		}
		else {
			array[i] = rightArray[r];
			i++;
			r++;
		}
	}
	
	while(l < leftSize) {
		array[i] = leftArray[l];
		i++;
		l++;
	}
	
	while(r < rightSize) {
		array[i] = rightArray[r];
		i++;
		r++;
	}
}
````

## Heapsort 
- $T(n) = O(n log(n))$
- $S(n) = O(1)$ 
- heap (albero binario) 
- Ricorsivo
- Concetto: costruire un heap con l'algoritmo  ````Heapify(Array)```` ed estrarre uno a uno il massimo 'usando i metodi dell'Heap' : cioé scambiando la 'cima' del mucchio con l'ultimo elemento e 'runnare' la  ````MaxHeapify(Array)````ogni volta. 
- Recensione: Meglio di merge per l’uso della memoria. Non é stabile. 

````Java
HEAPSORT(A){
	BUILD-MAX-HEAP(A) 
	for i := A.length downto 2 
		swap(A[1],A[i]) 
		A.heap-size := A.heap-size - 1 
		MAX-HEAPIFY(A,1)
}
````

## Quicksort 

- $T(n) = O(n^2)$ ma media $\Theta(n log(n))$ 
- $S(n) = O(1)$
- array 
- Ricorsivo 
- Concetto: sposto con uno 'swap' gli elementi piú piccoli del **pivot** (un elemento scelto arbitrariamente quasi) a sinistra, e i piú grossi a destra. Poi divido i due sotto array ricorsivamente, scegliendo per ogni array un pivot. 
- Recensione: Nel caso di molti elementi si converge alla media che è buona. 

````Java
private static void quickSort(int[] array, int start, int end) {
	if(end <= start) return; //base case
		
	int pivot = partition(array, start, end);
	quickSort(array, start, pivot - 1);
	quickSort(array, pivot + 1, end);		
	}
	
private static int partition(int[] array, int start, int end) {
	int pivot = array[end];
	int i = start - 1;
		
	for(int j = start; j <= end; j++) {
		if(array[j] < pivot) {
			i++;
			int temp = array[i];
			array[i] = array[j];
			array[j] = temp;
		}
	}
	i++;
	int temp = array[i];
	array[i] = array[end];
	array[end] = temp;
		
	return i;
	}
}
````

Lo pseudocodice qui sopra utilizza la partizione di Lomuto, la quale seleziona come pivot l'ultimo elemento. Questa in realtà non è la più efficiente. La partizione di Hoare è già più efficiente (anche più intuitiva a mio modesto parere) e pone il pivot al centro dell'array. In media la partizione di Hoare utilizza un terzo degli swaps usati nella partizione di Lomuto.