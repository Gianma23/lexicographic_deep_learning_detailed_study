## Capitolo 4 — Methodology

### 4.1 Problem formulation

Problema, notazione e convenzioni comuni ai metodi studiati.

#### 4.1.1 Taxonomy

Tassonomia, relazioni padre-figlio ed etichette come cammini coerenti. Sono
esplicitate le ipotesi strutturali richieste in seguito: ogni nodo non radice ha un
solo padre e ogni nodo non foglia ha almeno un figlio.

#### 4.1.2 Classifier and score conventions

Score per livello. Si distinguono i modelli con teste separate dalla
rappresentazione congiunta dei nodi adottata da Hier-COS.

#### 4.1.3 Training objectives

Loss specifiche delle famiglie e obiettivo scalarizzato usato dalle baseline. I
coefficienti della somma appartengono a questa formulazione e non vengono confusi
con i per-level learning rate della sezione successiva.

#### 4.1.4 Per-level learning rates

Nella modalità lessicografica ogni loss di livello genera un gradiente distinto.
I valori \(\eta_l\) sono quindi presentati come per-level learning rate: regolano
l'intensità con cui ciascun livello partecipa all'aggiornamento, senza definire una
nuova loss scalarizzata.

Nella decomposizione usata per Hier-COS la forma esatta è

$$
g_l(\eta_l)=\nabla_\theta\!\left[\eta_l\,\ell_l^{\mathrm{CE}}
+\alpha\mathcal R_l\right],
\qquad
\Delta\theta=-\mathcal A\!\left(g_1(\eta_1),\ldots,g_L(\eta_L)\right),
$$

dove \(\mathcal A\) indica proiezione e composizione dei gradienti. Il per-level
learning rate scala la componente di classificazione, mentre il coefficiente del
regolarizzatore rimane fissato; questa distinzione va mantenuta anche nel resoconto
degli esperimenti.

Oltre alla configurazione uniforme e alle assegnazioni derivate da Hier-COS, si
studiano due regole ricavate dalla tassonomia:

- **`cumulative_branching`**: \(\eta_l \propto C_l^{\beta}\), con
  \(\beta=0\) che restituisce la configurazione uniforme;
- **`marginal_branching`**: \([1,\ C_2/C_1,\ldots,C_L/C_{L-1}]\), poi
  normalizzato.

Sono regole fissate dalla struttura della tassonomia prima del training. La loro
valutazione resta un'ablazione della modalità lessicografica e non una proposta di
nuova scalarizzazione.

#### 4.1.5 Published Hier-COS node-space objective

Hier-COS non usa soltanto l'etichetta foglia. Per ogni esempio ricostruisce il
cammino corretto e costruisce una distribuzione target sull'insieme di tutti i
nodi. Se \(n_l(y)\) è il nodo corretto al livello \(l\),

$$
q_y(n)=
\begin{cases}
\pi_l, & n=n_l(y),\\
0, & \text{altrimenti},
\end{cases}
\qquad
\pi_l=
\frac{\exp\!\left(2/(L+1-l)\right)}
{\sum_k\exp\!\left(2/(L+1-k)\right)}.
$$

La distribuzione prevista si ottiene con una softmax globale sui valori assoluti
degli score dei nodi. L'obiettivo pubblicato combina la divergenza KL tra target e
predizione con il termine di regolarizzazione geometrica. Vanno poi distinte le
varianti locali con loss decomponibile per livello, necessarie per applicare la
modalità lessicografica o la LH-projection: sono adattamenti sperimentali, non
l'obiettivo originale di Hier-COS. Questa sottosezione descrive la baseline
pubblicata; non coincide con il quarto meccanismo introdotto in 4.4.

#### 4.1.6 The lexicographic problem

Ordinamento coarse-to-fine degli obiettivi e differenza rispetto a una somma
scalarizzata. La stessa priorità verrà realizzata in tre punti diversi del
modello: aggiornamento dei parametri, segnale trasmesso dalle teste e output.

---

### 4.2 Gradient-space lexicographic optimisation

La priorità agisce sull'aggiornamento: i gradienti dei livelli meno prioritari
sono corretti quando entrano in conflitto con quelli dei livelli precedenti.

#### 4.2.1 Setting and requirements

Servono tre loss di livello differenziabili e calcolabili separatamente. Restano
quindi esclusi gli obiettivi che restituiscono un unico scalare accoppiato sui
nodi, come la KL pubblicata di Hier-COS e il termine KL globale di H-CAST.

#### 4.2.2 Parameter blocks and the projection operator

I parametri sono raggruppati in base ai livelli da cui ricevono gradiente. La
proiezione viene definita e applicata sui blocchi condivisi, lasciando inalterati
i parametri raggiunti da una sola loss.

#### 4.2.3 Priority order

`coarse_first` esprime l'ipotesi principale; `fine_first` applica lo stesso
procedimento nell'ordine opposto e funge da controllo.

#### 4.2.4 Guarantee and scope

Garanzia locale, al primo ordine, e limiti dell'interpretazione.

---

### 4.3 LH-projection

La priorità viene introdotta nel backward di ciascuna testa. Nel punto in cui una
testa legge la rappresentazione condivisa, il segnale diretto al tronco viene
privato delle componenti appartenenti allo spazio letto dalle teste superiori.

#### 4.3.1 Setting and requirements

Il metodo richiede un tronco condiviso e teste lineari indipendenti. Per applicarlo
a Hier-COS servono tre modifiche: teste separate, quindi frame identità o
diagonale a blocchi; softmax per livello; rappresentazione più larga del numero di
classi non foglia, altrimenti il segnale del livello fine può annullarsi.

La costruzione viene valutata separatamente dalla modalità lessicografica e da
HCC.

#### 4.3.2 The branch projection

Definizione della proiezione sul ramo di ogni testa e ruolo della derivata
dell'attivazione terminale.

#### 4.3.3 Representation width and advantage parameterisation

Due assi di ablazione: dimensione delle feature e advantage function, nella quale
i logit di un livello sono espressi come residuo rispetto al logit, distaccato, del
genitore predetto.

#### 4.3.4 Guarantee and scope

Condizioni per l'ortogonalità dei gradienti, casi in cui la protezione si
indebolisce e configurazioni non ammissibili.

---

### 4.4 Direct supervision of Hier-COS subspace norms

Il quarto meccanismo elimina la separazione tra la quantità ottimizzata da
Hier-COS e quella usata per predire. Il metodo pubblicato applica la loss agli
score dei nodi e usa i `subspace_norm` come readout; questa variante addestra
invece direttamente una loss sui `subspace_norm`.

Il meccanismo deve ancora essere implementato. La sezione metodologica definitiva
verrà completata insieme all'implementazione, prima di eseguire gli esperimenti.

#### 4.4.1 Motivation and setting

Si formalizza il disallineamento tra obiettivo in node space e regola di readout.
Per ogni nodo \(c\) del livello \(l\), lo score supervisionato diventa

$$
s_l(c)=\lVert\Pi_{S_c}u\rVert_2,
$$

cioè la stessa quantità impiegata dall'inferenza nativa di Hier-COS.

#### 4.4.2 Target assignment and loss

Costruzione dei target gerarchici e loss applicata agli score \(s_l\). Prima del
training vanno fissati in modo esplicito l'ambito della normalizzazione, il ruolo
dei per-level learning rate e l'eventuale mantenimento del regolarizzatore
geometrico. Queste scelte non vengono anticipate nella proposta finché il metodo
non è implementato e verificato.

#### 4.4.3 Relation to the other mechanisms

La supervisione dei `subspace_norm` cambia il segnale ottimizzato ma non proietta
i gradienti e non corregge gli output. Isola quindi l'effetto di allineare training
e readout, mantenendolo distinto da modalità lessicografica, LH-projection e HCC.

#### 4.4.4 Verification and scope

Controlli necessari sull'instradamento dei gradienti attraverso norme e subspazi,
stabilità numerica e compatibilità con le varianti del frame. Le conclusioni
restano limitate a Hier-COS finché il meccanismo non viene definito per altre
famiglie.

---

### 4.5 Hierarchical Constraint Cascade

HCC viene presentato per ultimo perché richiede una costruzione più specifica. La
priorità agisce sugli output mediante una correzione affine ricorsiva nel forward
pass ed è quindi presente anche in inferenza.

#### 4.5.1 Setting and requirements

Score differenziabili per livello, tassonomia completa e ordinamento coerente dei
livelli. Si motiva l'uso di un vincolo hard; la descrizione generale di HardNet
resta nel capitolo di letteratura.

#### 4.5.2 The hierarchical cascade

La proiezione di HardNet-Aff viene applicata lungo le transizioni della gerarchia.
Sono discusse due scelte aggiuntive:

- il detach dell'ancora, che lascia invariato il forward ma impedisce alla loss
  inferiore di modificare direttamente lo score superiore;
- la composizione su più transizioni, assente nella singola applicazione di
  partenza.

Il modello viene addestrato sugli score corretti dalla cascata, così che il vincolo
faccia parte anche del segnale di training.

#### 4.5.3 Guarantee and scope

Soddisfacimento del vincolo su ogni transizione e limiti della cascata. La
correzione assegna uno stesso spostamento ai figli di un dato genitore: non cambia
l'ordine tra fratelli e può ridurre, ma non eliminare, l'inconsistenza. Si riporta
anche la quota di direzioni rimossa nel backward.

---

### 4.6 Hierarchical inference strategies

Passaggio dagli score alle predizioni. Readout, trasformazione e decoder sono
trattati come scelte distinte, per non attribuire al training un effetto prodotto
in inferenza.

#### 4.6.1 Readout rules

`node_score`, usato dalle normali teste di classificazione, e `subspace_norm`,
proprio di Hier-COS. Il secondo può essere calcolato anche su modelli non
addestrati con il frame di Hier-COS, purché siano disponibili le coordinate
necessarie.

#### 4.6.2 Decoding rules

Decodifica indipendente e top-down.

#### 4.6.3 Inference grid and post-hoc evaluation

Un checkpoint viene valutato incrociando:

$$
\{\text{node score},\text{subspace norm}\}
\times\{\text{nessuna trasformazione},\text{HCC}\}
\times\{\text{indipendente},\text{top-down}\}.
$$

Le otto combinazioni sono ottenute dallo stesso forward pass.

---

### 4.7 Comparing the mechanisms

#### 4.7.1 Where each mechanism acts

Tabella di confronto fra punto di intervento, granularità, garanzia e presenza in
inferenza. Le run considerate attivano un solo meccanismo per volta: baseline,
modalità lessicografica, LH-projection, supervisione diretta dei `subspace_norm`
oppure HCC.

#### 4.7.2 Supported model families

Matrice per H-CAST, LH-DNN, HT-CapsNet, HRN e Hier-COS. LH-DNN resta una baseline,
poiché applicargli uno dei meccanismi modificherebbe proprio la struttura che lo
caratterizza. Hier-COS è invece il substrato comune del confronto e l'unica
famiglia sulla quale è definita la supervisione diretta dei `subspace_norm`.

---

## Capitolo 5 — Experiments

### 5.1 Datasets

CIFAR-100, CUB-200-2011 e FGVC-Aircraft, scelti per coprire gerarchie costruite da
superclassi, tassonomie ricostruite e annotazioni di dominio.

#### 5.1.1 Dataset selection

Motivazione della scelta rispetto ai dataset dei lavori di riferimento e alle
proprietà richieste dai quattro meccanismi.

#### 5.1.2 Dataset descriptions

Dimensione, dominio, granularità, split e valore sperimentale.

#### 5.1.3 Hierarchy construction

Origine della tassonomia per ogni dataset, distinguendo annotazioni ufficiali,
ricostruzioni e adattamenti locali.

#### 5.1.4 Dataset analysis

Distribuzione della supervisione, struttura degli alberi e geometria delle
immagini. Il fan-out per transizione serve in particolare a interpretare HCC
(4.5.3).

### 5.2 Metrics

Accuratezza per livello, FPA, weighted AP, AHD e TICE. Per ogni metrica vengono
indicati significato e direzione di miglioramento; i risultati top-down e
independent restano separati.

### 5.3 Fidelity to the published methods

Per ciascun metodo si adottano gli stessi iperparametri del rispettivo paper. La
sezione documenta l'integrazione nel framework comune, gli adattamenti necessari e
le estensioni a impostazioni non presenti nei lavori originali, tra cui HRN su
CIFAR-100, Hier-COS su CUB e la gerarchia CIFAR usata in questa tesi.

Gli studi più estesi di sensibilità agli iperparametri sono mantenuti separati
dalle configurazioni di riferimento; si deciderà in seguito quali includere nel
testo finale.

### 5.4 Experimental design and protocol

#### 5.4.1 Operational research questions

Versione concreta, ancora provvisoria, delle domande introduttive:

| | Domanda operativa |
|---|---|
| **Q1** | La modalità lessicografica migliora accuratezza o coerenza rispetto alla baseline corrispondente? |
| **Q2** | L'effetto dipende dall'ordine coarse-to-fine? |
| **Q3** | Come cambia il risultato intervenendo sull'aggiornamento, sulle diramazioni, sulla supervisione dei `subspace_norm` o sugli output? |
| **Q4** | Quanto del risultato dipende dal training e quanto da readout, trasformazione e decoder? |
| **Q5** | Su quali modelli, dataset e tassonomie si ritrova lo stesso andamento? |
| **Q6** | Quali costi di tempo, memoria e inferenza introduce ciascun metodo? |

La fedeltà delle cinque baseline al protocollo comune è una verifica preliminare,
non una domanda di ricerca autonoma.

#### 5.4.2 Checkpoint selection

Ogni run conserva un checkpoint selezionato con metriche top-down e uno con
metriche independent. Nel confronto finale, ciascuna riga usa il checkpoint scelto
con la stessa modalità di decodifica riportata nella riga.

#### 5.4.3 Protocol and reporting rules

- Un solo meccanismo attivo per run.
- Le modifiche richieste dal modello ospite vengono valutate prima del meccanismo
  che rendono possibile.
- I risultati principali usano il readout nativo della famiglia; gli altri
  readout sono analizzati a parte.
- FPA, weighted AP e accuratezza sono higher-is-better; AHD e TICE
  lower-is-better. Le differenze tra percentuali sono espresse in punti
  percentuali.
- Per ogni aggregato si riportano media, deviazione standard e numero di seed. Le
  differenze inferiori alla dispersione tra seed non vengono interpretate come
  evidenza di un effetto.

#### 5.4.4 Organisation of the experimental campaign

Inventario delle run, numero di seed e celle escluse per costruzione.

---

### 5.5 Baselines under the unified protocol

I cinque modelli senza meccanismi aggiuntivi. Per Hier-COS la baseline usa il
target sul cammino e l'obiettivo in node space descritti in 4.1.5; non usa ancora
la loss sui `subspace_norm`.

### 5.6 Inference: readout, transform and decoder — *Q4*

I checkpoint congelati vengono valutati nella griglia di 4.6. Questa analisi non
richiede nuovo training e stabilisce come leggere i risultati successivi.

#### 5.6.1 Independent against top-down decoding

#### 5.6.2 Node score against subspace norm

#### 5.6.3 HCC applied post hoc

La cascata viene applicata a checkpoint addestrati senza HCC, isolandone l'effetto
come trasformazione degli output.

### 5.7 Gradient-space lexicographic optimisation — *Q1, Q2*

#### 5.7.1 Effect against the matched baseline

Le run principali usano per-level learning rate uniformi. Le altre assegnazioni
sono studiate separatamente in 5.7.3.

#### 5.7.2 Does the ordering matter?

Confronto `coarse_first` contro `fine_first`. Un controllo scalarizzato assegna un
coefficiente maggiore alla loss coarse senza proiettare i gradienti, per verificare
se basti privilegiare quel livello. Poiché `fine_first` è il solo controllo che
inverte direttamente l'ordine, le conclusioni su Q2 dovranno restare proporzionate
a questa copertura.

#### 5.7.3 Per-level learning rates

Ablazione dei moltiplicatori definiti in 4.1.4 rispetto alla configurazione
uniforme. Hier-COS è usato come substrato comune perché espone la decomposizione
necessaria e tutte le varianti considerate; la validità al di fuori di questa
famiglia non viene assunta.

#### 5.7.4 Gradient diagnostics

Norme, coseni, frequenza di attivazione e componenti rimosse, per verificare che
l'operatore agisca come previsto.

### 5.8 Preconditions for LH-projection on Hier-COS — *Q5*

Prima di misurare LH-projection si valuta il costo delle modifiche che la rendono
definibile su Hier-COS.

#### 5.8.1 Orthonormal frame variants

Frame ortonormale random, diagonale a blocchi random e identità.

#### 5.8.2 Global against per-level softmax

### 5.9 LH-projection on Hier-COS — *Q1, Q6*

#### 5.9.1 Effect against the matched baseline

#### 5.9.2 Comparison with gradient-space optimisation

#### 5.9.3 Representation width and advantage parameterisation

Le due ablazioni introdotte in 4.3.3.

### 5.10 Direct supervision of Hier-COS subspace norms — *Q3, Q4*

Sezione pianificata, subordinata all'implementazione del meccanismo descritto in
4.4. Non vengono previsti risultati finché loss, normalizzazione e controlli sui
gradienti non sono definiti e testati.

#### 5.10.1 Implementation verification

Verifica che la loss sia calcolata sui `subspace_norm`, che il gradiente raggiunga
la trasformazione apprendibile e che il readout usato nel training coincida con
quello valutato.

#### 5.10.2 Effect against native Hier-COS

Confronto con la baseline Hier-COS in node space, a parità di backbone, frame e
protocollo.

#### 5.10.3 Interaction with inference

Confronto tra readout nativo e controlli della griglia post-hoc, per stabilire se
l'allineamento tra loss e `subspace_norm` modifica la rappresentazione appresa o
soltanto la calibrazione degli score.

### 5.11 Hierarchical Constraint Cascade — *Q1, Q4*

HCC è l'ultimo approccio analizzato, coerentemente con l'ordine del Capitolo 4.

#### 5.11.1 Effect against the matched baseline

#### 5.11.2 Training-time against post-hoc enforcement

Confronto tra HCC attivo nel training e la trasformazione post-hoc di 5.6.3.

#### 5.11.3 Constraint diagnostics

Residui prima e dopo la correzione, spostamenti degli score, cambi di argmax e
massa assegnata al genitore corretto. L'attivazione viene verificata dai log, non
dedotta dal nome della run.

#### 5.11.4 Interaction with taxonomy shape

Risultati messi in relazione con fan-out, granularità e difficoltà dei livelli.

### 5.12 Comparing the four mechanisms — *Q3, Q6*

Confronto tra effetto misurato e punto di intervento. Si riportano inoltre tempo,
memoria, numero di backward pass e costo in inferenza.

### 5.13 Summary and coverage

#### 5.13.1 Answers supported by the experiments

Risposte alle domande operative, senza introdurre nuove analisi.

#### 5.13.2 Limits of the experimental coverage

Esperimenti non eseguiti e conseguenze sull'interpretazione. In particolare, i
per-level learning rate ricavati dalla tassonomia sono valutati soltanto in
modalità lessicografica e su Hier-COS: l'esperimento misura la sensibilità di
quella configurazione, non stabilisce una regola generale per le altre famiglie.
La supervisione diretta dei `subspace_norm` resta fuori dalle conclusioni finché
non sono disponibili implementazione verificata e run complete.
