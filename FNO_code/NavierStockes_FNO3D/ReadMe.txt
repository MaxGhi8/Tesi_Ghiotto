In questa cartella ci sono i seguenti contenuti:

- NavierStockes_FNO3D.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema
			     di Navier Stockes in 3D. Ovvero si prende come griglia una discretizzazione dello spazio xy
			     e anche del tempo. Come input si prendono i valori della soluzione nei primi T_in tempi
			     e la discretizzazione della griglia (u_1(x,y), ... , u_{T_in}(x,y), x, y, t) e come output
			     si vuole predire le soluzioni per i successivi T tempi in una volta sola.
			     

- folder data --> c'è un unico file con 5000 esempi, quindi è da dividere in test e training.
		  In ogni esempio c'è la valutazione della soluzione per 50 valori dei tempi, quindi per
		  dato iniziale prendo la soluzione per i primi 10 tempi, mentre come soluzione prendo gli ultimi 40 tempi

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- NavierStockes_FNO3D_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- ...... --> FNO con 1000 training instances, 200 test instances, risoluzione con mesh 64*64 (s=1) sia train che test,
		 ReLU activation function
		 Adam optimization, learning rate iniziale 0.001, ...... epoche, l.r. dimezzato ogni 100 epoche
		 k_{max, j} = 8, d_{v} = 20, L=4
		 P, Q e W sono trasformazioni affini e R lineare
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1
	

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		