In questa cartella ci sono i seguenti contenuti:

- DarcyFFNO.py --> file di codice python utilizzato per creare una rete FFNO per risolvere il problema di Darcy 2D
		   articolo --> "Factorize Fourier Neural Network", Alasdair Tran (drive)

- folder data --> cartella con coefficienti (valori di a) e soluzioni (u) per allenare la rete.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- DarcyFFNO_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- FFNO --> 1000 training instances, 200 test instances, risoluzione con mesh 85*85 (s=5) sia train che test
		 Adam optimizer, learning rate iniziale 0.001, 500 epoche, cosine annealing scheduler
		 k_{max, j} = 20 (cioè ho tenuto circa la prima metà), d_{v} = 32, L=4
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {train_loss = 0.01608, test_loss = 0.01861}

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		