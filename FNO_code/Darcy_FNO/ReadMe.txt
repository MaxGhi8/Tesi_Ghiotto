In questa cartella ci sono i seguenti contenuti:

- DarcyFNO.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di Darcy 2D

- folder data --> cartella con coefficienti (valori di a) e soluzioni (u) per allenare la rete.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- DarcyFNO_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- L4 --> FNO con 1000 training instances, 200 test instances, risoluzione con mesh 85*85 (s=5) sia train che test,
		 ReLU activation function
		 Adam optimization, learning rate iniziale 0.001, 200 epoche, l.r. dimezzato ogni 100 epoche
		 k_{max, j} = 12, d_{v} = 32, L=4
		 P, Q e W sono trasformazioni affini e R lineare
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {train_err = 0.02884, test_err=0.03168}

	- L6 --> come gauss_4 ma con L = 6
		 {train_err = 0.04484, test_err=0.04724}

	- L8 --> come gauss_4 ma con L = 8
		 {train_err = 0.05661, test_err=0.05793}

	- L12 --> come gauss_4 ma con L = 12
		  {train_err = 0.06094, test_err=0.06154}
	

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		