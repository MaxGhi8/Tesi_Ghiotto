In questa cartella ci sono i seguenti contenuti:

- Darcy_convFNO.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di Darcy 2D

- folder data --> cartella con coefficienti (valori di a) e soluzioni (u) per allenare la rete.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- Darcy_convFNO_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- model_Darcy_convolutional --> 1000 training instances, 200 test instances, risoluzione con mesh 85*85 (s=5) sia train che test
		 Adam optimizer, learning rate iniziale 0.001, 500 epoche, cosine annealing scheduler
		 k_{max, j} = 12, d_{v} = 32, L=4
		 P convol networks kernel_size=3 e padding=1, W è trasformazione affine, R lineare e Q shallow NN
		 Training loss norma L2 relativa
		 Gelu activation function
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {err_train = 0.07073 , err_test_L2 = 0.07524, err_test_H1 = 0.1648 }

	- model_Darcy_convolutional_lr5e-4 --> come model_Darcy_convolutional ma con l.r. = 0.0005 
		 {err_train = 0.04285, err_test_L2 = 0.04666, err_test_H1 = 0.1253}


E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		