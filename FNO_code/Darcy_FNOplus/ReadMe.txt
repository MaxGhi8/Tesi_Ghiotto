In questa cartella ci sono i seguenti contenuti:

- DarcyFNOplus.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di Darcy 2D
		      la base del codice è DarcyFNO.py con qualche accortezza per migliorare il training della rete.
			- Q da traformazione lineare è una rete shallow NN(d_v, 2, [4*d_u, d_u])
			- aggiunto padding per input non periodici
			- aggiunto cosine annealing scheduler
			- Tolto parametri complessi (https://github.com/pytorch/pytorch/issues/59998)
			- Spostato la trasformazione lineare da   Wz + b + K(z)   a   W(K(z))+b, tipo Trasformer
			- norm = 'ortho' for rfft2 e irfft2

- folder data --> cartella con coefficienti (valori di a) e soluzioni (u) per allenare la rete.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- DarcyFNOplus_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...


- DarcyFNOplus_old_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...
				Questo corrisponde alla versione di codice più vecchia in cui non c'era la possibilità
				di fare mulmesh, non c'era la norma H1 e c'èera pointwise fully connected layer invece
				della convoluzione di kernel size = 1

- Didascalia ... =
	- model_plus --> 1000 training instances, 200 test instances, risoluzione con mesh 85*85 (s=5) sia train che test
		 Adam optimizer, learning rate iniziale 0.001, 500 epoche, cosine annealing scheduler
		 k_{max, j} = 12, d_{v} = 32, L=4
		 P, W sono trasformazioni affini, R lineare e Q shallow NN
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {err_train = 0.01152, err_test_L2 = 0.01775, err_test_H1 = 0.1221}

	- model_plus_H1 --> come model_plus come training loss prendo la norma H1 relativa
			    (calcolata facendone un'approssimazione con la trasformata di Fourier)
			    {err_train = 0.1265, err_test_L2 = 0.05247, err_test_H1 = 0.1747}

	- model_plus_xavier --> come model_plus ma con xavier initial normalization per i pesi R
			    {err_train = 0.06228, err_test_L2 = 0.1111, err_test_H1 = 0.1782}

	- model_plus_tanh_H1 --> come model_plus_H1 ma con tanh come funzione di attivazione
			    {err_train = 0.07428, err_test_L2 = 0.04448, err_test_H1 = 0.1144}

	- model_plus_tanh --> come model_plus ma con tanh come funzione di attivazione	
			      {err_train = 0.01519, err_test_L2 = 0.02522, err_test_H1 = 0.0939}
	
	- model_plus_gelu --> come model_plus ma con gelu come funzione di attivazione	
			      {err_train = 0.01709 , err_test_L2 = 0.07606, err_test_H1 = 0.1806}

	- model_plus_mulmesh --> come model_plus ma con la differenza per quanto riguarda la mesh, al posto 
				 di suddividere la mesh in modo uniforme con la stessa griglia ho preso
				 S = [2, 3, 4, 5, 6], cioè prende 200 esempi ognuno con mesh corrispondenti
				 a 211*211, 141*141, 106*106, 85*85, 71*71
				 {err_train = 0.01085 , err_test_L2 = 0.02018, err_test_H1 = 0.0834}

	- model_plus_mulmesh_5e-4lr --> come model_plus_mulmesh ma con l.r. = 5e-4
				 {err_train = 0.01893 , err_test_L2 = 0.05116, err_test_H1 = 0.1786}

	- model_plus_1e-4lr --> come exp_plus_gauss ma con learning rate = 0.0001 = 1e-4
				{err_train = 0.01224, err_test_L2 = 0.02038, err_test_H1 = 0.1243}
	
	- model_plus_1e-4lr_tanh --> come model_plus_1e-4lr ma con tanh come funzione di attivazione
				     {err_train = 0.05331, err_test_L2 = 0.06096, err_test_H1 = 0.1555}
	
	- model_plus_5e-5lr --> come exp_plus_gauss ma con learning rate = 0.00005 = 5e-5
				{err_train = 0.05917, err_test_L2 = 0.09254, err_test_H1 = 0.1828}

	- model_plus_newarc --> trasformazione lineare raddopiata + residuo con la oss di alasdair (articolo FFNO)
				cioè calcola: x + sigma( W * sigma(W * K(x) + b)) + b). Altri parametri come model_plus.
				{err_train = 0.01146, err_test_L2 = 0.01591, err_test_H1 = 0.0979}

	- model_plus_newarc_H1 --> come model_plus_newarc ma con la norma H1 relativa come loss function
				{err_train = 0.5623, err_test_L2 = 0.5571, err_test_H1 = 0.8424}

-	- model_plus_newarc_mulmesh --> come model_plus_newarc ma con la mesh variabile, 200 epochs
				{err_train = 0.1095, err_test_L2 = 0.04806, err_test_H1 = 0.126}

	- model_plus_newarc_2_0 --> trasformazione lineare raddopiata + residuo con la oss di alasdair (articolo FFNO)
				cioè calcola: x + ( W * sigma(W * K(x) + b)) + b); la differenza rispetto a prima è che
				ho tolto la seconda funzione di attivazione. Altri parametri come model_plus.
				{err_train = 0.01754, err_test_L2 = 0.03318 err_test_H1 = 0.1456}

	- model_plus_newarc_2_0_H1 --> come model_plus_newarc_2_0 ma con la norma H1 relativa come loss function
					{err_train = 0.08081, err_test_L2 = 0.03635, err_test_H1 = 0.1061}



E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		