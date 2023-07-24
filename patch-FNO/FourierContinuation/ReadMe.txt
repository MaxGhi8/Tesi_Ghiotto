In questa cartella ci sono i seguenti contenuti:

- FNOcontinuation_DarcyL.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di Darcy 2D su dominio ad L
		      Sostanzialmente la L viene "paddata" ad un quadrato [-1,1]^2 e poi lo tratto con una FNO classica
			- Q è una rete shallow NN(d_v, 2, [4*d_u, d_u])
			- aggiunto padding per input non periodici
			- aggiunto cosine annealing scheduler
			- Tolto parametri complessi (https://github.com/pytorch/pytorch/issues/59998)

- folder data --> cartella con coefficienti (valori di a) e soluzioni (u) per allenare la rete.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Darcy.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- FNOcontinuation_DarcyL_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...


- Didascalia ... =
	- model_FNOcontinuation_Li --> 1000 training instances, 200 test instances, risoluzione con mesh 142*142 (s=3) sia train che test
		 Adam optimizer, learning rate iniziale 0.001, 500 epoche, cosine annealing scheduler
		 k_{max, j} = 12, d_{v} = 32, L=4
		 struttura originaria di Li con L(z) = Wz + b + K(z)
		 padding = 0, Xavier initial normalization and FFTnorm = None
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {err_train = 0.05164, err_test_L2 = 0.05469, err_test_H1 = 0.1388}

	- model_FNOcontinuation_Tran --> come il precedente con la differenza che la struttura per l'integral kernel operator 
		 ho utilizzato quella di Tran per le FFNO and FFTnorm = 'ortho' , ovvero
		 L(z) = z + sigma( W_2*sigma(W_1*K(z) + b_1) + b_2 )
		 {err_train = 0.02842, err_test_L2 = 0.05962, err_test_H1 = 0.1577}	

	- model_FNOcontinuation_Tran_H1 --> come il precedente ma con train_loss norma H1
		 {err_train = 0.06662, err_test_L2 = 0.07732, err_test_H1 = 0.1296}


E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		