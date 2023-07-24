In questa cartella ci sono i seguenti contenuti:

- TrivialPatchFNO_DarcyL.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di Darcy 2D su dominio ad L
		      Sostanzialmente viene divisa la L in 3 patch e su ognuna di esse viene allenata in modo indipendente una FNO
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

- TrivialPatchFNO_DarcyL_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...


- Didascalia ... =
	- model_patch<i> --> 1000 training instances, 200 test instances, risoluzione con mesh 70*70 (s=3) sia train che test
		 Adam optimizer, learning rate iniziale 0.001, 500 epoche, cosine annealing scheduler
		 k_{max, j} = 12, d_{v} = 32, L=4
		 struttura originaria di Li con L(z) = Wz + b + K(z)
		 padding = 9
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 Uno per patch (i=0, i=1, i=2)
		 i = 0 --> {err_train = 0.05125, err_test_L2 = 0.0824, err_test_H1 = 0.1414}
		 i = 1 --> {err_train = 0.08278, err_test_L2 = 0.1518, err_test_H1 = 0.1888}
		 i = 2 --> {err_train = 0.05037, err_test_L2 = 0.09599, err_test_H1 = 0.1331}

	- model_test_tran<i> --> come il precedente con la differenza che la struttura per l'integral kernel operator 
		 ho utilizzato quella di Tran per le FFNO, ovvero
		 L(z) = z + sigma( W_2*sigma(W_1*K(z) + b_1) + b_2 )
		 i = 0 --> {err_train = 0.05073, err_test_L2 = 0.07786, err_test_H1 = 0.1387}
		 i = 1 --> {err_train = 0.09013, err_test_L2 = 0.1654, err_test_H1 = 0.1798}
		 i = 2 --> {err_train = 0.04595, err_test_L2 = 0.07912, err_test_H1 = 0.1405}

	- model_test_residual<i> --> come il precedente con la differenza che la struttura per l'integral kernel operator 
		 ho utilizzato quella di Tran per le FFNO, ovvero
		 L(z) = sigma(W*K(z) + b)
		 i = 0 --> {err_train = 0.02833, err_test_L2 = 0.08115, err_test_H1 = 0.1493}
		 i = 1 --> {err_train = 0.06254, err_test_L2 = 0.09759, err_test_H1 = 0.1591}
		 i = 2 --> {err_train = , err_test_L2 = , err_test_H1 = } ...da fare...

In generale si nota che nella patch 1 (quella in alto a dx nello spigolo della L) c'è l'errore maggiore.
Anche il coomportamento della loss è molto più oscillante.

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		