In questa cartella ci sono i seguenti contenuti:

- BurgerFNO.py --> file di codice python utilizzato per creare una rete FNO per risolvere Burger equation

- folder data --> contiene 1000 esempi con i coefficienti 'a' e le corrispettive soluzioni 'u' su una
		  griglia da 8192 punti.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file BurgerFNO.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- BurgerFNO_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- exp_burger --> FNO con 900 training instances, 100 test instances, risoluzione con mesh 1024 (s=8) sia train che test,
		 ReLU activation function
		 Adam optimization, learning rate iniziale 0.001, 200 epoche, l.r. dimezzato ogni 100 epoche
		 k_{max, j} = 16, d_{v} = 64, L=4
		 P, Q e W sono trasformazioni affini e R lineare
		 Training loss norma L2 relativa
		 Il dato iniziale è stato diviso in modo che ogni features sia una normale di media 0 e varianza 1.
		 {err_train = 0.007319, err_test = 0.009413 }

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		