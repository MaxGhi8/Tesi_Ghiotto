In questa cartella ci sono i seguenti contenuti:

- Elasticity_Omesh_geoFNO.py --> file di codice python utilizzato per creare una rete FNO per risolvere il problema di elasticita'

- folder data --> Come dati di input si prendono le coordinate della mesh che discretizza la airfoil utilizzaando come 
		  discretizzazione euristica la O-mesh ottenuta con una suddivisione uniforme in 64 punti per
		  l'angolo e di 41 valori equispaziati tra i valori ammissibili per il raggio. L'output e'
		  il valore dello stress del materiale.

- model_... --> file con salvati i parametri della rete dopo averla allenata con il codice del file Plasticity_geoFNO.py. 
		Al posto dei tre punti di sospensione ci va un "codice" per riconoscere i diversi tipi di architetture di rete.

- exp_... --> folder con qualche plot delle cose interessanti ottenute salvando la rete (test_loss, train_loss, plot).
	      I tre punti di sospensione è lo stesso codice di model_...
	      Per vedere i grafici si deve aprire il cmd nella cartella in cui è salvata exp_..., scrivere:
		tensorboard --logdir=exp_...
	      E aprire su un browser il link che compare sul terminale

- Elasticity_Omesh_geoFNO_Trained.py --> codice per fare prove con le reti già allenate, utilizzando i parametri contenuti nei model_...

- Didascalia ... =
	- exp_plasticity --> FNO con 1000 training instances, 200 test instances, risoluzione originale (64*41)
		 GeLU activation function
		 Adam optimization, learning rate iniziale 0.001, 500 epoche, CosineAnnealingScheduler
		 k_{max, j} = 12, d_{v} = 32, L=4
		 P, W sono trasformazioni affini, R lineare e Q shallow NN
		 Training loss norma L2 relativa
		 {train_loss = , test_loss_L2 = , test_loss_H1 = }
	

E' meglio salvare tutti i file sopra elencati nella stessa cartella per non dovere cambiare i path nel file python
		