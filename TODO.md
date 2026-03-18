# TODO

### Datasets
Rivedere il codice di:
- [x] build_dataloader
- [ ] CIFAR  
- [ ] CUB-200-2011
- [ ] Aircraft  
- [ ] iNat
Da implementare:
- [x] create_transform di timm usata da H-CAST
- [ ] Rivedere trasformazioni e data augmentation 

### Train
Rivedere
- [ ] engine
    - [x] check funzionalità
    - [ ] funzionamento barra caricamento
- [ ] eval
- [x] metrics (ci sarebbe anche da decidere le metriche)
- [x] train
- [x] utils
- [x] serve calcolare le metriche nel train set? no ma aiuta a vedere l'overfitting
- [x] build_optimizer senza timm
- [x] build_scheduler con timm

### Preprocessing e trasformazioni
- [ ] Rivedere caricamento pretrained model
- [ ] ModelEma
- [ ] RandomErase

### Riproducibilità e tuning
- [ ] Salvare configs nell'output, così da sapere cosa ha generato quei risultati

### Leggibilità
- [ ] logger

# H-CAST
- [x] Check loss
    - [x] gk_loss negativa a volte? risolto
- [ ] Ricontrollare iperparametri
- [x] aggiungere SEEDS hyperpixels

# HT-CapsNet
- [ ] Sostituire backbone generico con EfficientNetB7
- [ ] Check di tutte le dimensioni