# TODO

### Datasets
Rivedere il codice di:
- [ ] build_dataloader
- [ ] CIFAR  
- [ ] CUB-200-2011
- [ ] Aircraft  
- [ ] iNat

### Train
Rivedere
- [ ] engine
    - [x] check funzionalità
    - [ ] funzionamento barra caricamento
- [ ] eval
- [ ] metrics (ci sarebbe anche da decidere le metriche)
- [x] train
- [x] utils
- [ ] serve calcolare le metriche nel train set?
- [ ] Definire insiemi di optimizer, scheduler, preprocessing e data augmentation che si utilizzano

### Riproducibilità e tuning
- [ ] Salvare configs nell'output, così da sapere cosa ha generato quei risultati

### Leggibilità
- [ ] logger

# H-CAST
- [ ] Check loss
- [ ] Ricontrollare iperparametri

# HT-CapsNet
- [ ] Sostituire backbone generico con EfficientNetB7
- [ ] Check di tutte le dimensioni