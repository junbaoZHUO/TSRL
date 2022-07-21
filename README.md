# Codes release for our paper "Zero-shot Video Classification with Appropriate Web and Task Knowledge Transfer"


## Enviroment for GoogLeNet based implementation
python 3.8.10<br />
pytorch 1.9.0<br />
torchvision 0.10.0<br />
numpy 1.20.3<br />
pillow 8.4.0<br />
networkx 2.6.3<br />
pot 0.8.1.0<br />

## Enviroment for CSWin based implementation
python 3.7.11<br />
pytorch 1.4.0<br />
torchvision 0.5.0<br />
numpy 1.20.3<br />
pillow 8.4.0<br />
networkx 2.6.3<br />
pot 0.8.2<br />



## Training scripts for GoogLeNet based implementation
### --setting: inductive or transductive or generalized_transductive
### --mode: INIT or GCN_ONLY or GCN+INIT or GCN+INIT+SIM
```
python train.py  --gpu 3 --learning_rate 0.00002 --split_ind 0 --datadir oly --dataset oly --label_num 16 --num_class 163 --batch_size 128 --epoch 5 --mode GCN+INIT+SIM
```
```
python train.py  --gpu 6 --learning_rate 0.000015 --split_ind 0 --datadir ucf101 --dataset ucf101 --label_num 101 --num_class 1320 --batch_size 128 --epoch 5 --mode GCN+INIT+SIM --setting transductive --bnm 10
```
```
python train.py  --gpu 7 --learning_rate 0.0001 --split_ind 0 --datadir hmdb51 --dataset hmdb51 --label_num 51 --num_class 847 --batch_size 128 --epoch 5 --mode GCN+INIT+SIM --setting generalized_transductive --bnm 50
```
```
python train.py  --gpu 5 --learning_rate 0.0005 --split_ind 0 --datadir fcvid --dataset fcvid --label_num 238 --num_class 2066 --batch_size 128 --epoch 20
```

## Training scripts for CSWin based implementation
### --setting: inductive or transductive or generalized_transductive
### --mode: INIT or GCN_ONLY or GCN+INIT or GCN+INIT+SIM
```
python train.py --dataset oly --gpu 0 --datadir oly_cs --split_all 50  --topK 15  --learning_rate 0.0003  --label_num 16 --num_class 140 --batch_size 256 --epoch 10 --weight_decay 0.0001 --mode GCN+INIT+SIM
```
```
python train.py --topK 10 --gpu 2 --learning_rate 0.001 --datadir hmdb51_cs --dataset hmdb51 --label_num 51 --num_class 361 --batch_size 256 --epoch 10 --mode GCN+INIT+SIM --setting transductive --bnm 15
```

```
python train.py --topK 15 --gpu 1 --dataset ucf101 --learning_rate 0.00001 --datadir ucf101_cs --label_num 101 --num_class 2546 --batch_size 1024 --epoch 5  --split_all 50 --bnm 50 --setting generalized_transductive --mode GCN+INIT+SIM --weight_decay 0.0001
```

```
python train.py --topK 10 --gpu 3 --dataset fcvid --learning_rate 0.00001 --datadir fcvid_cs --label_num 238 --num_class 1680 --batch_size 1408 --epoch 5  --split_all 1  --weight_decay 0.0001 --mode GCN+INIT+SIM
```
