#!/bin/bash

echo "==========================="
echo "Experimento 1: FedAvg: CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedavg" --baseline --trial 1 --seed 42

echo "==========================="
echo "Experimento 2: FLEG Full - CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 100 --gan_epochs 30 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 3: FLEG Smart - CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 10 --gan_epochs 35 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 4: FLEG Eco - CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 5: FedProx: CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedprox" --baseline --trial 1 --seed 42

echo "==========================="
echo "Experimento 6: FLEG Full + FedProx - CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedprox" --num_chunks 100 --gan_epochs 30 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 7: FLEG Eco + FedProx - CIFAR-10, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "ClassPartitioner" --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 8: FedAvg: CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --baseline --trial 2 --seed 30

echo "==========================="
echo "Experimento 9: FLEG Full - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 100 --gan_epochs 30 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 10: FLEG Smart - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 10 --gan_epochs 35 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 11: FLEG Eco - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 12: FedProx: CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 13: FLEG Full + FedProx - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 100 --gan_epochs 30 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 14: FLEG Smart + FedProx - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 10 --gan_epochs 35 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 15: FLEG Eco + FedProx - CIFAR-10, Dir01"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 16: FedAvg: CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --baseline --trial 1 --seed 42

echo "==========================="
echo "Experimento 17: FLEG Smart - CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 18: FLEG Eco - CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 19: FLEG Full - CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --num_chunks 50 --gan_epochs 25 --num_syn "fixed" --trial 1 --seed 42

echo "==========================="
echo "Experimento 20: FedProx: CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 21: FLEG Smart + FedProx - CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 22: FLEG Eco + FedProx - CIFAR-10, Dir05"
echo "==========================="
python ./FLEG.py --dataset "cifar10" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 23: FedAvg: MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedavg" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 24: FLEG Full - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 100 --gan_epochs 25 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 25: FLEG Smart - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 26: FLEG Eco - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 27: FedProx: MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedprox" --baseline --trial 2 --seed 30

echo "==========================="
echo "Experimento 28: FLEG Full + FedProx - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedprox" --num_chunks 100 --gan_epochs 25 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 29: FLEG Smart + FedProx - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedprox" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 30: FLEG Eco + FedProx - MNIST, ClassPartition"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "ClassPartitioner" --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 31: FedAvg: MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 32: FLEG Full - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 100 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 33: FLEG Smart - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 3 --seed 20

echo "==========================="
echo "Experimento 34: FLEG Eco - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 35: FedProx: MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 36: FLEG Full + FedProx - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 100 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 37: FLEG Smart + FedProx - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 10 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 38: FLEG Eco + FedProx - MNIST, Dir01"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.1 --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 39: FLEG Full - MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --num_chunks 100 --gan_epochs 20 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 40: FedAvg: MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 41: FLEG Eco - MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedavg" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Experimento 42: FLEG Full + FedProx - MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --num_chunks 100 --gan_epochs 20 --num_syn "dynamic" --trial 1 --seed 42

echo "==========================="
echo "Experimento 43: FedProx: MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --baseline --trial 3 --seed 20

echo "==========================="
echo "Experimento 44: FLEG Eco + FedProx - MNIST, Dir05"
echo "==========================="
python ./FLEG.py --dataset "mnist" --partitioner "Dirichlet" --alpha 0.5 --strategy "fedprox" --num_chunks 1 --gan_epochs 25 --num_syn "dynamic" --trial 2 --seed 30

echo "==========================="
echo "Todos os experiemntos foram executados!"
echo "==========================="