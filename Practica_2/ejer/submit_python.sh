#! /bin/bash
#$ -N DNNP2
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola de la GPU del nodo 6-2
##$ -q gpu@compute-6-2.local
##  pido la cola de las GPUs
#$ -q cpu
## pido una placa
##$ -l gpu=1
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda
module load miniconda

source activate deep_learning

# Execute the script
hostname

echo "Ejer 3"
python ejer3.py

echo "Ejer 4"
python ejer4.py

echo "Ejer 5"
python ejer5.py
