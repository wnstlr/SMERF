#!/bin/bash
for i in {1..13}
do
    python run_experiment.py --exp 1.11;
done

for i in {1..9}
do
    python run_experiment.py --exp 2.11;
done

for i in {1..13}
do
    python run_experiment.py --exp 1.2;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.71;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.72;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.73;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.74;
done
