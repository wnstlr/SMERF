#!/bin/bash
for i in {1..13}
do
    python run_experiment.py --exp 1.11 --bg 0;
done

for i in {1..9}
do
    python run_experiment.py --exp 2.11 --bg 0;
done

for i in {1..13}
do
    python run_experiment.py --exp 1.2 --bg 0;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.71 --bg 0 --ep 10;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.72 --bg 0 --ep 10;
done
#
for i in {1..11}
do
    python run_experiment.py --exp 3.73 --bg 0 --ep 10;
done

for i in {1..11}
do
    python run_experiment.py --exp 3.74 --bg 0 --ep 10;
done
