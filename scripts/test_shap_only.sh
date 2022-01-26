#!/bin/bash

python test_shap.py --exp 1.11 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 2.11 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 1.2 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 3.71 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 3.72 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 3.73 --model_type 1 --cache_dir ../outputs_vgg/cache;
python test_shap.py --exp 3.74 --model_type 1 --cache_dir ../outputs_vgg/cache;

python test_shap.py --exp 1.11 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 2.11 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 1.2 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 3.71 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 3.72 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 3.73 --model_type 2 --cache_dir ../outputs_alex/cache;
python test_shap.py --exp 3.74 --model_type 2 --cache_dir ../outputs_alex/cache;
