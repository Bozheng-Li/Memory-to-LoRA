#!/bin/bash
cd /home/libaizheng/Lora-Grpo-memory
export CUDA_VISIBLE_DEVICES=2
exec .venv-d2l/bin/python -u scripts/_pilot/pilot_mab.py --row 0 --n 25 --out research/pilot_20261028/raw/results_mab_cr_v2.jsonl
