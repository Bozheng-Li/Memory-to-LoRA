#!/bin/bash
cd /home/libaizheng/Lora-Grpo-memory
export CUDA_VISIBLE_DEVICES=2
exec .venv-d2l/bin/python -u scripts/_pilot/pilot_dilution.py --n 12
