# TinyNCCL

## Learning Purpose

I do this project for learning about NCCL and Distributed system.

## Hardware Requirements

2* A5000

## Environment


| Component      | Version         |
|---------------|-----------------|
| NVIDIA Driver  | ≥ 525.60        |
| nvcc          | 12.6            |
| Python         | 3.12            |
| PyTorch        | 2.5.1+cu126     |

## Build & Install

```bash
pip install -e .
```

## Run Tests

```bash
python tests/test_kernels.py
```
