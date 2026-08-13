# Basic

## Reduce

Reduce is a collective communication primitive in distributed training. Each rank provides its own local input tensor. All ranks collaboratively perform a specified element‑wise mathematical operation (sum, max, min, product, etc.) across all input tensors. 

AllReduce, AllReduce=ReduceScatter+AllGather

Ring AllReduce

use cuda API to fetch data between gpus, will choose NVLink or PCIe autoly.