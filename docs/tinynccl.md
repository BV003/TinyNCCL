# TinyNCCL

TinyNCCL 是一个用于学习 NCCL 与分布式集合通信的轻量实现：单机多卡、基于 CUDA IPC 的 Ring AllReduce。对外以 PyTorch C++ 扩展 `tiny` 暴露，内部由 `ipc` / `transport` / `kernels` / `cuda_check` / `ring_allreduce` 几个模块协作完成。

多进程 API 是持久化 communicator：通信建立一次，之后重复 `allreduce`；同步完全走 GPU 侧的内存计数器。

- `tiny.RingCommunicator(rank, world, max_count, ipc_dir)`
- `comm.allreduce(send, recv)`

---

## 1. Data types

### C++ 侧

| 类型 | 文件 | 说明 |
|------|------|------|
| `IpcHandle` | `src/ipc.h` | 单个 rank 的 CUDA IPC 显存句柄：`cudaIpcMemHandle_t mem_handle`、`void* base_ptr`、`size_t size`、`int device` |
| `P2pState` | `src/transport.cu` | P2P 探测缓存状态枚举：`kUnknown` / `kSupported` / `kUnsupported` |
| `MemOps` | `src/ring_allreduce.cu` | 运行时解析出的 driver 函数指针：`wait32`、`write32`（`cuStreamWaitValue32` / `cuStreamWriteValue32`） |

### Python 侧（`scripts/topology.py`）

| 类型 | 说明 |
|------|------|
| `GPUInfo` | 单张 GPU 的信息：`index`、`name`、`memory_gb`、`sm_capability` |
| `NodeTopology` | 整机拓扑：`gpus: list[GPUInfo]`、`p2p_matrix: dict[tuple[int, int], bool]`、`topo: list[list[str]]` |

### Torch 侧

| 类型 | 说明 |
|------|------|
| `torch::Tensor` | 所有 C++ 接口的入参/出参张量类型；核心只支持 `float32` CUDA 张量 |

---

## 2. Classes

### `RingCommunicator`（`src/ring_allreduce.h`，核心类）

持久化多进程通信器。构造时一次性建立 stream、`scratch_` / `commbuff_` / `counter_` 缓冲、交换并打开远端内存句柄与计数器；之后每次 `allreduce` 只在 stream 上排 GPU 操作。

```cpp
RingCommunicator(int my_rank, int world_size,
                 size_t max_count, const std::string& ipc_dir);
~RingCommunicator();

void allreduce(float* sendbuff, float* recvbuff, size_t count);
int  world_size() const;
size_t max_count() const;
```

私有成员与状态：

| 成员 | 说明 |
|------|------|
| `my_rank_`, `world_size_`, `max_count_`, `ipc_dir_` | 构造参数 |
| `steps_ = world_size - 1`, `prev_` | 环邻居 |
| `epoch_` | 调用计数，用于计算计数器基准并防止 uint32 回绕 |
| `local_dev_`, `stream_` | 本地设备与非默认 stream |
| `scratch_`, `commbuff_`, `counter_` | 临时块、通信缓冲、单调递增的 32 位同步计数器（显存） |
| `handles_`, `remote_ptrs_`, `remote_counters_` | 远端内存/IPC 映射 |

私有方法：`initialize` / `setup_stream_and_buffers` / `setup_remote_memory` / `reduce_scatter` / `all_gather` / `signal` / `wait_for` / `teardown` / `signals_per_call()`。

### `DeviceGuard`（`src/transport.cu`，内部）

RAII：构造时保存当前 device，析构时恢复；保证拷贝抛异常也不改变调用方的 device。

```cpp
DeviceGuard();
~DeviceGuard();
```

### `DeviceStreamBuffers`（`src/ring_allreduce.cu`，内部）

单进程多卡路径（`tiny_ring_allreduce_sum_impl`）的 RAII 资源持有者：per-device 的 stream 与临时缓冲；部分分配失败时回滚已分配的资源。

```cpp
DeviceStreamBuffers(int n_gpus, size_t chunk_bytes);
~DeviceStreamBuffers();
cudaStream_t stream(int i) const;
float*       scratch(int i) const;
```

### `IpcHandle`（`src/ipc.h`）

```cpp
cudaIpcMemHandle_t mem_handle;
void*              base_ptr;   // 本 rank 的原始指针（用于校验）
size_t             size;       // buffer 字节数
int                device;     // buffer 所在 GPU device
```

### `GPUInfo` / `NodeTopology`（`scripts/topology.py`）

```python
@dataclass
class GPUInfo:
    index: int
    name: str
    memory_gb: float
    sm_capability: str

@dataclass
class NodeTopology:
    gpus: list[GPUInfo] = field(default_factory=list)
    p2p_matrix: dict[tuple[int, int], bool] = field(default_factory=dict)
    topo: list[list[str]] = field(default_factory=list)
```

### `TinyExtension`（PyTorch 扩展模块 `tiny`，由 `PYBIND11_MODULE` 注册）

不是 C++ 类，而是对外的模块边界；其方法见第 3 节。

```python
tiny_reduce_sum(dst, a, b) -> None
tiny_copy_peer(dst, src) -> None
tiny_ring_allreduce_sum(sendbuffs, recvbuffs) -> None

RingCommunicator(my_rank, world_size, max_count, ipc_dir)   # 类
    .allreduce(sendbuff, recvbuff) -> None
    .world_size -> int
    .max_count  -> int
```

---

## 3. Functions

### `src/cuda_check.h`

```cpp
// err != cudaSuccess 时抛出带上下文与 cudaGetErrorString 的 runtime_error
inline void check_cuda(cudaError_t err, const std::string& context);
```

### `ipc.h` / `ipc.cu`

```cpp
IpcHandle ipc_create_handle(void* ptr, size_t size, int device);
void      ipc_save_handle(const IpcHandle& handle, const std::string& path);
IpcHandle ipc_load_handle(const std::string& path);
void*     ipc_open_handle(const IpcHandle& handle, int local_device);
void      ipc_close_handle(void* mapped_ptr);
std::vector<IpcHandle> ipc_exchange_handles(
    int my_rank, int world_size,
    void* my_ptr, size_t my_size, int my_device,
    const std::string& ipc_dir);
```

内部辅助（匿名命名空间）：

```cpp
constexpr int  kFileWaitRetryMicros;
constexpr int  kFileWaitMaxRetries;
bool regular_file_with_size(const std::string& path, size_t expected_size);
void wait_for_file(const std::string& path, size_t expected_size, const char* what);
void atomic_write(const std::string& path, const void* data, size_t size);
```

### `transport.h` / `transport.cu`

```cpp
void transport_copy(void* dst, int dst_dev, const void* src, int src_dev,
                    size_t nbytes, cudaStream_t stream);
void transport_copy_ipc(void* dst, int dst_dev, const void* src, int src_dev,
                        size_t nbytes, cudaStream_t stream);
```

内部辅助：

```cpp
constexpr int kMaxTrackedDevices;
enum class P2pState;
void  check_dev_index(int src_dev, int dst_dev);
bool  p2p_available(int src_dev, int dst_dev);
void* get_staging(int dst_dev, size_t nbytes);
void  copy_via_host(void* dst, int dst_dev, const void* src,
                    size_t nbytes, cudaStream_t stream);
```

### `kernels.h` / `kernels.cu`

```cpp
// size_t 索引 + grid-stride，无 2^31 元素上限
__global__ void reduce_sum_kernel(float* dst, const float* a,
                                  const float* b, std::size_t n);

void tiny_reduce_sum_kernel(float* dst, const float* a, const float* b,
                            std::size_t n, cudaStream_t stream);
```

内部常量：`kThreadsPerBlock`、`kMaxBlocks`。

### `ring_allreduce.h` / `ring_allreduce.cu`

```cpp
// 单进程多 GPU 版本（保留用于单进程测试）
void tiny_ring_allreduce_sum_impl(float** send, float** recv,
                                  size_t count, int n_gpus);
```

内部辅助（匿名命名空间）：

```cpp
void check_cuda_ipc(cudaError_t err, const char* operation,
                    int rank, const char* phase, int step);
void synchronized_copy(float* dst, float* src, size_t count, int device);

const MemOps& mem_ops();
void ensure_mem_ops_available();

// 单进程路径
void impl_reduce_scatter(float** recv, int n_gpus, size_t chunk,
                         size_t chunk_bytes, const DeviceStreamBuffers& buffers);
void impl_all_gather(float** recv, int n_gpus, size_t chunk,
                     size_t chunk_bytes, const DeviceStreamBuffers& buffers);
```

### `ring_schedule.h`（Ring 调度的纯主机端计算）

不依赖 CUDA 头文件，供 `ring_allreduce.cu` 使用，也可在纯 CPU 机器上单独测试。

```cpp
namespace tinynccl::ring {
int prev_rank(int rank, int n);              // (rank - 1) mod n
int rs_chunk(int rank, int step, int n);     // RS 第 s 步：(rank - s) mod n
int owned_chunk(int rank, int n);            // RS 结束后持有的完整块：(rank + 2) mod n
int ag_chunk(int rank, int step, int n);     // AG 第 s 步：(rank - s + 1) mod n

// 计数器信号编号：0 = 输入已拷贝，1..steps = reduce-scatter，steps+1..2*steps = all-gather
int      signals_per_call(int steps);        // 1 + 2*steps
uint32_t input_ready_signal(uint32_t base);
uint32_t rs_step_signal(uint32_t base, int step);
uint32_t ag_step_signal(uint32_t base, int steps, int step);
uint32_t rs_step_dependency(uint32_t base, int step);
uint32_t ag_step_dependency(uint32_t base, int steps, int step);
}
```

AG 第 0 步读取的必须是 prev 的 `owned_chunk`。n = 2 时 `(rank - s - 1)` 与
`(rank - s + 1)` 恰好相同，所以这个下标写错只有在 3 卡及以上才会暴露。

### `bindings.cpp`（PyTorch 扩展入口）

```cpp
void tiny_reduce_sum(torch::Tensor dst, torch::Tensor a, torch::Tensor b);
void tiny_copy_peer(torch::Tensor dst, torch::Tensor src);
void tiny_ring_allreduce_sum(std::vector<torch::Tensor> sendbuffs,
                             std::vector<torch::Tensor> recvbuffs);
```

另注册 `RingCommunicator` 类（`.allreduce` / `.world_size` / `.max_count`）。

内部校验辅助（匿名命名空间）：

```cpp
void require_same_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* message);
void require_float32(const torch::Tensor& tensor, const char* name);
void require_cuda(const torch::Tensor& tensor, const char* name);
void require_divisible(size_t count, size_t divisor, const char* name);
void require_allreduce_inputs(const torch::Tensor& sendbuff,
                              const torch::Tensor& recvbuff, size_t world_size);
```

### `scripts/topology.py`

```python
def _nvidia_smi(args: list[str]) -> str
def get_gpu_info() -> list[GPUInfo]
def get_topo_matrix_simple() -> list[str]
def check_p2p() -> dict[tuple[int, int], bool]
def detect() -> NodeTopology
def format_report(t: NodeTopology) -> str
def main()
```

### `scripts/benchmark.py`

```python
def parse_args() -> argparse.Namespace
def barrier(local_rank: int) -> None
def check_result(result, reference) -> bool
def _time(run_once, warmup, iters, local_rank) -> float
def run_tiny(...)             # 持久化 RingCommunicator
def run_nccl(...)             # torch.distributed 基线
def format_result(name, count, seconds, world) -> str
def main() -> int
```

### `tests/`

```python
# harness.py（多进程测试共享脚手架）
barrier(local_rank) / init_distributed() / make_ipc_root(tag, rank, local_rank)
cleanup_ipc_root(root, rank, local_rank) / all_ranks_ok(local_ok, local_rank)

# assertions.py
require(condition, message) -> None      # 不被 python -O 剥掉

# test_kernels.py
test_reduce_sum_scalar() / test_reduce_sum_small()
test_reduce_sum_large()  / test_reduce_sum_2d()

# test_transport.py
check_copy(src_device, dst_device, count) -> None

# test_single_process.py
run_case(count, n_gpus) -> None

# 多进程测试（torchrun）
main() -> int          # test_fixed_data / test_vs_torch_distributed /
                       # test_persistent_communicator / smoke_test

# test_ring_schedule.cpp（纯 CPU，无需 GPU）
# 用 ring_schedule.h 的真实下标/信号计算，在随机交错执行下模拟 1..8 个 rank，
# 覆盖持久化通信器（含多次调用重叠时的 epoch barrier）和单进程路径

# 一键运行（编译 + 跑全套；多进程测试默认使用全部可见 GPU，可用 NPROC=N 覆盖；
# < 2 卡自动跳过多进程测试）
# scripts/run_tests.sh [--no-build]
```

---

## 4. Relations

```mermaid
%%{init: {'class': {'hideEmptyMembersBox': true}}}%%
classDiagram
    direction LR

    class TinyExtension {
        <<pybind11 module>>
    }
    class RingCommunicator
    class SingleProcessImpl {
        <<function>>
    }
    class Transport {
        <<module>>
    }
    class Kernels {
        <<module>>
    }
    class Ipc {
        <<module>>
    }
    class CudaCheck {
        <<header>>
    }
    class MemOps {
        <<struct>>
    }
    class IpcHandle
    class DeviceGuard
    class DeviceStreamBuffers
    class GPUInfo
    class NodeTopology
    class TopologyScript {
        <<script>>
    }
    class BenchmarkScript {
        <<script>>
    }
    class TestHarness {
        <<tests>>
    }
    class RunTests {
        <<script>>
    }
    class RingSchedule {
        <<header>>
    }
    class ScheduleTest {
        <<tests>>
    }
    class TorchCUDA {
        <<external>>
    }
    class NvidiaSMI {
        <<external>>
    }
    class CudaDriver {
        <<external>>
    }

    TinyExtension --> RingCommunicator : binds class
    TinyExtension --> Transport : tiny_copy_peer
    TinyExtension --> Kernels : tiny_reduce_sum
    TinyExtension ..> SingleProcessImpl : tiny_ring_allreduce_sum

    RingCommunicator --> Transport : transport_copy_ipc
    RingCommunicator --> Kernels : tiny_reduce_sum_kernel
    RingCommunicator --> Ipc : exchange / open handles
    RingCommunicator --> MemOps : signal / wait_for
    RingCommunicator --> CudaCheck : check_cuda
    RingCommunicator --> RingSchedule : chunk indices / signals

    SingleProcessImpl --> DeviceStreamBuffers : owns
    SingleProcessImpl --> Transport : transport_copy
    SingleProcessImpl --> Kernels : tiny_reduce_sum_kernel
    SingleProcessImpl --> RingSchedule : chunk indices

    MemOps --> CudaDriver : cudaGetDriverEntryPoint + cuStream*Value32
    Ipc ..> IpcHandle : creates / serializes
    Ipc --> CudaCheck : check_cuda
    Transport --> DeviceGuard : restores device
    Transport --> CudaCheck : check_cuda
    Transport --> TorchCUDA : cudaMemcpyPeerAsync / cudaMemcpyAsync
    Kernels --> TorchCUDA : launches kernel on stream
    Ipc --> TorchCUDA : cudaIpc* APIs

    NodeTopology "1" o-- "0..*" GPUInfo : contains
    TopologyScript ..> NodeTopology : detect / format_report
    TopologyScript --> NvidiaSMI : queries
    TopologyScript --> TorchCUDA : probes P2P

    BenchmarkScript ..> RingCommunicator : times allreduce
    BenchmarkScript ..> TorchCUDA : NCCL baseline (torch.distributed)
    TestHarness ..> TinyExtension : drives tests
    RunTests ..> TestHarness : runs tests
    RunTests ..> ScheduleTest : runs (CPU only)
    ScheduleTest ..> RingSchedule : simulates
```
