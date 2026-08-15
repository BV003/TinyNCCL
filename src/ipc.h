#ifndef TINYNCCL_IPC_H
#define TINYNCCL_IPC_H

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

// 每个 rank 的 IPC handle 信息
struct IpcHandle {
    cudaIpcMemHandle_t mem_handle;
    void* base_ptr;      // 原始指针（用于校验）
    size_t size;         // buffer 大小（字节）
    int device;          // 该 buffer 所在的 GPU device
};

struct IpcEventSet {
    cudaEvent_t init_ready = nullptr;
    cudaEvent_t reduce_scatter_done = nullptr;
    cudaIpcEventHandle_t init_ready_handle{};
    cudaIpcEventHandle_t reduce_scatter_done_handle{};
    int device = -1;
};

struct IpcEventHandleSet {
    cudaIpcEventHandle_t init_ready_handle{};
    cudaIpcEventHandle_t reduce_scatter_done_handle{};
    int device = -1;
};

// 将当前 rank 的 buffer 创建 IPC handle
IpcHandle ipc_create_handle(void* ptr, size_t size, int device);

// 将 handle 写入文件（用于进程间交换）
void ipc_save_handle(const IpcHandle& handle, const std::string& path);

// 从文件读取 handle
IpcHandle ipc_load_handle(const std::string& path);

// 打开远程 handle，返回本地可访问的 device pointer
void* ipc_open_handle(const IpcHandle& handle, int local_device);

// 关闭已打开的 IPC handle
void ipc_close_handle(void* mapped_ptr);

IpcEventSet ipc_create_events(int device);
void ipc_destroy_events(IpcEventSet& events);
void ipc_save_event_handles(const IpcEventSet& events, const std::string& path);
IpcEventHandleSet ipc_load_event_handles(const std::string& path);
cudaEvent_t ipc_open_event(const cudaIpcEventHandle_t& handle, int local_device);
void ipc_close_event(cudaEvent_t event);

std::vector<IpcEventHandleSet> ipc_exchange_event_handles(
    int my_rank, int world_size,
    const IpcEventSet& local_events,
    const std::string& ipc_dir);

// 批量交换 handles：每个 rank 把自己的 handle 写到共享目录，然后读取所有其他 rank 的
// rank_dir: 例如 "/tmp/tinynccl_ipc"
// 返回: vector of IpcHandle，按 rank 索引排序
std::vector<IpcHandle> ipc_exchange_handles(
    int my_rank, int world_size,
    void* my_ptr, size_t my_size, int my_device,
    const std::string& ipc_dir);

// 基于共享文件的进程间 barrier，用于 ring 每一步的跨进程同步。
void ipc_barrier(
    int my_rank, int world_size,
    const std::string& ipc_dir,
    const std::string& phase, int step);

#endif
