#ifndef TINYNCCL_IPC_H
#define TINYNCCL_IPC_H

#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

// CUDA IPC memory handle for one rank's buffer.
struct IpcHandle {
    cudaIpcMemHandle_t mem_handle;
    void* base_ptr;  // owning rank's original pointer (for validation)
    size_t size;     // buffer size in bytes
    int device;      // GPU device the buffer lives on
};

// Export the current rank's buffer as an IPC handle.
IpcHandle ipc_create_handle(void* ptr, size_t size, int device);

// Publish a handle to a file so the other ranks can read it.
void ipc_save_handle(const IpcHandle& handle, const std::string& path);

// Read a handle written by ipc_save_handle.
IpcHandle ipc_load_handle(const std::string& path);

// Map a remote handle into this process and return a usable device pointer.
void* ipc_open_handle(const IpcHandle& handle, int local_device);

// Close a pointer returned by ipc_open_handle.
void ipc_close_handle(void* mapped_ptr);

// Exchange every rank's handle through a shared directory (for example
// "/tmp/tinynccl_ipc") and return them indexed by rank. Handles are written as
// raw struct bytes, so all ranks must run binaries built with the same ABI.
std::vector<IpcHandle> ipc_exchange_handles(int my_rank, int world_size,
                                            void* my_ptr, size_t my_size,
                                            int my_device,
                                            const std::string& ipc_dir);

#endif
