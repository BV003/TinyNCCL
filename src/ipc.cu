#include "ipc.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

IpcHandle ipc_create_handle(void* ptr, size_t size, int device) {
    IpcHandle handle{};
    handle.base_ptr = ptr;
    handle.size = size;
    handle.device = device;

    int old_dev;
    check_cuda(cudaGetDevice(&old_dev), "cudaGetDevice");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cuda(cudaIpcGetMemHandle(&handle.mem_handle, ptr),
               "cudaIpcGetMemHandle");
    check_cuda(cudaSetDevice(old_dev), "cudaSetDevice restore");

    fprintf(stderr, "[IPC] created handle for dev=%d ptr=%p size=%zu\n",
            device, ptr, size);
    return handle;
}

void ipc_save_handle(const IpcHandle& handle, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(&handle), sizeof(IpcHandle));
    if (!ofs) {
        throw std::runtime_error("Failed to write handle to: " + path);
    }
    fprintf(stderr, "[IPC] saved handle to %s\n", path.c_str());
}

IpcHandle ipc_load_handle(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    IpcHandle handle{};
    ifs.read(reinterpret_cast<char*>(&handle), sizeof(IpcHandle));
    if (!ifs) {
        throw std::runtime_error("Failed to read handle from: " + path);
    }
    fprintf(stderr, "[IPC] loaded handle from %s (remote dev=%d)\n",
            path.c_str(), handle.device);
    return handle;
}

void* ipc_open_handle(const IpcHandle& handle, int local_device) {
    void* mapped = nullptr;
    int old_dev;
    check_cuda(cudaGetDevice(&old_dev), "cudaGetDevice");
    check_cuda(cudaSetDevice(local_device), "cudaSetDevice");
    check_cuda(cudaIpcOpenMemHandle(&mapped, handle.mem_handle,
                                     cudaIpcMemLazyEnablePeerAccess),
               "cudaIpcOpenMemHandle");
    check_cuda(cudaSetDevice(old_dev), "cudaSetDevice restore");

    fprintf(stderr, "[IPC] opened remote handle (dev=%d) -> local ptr=%p on dev=%d\n",
            handle.device, mapped, local_device);
    return mapped;
}

void ipc_close_handle(void* mapped_ptr) {
    if (mapped_ptr) {
        cudaError_t err = cudaIpcCloseMemHandle(mapped_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[IPC] WARNING: cudaIpcCloseMemHandle failed: %s\n",
                    cudaGetErrorString(err));
        }
    }
}

std::vector<IpcHandle> ipc_exchange_handles(
    int my_rank, int world_size,
    void* my_ptr, size_t my_size, int my_device,
    const std::string& ipc_dir)
{
    // 创建共享目录
    mkdir(ipc_dir.c_str(), 0755);

    // 1. 创建并保存自己的 handle
    IpcHandle my_handle = ipc_create_handle(my_ptr, my_size, my_device);
    std::string my_path = ipc_dir + "/rank_" + std::to_string(my_rank) + ".bin";
    ipc_save_handle(my_handle, my_path);

    // 2. 等待所有 rank 写完（简单轮询）
    for (int r = 0; r < world_size; r++) {
        std::string path = ipc_dir + "/rank_" + std::to_string(r) + ".bin";
        int retries = 100;
        while (retries > 0) {
            std::ifstream ifs(path, std::ios::binary);
            if (ifs) break;
            usleep(10000); // 10ms
            retries--;
        }
        if (retries == 0) {
            throw std::runtime_error("Timeout waiting for rank " +
                                     std::to_string(r) + " handle");
        }
    }

    // 3. 读取所有 rank 的 handle
    std::vector<IpcHandle> handles(world_size);
    for (int r = 0; r < world_size; r++) {
        if (r == my_rank) {
            handles[r] = my_handle;
        } else {
            std::string path = ipc_dir + "/rank_" + std::to_string(r) + ".bin";
            handles[r] = ipc_load_handle(path);
        }
    }

    return handles;
}
