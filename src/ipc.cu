#include "ipc.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include "cuda_check.h"

namespace {

constexpr int kFileWaitMaxRetries = 1000;
constexpr int kFileWaitRetryMicros = 10000;
constexpr char kBarrierMarker[] = "ready\n";

bool regular_file_with_size(const std::string& path, size_t expected_size) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) &&
           static_cast<size_t>(st.st_size) == expected_size;
}

void wait_for_file(const std::string& path, size_t expected_size,
                   const char* what) {
    for (int retry = 0; retry < kFileWaitMaxRetries; retry++) {
        if (regular_file_with_size(path, expected_size))
            return;
        usleep(kFileWaitRetryMicros);
    }
    throw std::runtime_error(std::string("Timeout waiting for complete ") +
                             what + ": " + path);
}

// Atomically publish `size` bytes at `data` to `path`: write a per-process
// temp file, then rename it into place so readers never see a partial file.
void atomic_write(const std::string& path, const void* data, size_t size) {
    const std::string tmp_path = path + ".tmp." + std::to_string(getpid());
    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open temporary file: " + tmp_path);
    }
    ofs.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(size));
    ofs.close();
    if (!ofs) {
        unlink(tmp_path.c_str());
        throw std::runtime_error("Failed to write temporary file: " + tmp_path);
    }
    if (rename(tmp_path.c_str(), path.c_str()) != 0) {
        unlink(tmp_path.c_str());
        throw std::runtime_error("Failed to publish file: " + path);
    }
}

}  // namespace

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

    return handle;
}

void ipc_save_handle(const IpcHandle& handle, const std::string& path) {
    atomic_write(path, &handle, sizeof(handle));
}

IpcHandle ipc_load_handle(const std::string& path) {
    wait_for_file(path, sizeof(IpcHandle), "IPC handle");
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    IpcHandle handle{};
    ifs.read(reinterpret_cast<char*>(&handle), sizeof(IpcHandle));
    if (!ifs) {
        throw std::runtime_error("Failed to read handle from: " + path);
    }
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

IpcEventSet ipc_create_events(int device) {
    IpcEventSet events{};
    events.device = device;
    int old_device;
    check_cuda(cudaGetDevice(&old_device), "cudaGetDevice");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cuda(
        cudaEventCreateWithFlags(
            &events.init_ready, cudaEventDisableTiming | cudaEventInterprocess),
        "cudaEventCreateWithFlags(init_ready)");
    check_cuda(cudaEventCreateWithFlags(
                   &events.reduce_scatter_done,
                   cudaEventDisableTiming | cudaEventInterprocess),
               "cudaEventCreateWithFlags(reduce_scatter_done)");
    check_cuda(
        cudaIpcGetEventHandle(&events.init_ready_handle, events.init_ready),
        "cudaIpcGetEventHandle(init_ready)");
    check_cuda(cudaIpcGetEventHandle(&events.reduce_scatter_done_handle,
                                     events.reduce_scatter_done),
               "cudaIpcGetEventHandle(reduce_scatter_done)");
    check_cuda(cudaSetDevice(old_device), "cudaSetDevice restore");
    return events;
}

void ipc_destroy_events(IpcEventSet& events) {
    int old_device;
    if (cudaGetDevice(&old_device) != cudaSuccess) {
        // Best-effort teardown, called from destructors: it must not throw.
        cudaGetLastError();
        return;
    }
    if (events.device >= 0)
        cudaSetDevice(events.device);
    if (events.init_ready)
        cudaEventDestroy(events.init_ready);
    if (events.reduce_scatter_done)
        cudaEventDestroy(events.reduce_scatter_done);
    events.init_ready = nullptr;
    events.reduce_scatter_done = nullptr;
    cudaSetDevice(old_device);
}

void ipc_save_event_handles(const IpcEventSet& events,
                            const std::string& path) {
    IpcEventHandleSet handles{};
    handles.init_ready_handle = events.init_ready_handle;
    handles.reduce_scatter_done_handle = events.reduce_scatter_done_handle;
    handles.device = events.device;
    atomic_write(path, &handles, sizeof(handles));
}

IpcEventHandleSet ipc_load_event_handles(const std::string& path) {
    wait_for_file(path, sizeof(IpcEventHandleSet), "IPC event handles");
    std::ifstream ifs(path, std::ios::binary);
    IpcEventHandleSet handles{};
    ifs.read(reinterpret_cast<char*>(&handles), sizeof(handles));
    if (!ifs)
        throw std::runtime_error("Failed to read event handles: " + path);
    return handles;
}

cudaEvent_t ipc_open_event(const cudaIpcEventHandle_t& handle,
                           int local_device) {
    cudaEvent_t event = nullptr;
    int old_device;
    check_cuda(cudaGetDevice(&old_device), "cudaGetDevice");
    check_cuda(cudaSetDevice(local_device), "cudaSetDevice");
    check_cuda(cudaIpcOpenEventHandle(&event, handle),
               "cudaIpcOpenEventHandle");
    check_cuda(cudaSetDevice(old_device), "cudaSetDevice restore");
    return event;
}

void ipc_close_event(cudaEvent_t event) {
    if (event) {
        cudaError_t err = cudaEventDestroy(event);
        if (err != cudaSuccess) {
            fprintf(stderr, "[IPC EVENT] close failed: %s\n",
                    cudaGetErrorString(err));
        }
    }
}

std::vector<IpcHandle> ipc_exchange_handles(int my_rank, int world_size,
                                            void* my_ptr, size_t my_size,
                                            int my_device,
                                            const std::string& ipc_dir) {
    mkdir(ipc_dir.c_str(), 0755);

    IpcHandle my_handle = ipc_create_handle(my_ptr, my_size, my_device);
    std::string my_path = ipc_dir + "/rank_" + std::to_string(my_rank) + ".bin";
    ipc_save_handle(my_handle, my_path);

    for (int r = 0; r < world_size; r++) {
        std::string path = ipc_dir + "/rank_" + std::to_string(r) + ".bin";
        wait_for_file(path, sizeof(IpcHandle), "rank IPC handle");
    }

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

std::vector<IpcEventHandleSet> ipc_exchange_event_handles(
    int my_rank, int world_size, const IpcEventSet& local_events,
    const std::string& ipc_dir) {
    mkdir(ipc_dir.c_str(), 0755);
    const std::string my_path =
        ipc_dir + "/events_rank_" + std::to_string(my_rank) + ".bin";
    ipc_save_event_handles(local_events, my_path);

    std::vector<IpcEventHandleSet> handles(world_size);
    for (int r = 0; r < world_size; r++) {
        const std::string path =
            ipc_dir + "/events_rank_" + std::to_string(r) + ".bin";
        wait_for_file(path, sizeof(IpcEventHandleSet), "rank IPC events");
        handles[r] = ipc_load_event_handles(path);
    }
    return handles;
}

void ipc_barrier(int my_rank, int world_size, const std::string& ipc_dir,
                 const std::string& phase, int step) {
    const std::string prefix =
        ipc_dir + "/barrier_" + phase + "_" + std::to_string(step) + "_";
    const std::string marker = prefix + "rank_" + std::to_string(my_rank);
    atomic_write(marker, kBarrierMarker, sizeof(kBarrierMarker) - 1);

    for (int r = 0; r < world_size; r++) {
        wait_for_file(prefix + "rank_" + std::to_string(r), 6,
                      "barrier marker");
    }
}
