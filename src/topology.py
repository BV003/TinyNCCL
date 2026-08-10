#=============================================================================
# topology.py
#
# TinyNCCL — hardware topology detection.
#
# Determines, for the current node:
#   * how many GPUs are visible and their basic attributes
#   * the PCIe / NVLink inter-GPU topology (like `nvidia-smi topo -m`)
#   * whether P2P (GPUDirect peer) is available between each pair of GPUs
#
# Usage:
#   python topology.py                 # print a one-page report
#   python -c "import topology"        # use as a library (see get_report)
#
# The P2P result is critical: it decides which transport the Ring
# AllReduce uses (GPUDirect P2P vs host-staging fallback).
#=============================================================================
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_gb: float
    sm_capability: str


@dataclass
class NodeTopology:
    gpus: list[GPUInfo] = field(default_factory=list)
    p2p_matrix: dict = field(default_factory=dict)   # {(i, j): bool}
    topo: list[list[str]] = field(default_factory=list)  # from nvidia-smi topo -m


# ---------------------------------------------------------------------------
# Pure query functions (no side effects on the GPU)
# ---------------------------------------------------------------------------

def _nvidia_smi(args: list[str]) -> str:
    """Run nvidia-smi with the given args; return stdout ('' on failure)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _torch_cuda_count() -> int:
    """Number of GPUs visible to PyTorch/torch.cuda."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_gpu_info() -> list[GPUInfo]:
    """List GPUs visible to nvidia-smi (falls back to torch if needed)."""
    gpus: list[GPUInfo] = []
    out = _nvidia_smi([
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                try:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_gb=float(parts[2]) / 1024.0,
                        sm_capability=parts[3],
                    ))
                except ValueError:
                    pass
    if gpus:
        return gpus

    # Fallback: torch only (no memory breakdown)
    import torch
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        prop = torch.cuda.get_device_properties(i)
        gpus.append(GPUInfo(
            index=i, name=name,
            memory_gb=prop.total_memory / (1024 ** 3),
            sm_capability=f"{prop.major}.{prop.minor}",
        ))
    return gpus


def get_topo_matrix_simple() -> list[str]:
    """Return the raw nvidia-smi topo -m output (for debugging / display)."""
    out = _nvidia_smi(["topo", "-m"])
    return out.splitlines() if out else ["(nvidia-smi topo -m unavailable)"]


def check_p2p() -> dict[tuple[int, int], bool]:
    """
    Probe P2P accessibility between every ordered pair of GPUs using torch.
    result[(i, j)] == True means GPU j's memory is directly accessible
    from GPU i (GPUDirect P2P path available).
    """
    import torch
    n = torch.cuda.device_count()
    pairs: dict[tuple[int, int], bool] = {}
    for i in range(n):
        pairs[(i, i)] = True  # same device: always "accessible"
        for j in range(n):
            if i == j:
                continue
            try:
                with torch.cuda.device(i):
                    pairs[(i, j)] = bool(torch.cuda.can_device_access_peer(i, j))
            except Exception:
                pairs[(i, j)] = False
    return pairs


def detect() -> NodeTopology:
    """Gather everything into one object."""
    t = NodeTopology()
    t.gpus = get_gpu_info()
    t.p2p_matrix = check_p2p()
    t.topo = get_topo_matrix_simple()
    return t


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(t: NodeTopology) -> str:
    lines = []
    lines.append("=" * 58)
    lines.append("TinyNCCL — Hardware Topology Report")
    lines.append("=" * 58)

    lines.append(f"\n[GPUs] visible to nvidia-smi: {len(t.gpus)}")
    for g in t.gpus:
        lines.append(
            f"  GPU{g.index:<2} {g.name:<28} "
            f"{g.memory_gb:7.1f} GB  sm_{g.sm_capability}"
        )

    n = len(t.gpus)
    lines.append(f"\n[Topology] nvidia-smi topo -m:")
    for row in t.topo:
        lines.append("  " + row)

    lines.append(f"\n[P2P matrix] ((i,j) = can GPU i access GPU j?)  N={n}")
    if n == 0:
        lines.append("  (no GPUs)")
    else:
        header = "     " + "  ".join(f"GPU{j}" for j in range(n))
        lines.append(header)
        for i in range(n):
            row = f"GPU{i}  "
            for j in range(n):
                v = t.p2p_matrix.get((i, j), "-")
                row += f"   {'Y' if v else 'N'}"
            lines.append(row)

    # Summary / recommendation
    p2p_any = n > 0 and any(
        v for (i, j), v in t.p2p_matrix.items() if i != j and v
    )
    using = "GPUDirect P2P" if p2p_any else "host-staging fallback"
    lines.append(f"\n[Transport choice] P2P available -> will use: {using}")

    return "\n".join(lines)


def main():
    t = detect()
    print(format_report(t))


if __name__ == "__main__":
    main()
