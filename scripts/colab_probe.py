"""Google Colab CLIのGPUランタイムを小さく検証する。"""

from __future__ import annotations

import json
import platform
import subprocess
import time


def main() -> None:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError as error:
        result.update({"torch_available": False, "error": repr(error)})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    result.update(
        {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        result["device_name"] = torch.cuda.get_device_name(device)
        result["device_capability"] = list(torch.cuda.get_device_capability(device))
        left = torch.randn((2048, 2048), device=device)
        right = torch.randn((2048, 2048), device=device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(10):
            output = left @ right
        torch.cuda.synchronize()
        result["matmul_2048x2048_10_iterations_seconds"] = time.perf_counter() - started
        result["output_mean"] = float(output.mean().item())
        result["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    else:
        result["device_name"] = None

    try:
        result["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError as error:
        result["nvidia_smi_error"] = repr(error)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
