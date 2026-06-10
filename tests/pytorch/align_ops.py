#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Result:
    name: str
    max_abs_error: float
    mean_abs_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return np.isfinite(self.max_abs_error) and self.max_abs_error <= self.tolerance


def write_float32(path: Path, values: np.ndarray) -> None:
    np.asarray(values, dtype=np.float32).tofile(path)


def read_float32(path: Path, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != count:
        raise RuntimeError(f"{path} contains {values.size} floats, expected {count}")
    return values


def compare(name: str, actual: np.ndarray, expected: torch.Tensor, tolerance: float) -> Result:
    reference = expected.detach().cpu().numpy().astype(np.float32, copy=False)
    errors = np.abs(actual - reference)
    return Result(
        name=name,
        max_abs_error=float(errors.max(initial=0.0)),
        mean_abs_error=float(errors.mean()),
        tolerance=tolerance,
    )


def run_command(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def align_matmul(runner: Path, directory: Path, rng: np.random.Generator) -> list[Result]:
    results = []
    cases = [
        ("matmul_3x5", 5, 3, 1e-6),
        ("matmul_17x31", 31, 17, 2e-6),
        ("matmul_768x288", 288, 768, 2e-5),
    ]

    for name, in_features, out_features, tolerance in cases:
        x = rng.normal(0.0, 0.5, size=in_features).astype(np.float32)
        weights = rng.normal(
            0.0,
            0.25,
            size=(out_features, in_features),
        ).astype(np.float32)

        x_path = directory / f"{name}_x.bin"
        weights_path = directory / f"{name}_weights.bin"
        output_path = directory / f"{name}_output.bin"
        write_float32(x_path, x)
        write_float32(weights_path, weights)

        run_command(
            [
                str(runner),
                "matmul",
                str(x_path),
                str(weights_path),
                str(output_path),
                str(in_features),
                str(out_features),
            ]
        )

        actual = read_float32(output_path, out_features)
        reference = torch.from_numpy(weights) @ torch.from_numpy(x)
        results.append(compare(name, actual, reference, tolerance))

    return results


def align_rmsnorm(runner: Path, directory: Path, rng: np.random.Generator) -> list[Result]:
    results = []
    for size in (4, 288, 768):
        name = f"rmsnorm_{size}"
        x = rng.normal(0.0, 1.5, size=size).astype(np.float32)
        weight = rng.normal(1.0, 0.2, size=size).astype(np.float32)

        x_path = directory / f"{name}_x.bin"
        weight_path = directory / f"{name}_weight.bin"
        output_path = directory / f"{name}_output.bin"
        write_float32(x_path, x)
        write_float32(weight_path, weight)

        run_command(
            [
                str(runner),
                "rmsnorm",
                str(x_path),
                str(weight_path),
                str(output_path),
                str(size),
            ]
        )

        actual = read_float32(output_path, size)
        x_tensor = torch.from_numpy(x)
        weight_tensor = torch.from_numpy(weight)
        reference = weight_tensor * x_tensor * torch.rsqrt(
            torch.mean(x_tensor * x_tensor) + 1e-5
        )
        results.append(compare(name, actual, reference, 2e-6))

    return results


def align_softmax(runner: Path, directory: Path, rng: np.random.Generator) -> list[Result]:
    results = []
    cases = [
        ("softmax_7", rng.normal(0.0, 2.0, size=7).astype(np.float32)),
        ("softmax_256", rng.normal(0.0, 4.0, size=256).astype(np.float32)),
        (
            "softmax_large_logits",
            (1000.0 + rng.normal(0.0, 3.0, size=257)).astype(np.float32),
        ),
    ]

    for name, values in cases:
        input_path = directory / f"{name}_input.bin"
        output_path = directory / f"{name}_output.bin"
        write_float32(input_path, values)

        run_command(
            [
                str(runner),
                "softmax",
                str(input_path),
                str(output_path),
                str(values.size),
            ]
        )

        actual = read_float32(output_path, values.size)
        reference = torch.softmax(torch.from_numpy(values), dim=0)
        results.append(compare(name, actual, reference, 5e-7))

    return results


def print_results(results: list[Result]) -> None:
    print(f"{'case':<24} {'max_abs':>12} {'mean_abs':>12} {'tolerance':>12} status")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.name:<24} "
            f"{result.max_abs_error:>12.4e} "
            f"{result.mean_abs_error:>12.4e} "
            f"{result.tolerance:>12.4e} "
            f"{status}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ZeroInfer operators against PyTorch float32 references."
    )
    parser.add_argument("--runner", required=True, type=Path)
    args = parser.parse_args()

    runner = args.runner.resolve()
    if not runner.is_file():
        parser.error(f"runner does not exist: {runner}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260609)

    with tempfile.TemporaryDirectory(prefix="zeroinfer-pytorch-") as temporary:
        directory = Path(temporary)
        results = [
            *align_matmul(runner, directory, rng),
            *align_rmsnorm(runner, directory, rng),
            *align_softmax(runner, directory, rng),
        ]

    print(f"PyTorch {torch.__version__}, NumPy {np.__version__}, device=cpu")
    print_results(results)

    failures = [result for result in results if not result.passed]
    if failures:
        print(f"{len(failures)} alignment case(s) failed")
        return 1

    print(f"All {len(results)} PyTorch alignment cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
