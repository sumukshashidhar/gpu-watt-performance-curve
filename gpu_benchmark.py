#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpu_benchmark.py

A script to measure and visualize the relationship between GPU wattage and TFLOPS
across different precision modes in PyTorch. Designed for NVIDIA GPUs on Linux.

Optional Features:
    1. Multiple GPU support (select which GPU index or run benchmarks on multiple indices).
    2. Offline plotting from existing results (CSV or JSON) without running new benchmarks.
    3. Extended precision modes (e.g., BF16, FP32, TF32, FP8, FP4, INT4, etc.) for experimentation.
       Note that not all modes are natively supported in standard PyTorch. Some modes (like FP8, FP4, INT4)
       may require specialized kernels or custom quantization methods. This example includes placeholders
       to demonstrate how you might wire them in, but you'll need to adapt or install libraries that
       provide these capabilities.

Requirements:
    - Python 3.9+
    - PyTorch
    - Matplotlib
    - loguru
    - pandas
    - An NVIDIA GPU with appropriate drivers and nvidia-smi installed

Example Usage:
    # Run a benchmark on GPU index 1, from 150W to 300W, stepping by 50:
    python gpu_benchmark.py --gpu-index 1 \
                            --min-wattage 150 \
                            --max-wattage 300 \
                            --wattage-step 50

    # Only generate plots from an existing CSV:
    python gpu_benchmark.py --plot-only --results-csv existing_results.csv

    # Run with extended precision modes (warning: some modes may require specialized setups):
    python gpu_benchmark.py --precision-modes bf16 fp32 tf32 fp8 fp4 int4
"""

import argparse
import subprocess
import time
import json
import datetime
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from typing import Optional

@dataclass
class BenchmarkConfig:
    """
    Holds configuration parameters for the GPU benchmark.
    """
    gpu_indices: list[int] = field(default_factory=lambda: [0])  # Support multiple GPUs
    min_wattage: int = 100
    max_wattage: int = 390
    wattage_step: int = 10
    temperature_threshold: int = 55    # Wait until GPU temperature <= threshold
    benchmark_duration: int = 60       # Duration of each benchmark in seconds
    precision_modes: list[str] = field(default_factory=lambda: ["fp32", "tf32", "bf16"])
    results_save_path: Path = Path("gpu_benchmark_results.csv")
    json_save_path: Path = Path("gpu_benchmark_results.json")
    plot_save_path: Path = Path("tflops_vs_watts.png")
    baseline_wait_interval: int = 5    # Number of seconds to wait between temp checks
    matrix_size: int = 8192           # Size for matrix multiplication (N x N)
    plot_only: bool = False           # If True, only read existing data and plot
    existing_results_csv: Optional[str] = None
    existing_results_json: Optional[str] = None


@dataclass
class BenchmarkResult:
    """
    Stores the result of a single benchmark run.
    """
    timestamp: str
    gpu_index: int
    wattage: int
    precision_mode: str
    tflops: float
    gpu_temp_before: int
    gpu_temp_after: int


class GpuBenchmark:
    """
    Primary class for orchestrating the GPU wattage and TFLOPS benchmarking process.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize the GpuBenchmark with a given configuration.
        """
        self.config: BenchmarkConfig = config
        self.results: list[BenchmarkResult] = []

    def run(self) -> None:
        """
        Main entry point to execute the complete benchmark if `plot_only` is False.
        Otherwise, skip directly to reading existing data and plotting.
        """
        if self.config.plot_only:
            logger.info("Skipping benchmark. Only generating plots from existing results.")
            self._load_existing_results()
            self._generate_plot()
            return

        logger.info("Starting GPU Benchmark...")

        # For each GPU, run full suite (wattage, precision modes)
        for gpu_index in self.config.gpu_indices:
            logger.info(f"Preparing to benchmark GPU index {gpu_index}...")
            self._ensure_temp_is_under_threshold(gpu_index)

            for wattage in range(self.config.min_wattage,
                                 self.config.max_wattage + 1,
                                 self.config.wattage_step):
                logger.info(f"[GPU {gpu_index}] Setting power limit to {wattage}W.")
                self._set_power_limit(gpu_index, wattage)

                # Ensure GPU temperature is stable before running the benchmark
                self._ensure_temp_is_under_threshold(gpu_index)

                # Run benchmarks for each precision mode
                for mode in self.config.precision_modes:
                    gpu_temp_before: int = self._get_gpu_temperature(gpu_index)
                    tflops_val: float = self._run_single_benchmark(mode, gpu_index)
                    gpu_temp_after: int = self._get_gpu_temperature(gpu_index)
                    timestamp_str: str = datetime.datetime.now().isoformat()

                    # Store the result
                    result = BenchmarkResult(
                        timestamp=timestamp_str,
                        gpu_index=gpu_index,
                        wattage=wattage,
                        precision_mode=mode,
                        tflops=tflops_val,
                        gpu_temp_before=gpu_temp_before,
                        gpu_temp_after=gpu_temp_after
                    )
                    self.results.append(result)
                    logger.info(f"[GPU {gpu_index}] Completed {mode} benchmark at {wattage}W: "
                                f"{tflops_val:.3f} TFLOPS")

        logger.info("All benchmarks completed. Saving results...")
        self._save_results()
        self._generate_plot()
        logger.info("GPU Benchmark process finished successfully.")

    def _ensure_temp_is_under_threshold(self, gpu_index: int) -> None:
        """
        Blocks execution until the GPU temperature is below or equal to the configured threshold.
        This helps maintain a consistent thermal baseline before each test.
        """
        while True:
            current_temp: int = self._get_gpu_temperature(gpu_index)
            if current_temp <= self.config.temperature_threshold:
                logger.debug(f"[GPU {gpu_index}] Temperature {current_temp}°C <= threshold "
                             f"{self.config.temperature_threshold}°C. Continuing.")
                break
            logger.debug(f"[GPU {gpu_index}] Temperature {current_temp}°C > threshold "
                         f"{self.config.temperature_threshold}°C. Waiting...")
            time.sleep(self.config.baseline_wait_interval)

    def _set_power_limit(self, gpu_index: int, wattage: int) -> None:
        """
        Sets the GPU power limit using the `nvidia-smi` command.
        """
        try:
            command = [
                "sudo",
                "nvidia-smi",
                "-i", str(gpu_index),
                "-pl", str(wattage)
            ]
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"[GPU {gpu_index}] Failed to set power limit to {wattage}W: {e}")

    def _get_gpu_temperature(self, gpu_index: int) -> int:
        """
        Retrieves the current GPU temperature for the configured GPU index.
        """
        try:
            command = [
                "nvidia-smi",
                f"--query-gpu=temperature.gpu",
                "--format=csv,noheader",
                "-i", str(gpu_index)
            ]
            result = subprocess.run(command, capture_output=True, check=True, text=True)
            temperature_str: str = result.stdout.strip()
            temperature_val: int = int(temperature_str)
            return temperature_val
        except subprocess.CalledProcessError as e:
            logger.error(f"[GPU {gpu_index}] Failed to get GPU temperature: {e}")
            return -1

    def _run_single_benchmark(self, precision_mode: str, gpu_index: int) -> float:
        """
        Runs a single benchmark for a specified precision mode on a given GPU.
        Returns the measured TFLOPS.

        The strategy used here is to perform a large matrix multiplication of size N x N
        for config.benchmark_duration seconds, measuring how many operations are done.

        For extended modes like FP8, FP4, INT4, you'd need custom kernels or quantization
        flows not provided by default in PyTorch. The placeholders below illustrate the idea.
        """
        # Select device
        device = torch.device(f"cuda:{gpu_index}")

        # Configure global matmul settings (for TF32)
        # Note: Only relevant for fp32 or tf32; does not affect custom or quant modes
        if precision_mode == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            dtype = torch.float32
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            # Decide the dtype based on mode
            if precision_mode == "bf16":
                dtype = torch.bfloat16
            elif precision_mode == "fp16":
                dtype = torch.float16
            elif precision_mode == "int8":
                # INT8 typically requires custom quantization flows
                # For demonstration, let's fallback to int8 Tensors (less common in direct mm)
                dtype = torch.int8
            elif precision_mode == "fp8":
                # PyTorch doesn't natively support fp8 yet, placeholders
                # Would require specialized ops or a library such as NVIDIA's Transformer Engine.
                # We'll degrade to float16 for demonstration
                dtype = torch.float16
            elif precision_mode == "fp4":
                # Not natively supported, placeholder
                # Could degrade to float16 or implement custom approach
                dtype = torch.float16
            elif precision_mode == "int4":
                # Not natively supported, placeholder
                # We might degrade to int8 or do a custom quant
                dtype = torch.int8
            else:
                # Default to fp32
                dtype = torch.float32

        n: int = self.config.matrix_size
        a = torch.randn((n, n), device=device, dtype=dtype)
        b = torch.randn((n, n), device=device, dtype=dtype)

        # Warm up
        try:
            _ = torch.mm(a, b)
        except RuntimeError as e:
            logger.error(f"[GPU {gpu_index}][{precision_mode}] Error during warm-up mm: {e}")
            return 0.0

        # Perform repeated matmul for a fixed duration
        flops_per_mm: float = 2.0 * (n**3)
        start_time: float = time.time()
        ops_count: float = 0.0

        while True:
            # Some lower-precision modes may fail if PyTorch doesn't support mm on that dtype
            try:
                _ = torch.mm(a, b)
                ops_count += flops_per_mm
            except RuntimeError as e:
                logger.error(f"[GPU {gpu_index}][{precision_mode}] Error during matmul: {e}")
                break

            elapsed: float = time.time() - start_time
            if elapsed >= self.config.benchmark_duration:
                break

        # Compute TFLOPS
        tflops: float = (ops_count / elapsed) / 1e12 if elapsed > 0 else 0.0
        return tflops

    def _save_results(self) -> None:
        """
        Saves benchmark results to disk in both CSV and JSON formats for future analysis.
        """
        data_dict = {
            "timestamp": [],
            "gpu_index": [],
            "wattage": [],
            "precision_mode": [],
            "tflops": [],
            "gpu_temp_before": [],
            "gpu_temp_after": []
        }

        for r in self.results:
            data_dict["timestamp"].append(r.timestamp)
            data_dict["gpu_index"].append(r.gpu_index)
            data_dict["wattage"].append(r.wattage)
            data_dict["precision_mode"].append(r.precision_mode)
            data_dict["tflops"].append(r.tflops)
            data_dict["gpu_temp_before"].append(r.gpu_temp_before)
            data_dict["gpu_temp_after"].append(r.gpu_temp_after)

        df = pd.DataFrame(data_dict)
        df.to_csv(self.config.results_save_path, index=False)
        logger.info(f"Results saved to CSV at {self.config.results_save_path}")

        json_records = df.to_dict(orient="records")
        with open(self.config.json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_records, f, indent=4)
        logger.info(f"Results saved to JSON at {self.config.json_save_path}")

    def _load_existing_results(self) -> None:
        """
        Load existing results into self.results if plot_only is specified and 
        existing CSV/JSON files are provided.
        """
        loaded_any = False
        if self.config.existing_results_csv is not None:
            try:
                df = pd.read_csv(self.config.existing_results_csv)
                loaded_any = True
            except FileNotFoundError:
                logger.error(f"CSV file {self.config.existing_results_csv} not found.")
                return
        elif self.config.existing_results_json is not None:
            try:
                with open(self.config.existing_results_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                loaded_any = True
            except FileNotFoundError:
                logger.error(f"JSON file {self.config.existing_results_json} not found.")
                return
        else:
            logger.error("No existing CSV or JSON file provided. Cannot plot.")
            return

        if not loaded_any:
            logger.error("Failed to load any results. Cannot plot.")
            return

        # Convert loaded df to self.results
        self.results = []
        for idx, row in df.iterrows():
            try:
                result = BenchmarkResult(
                    timestamp=str(row["timestamp"]),
                    gpu_index=int(row["gpu_index"]),
                    wattage=int(row["wattage"]),
                    precision_mode=str(row["precision_mode"]),
                    tflops=float(row["tflops"]),
                    gpu_temp_before=int(row["gpu_temp_before"]),
                    gpu_temp_after=int(row["gpu_temp_after"]),
                )
                self.results.append(result)
            except Exception as e:
                logger.error(f"Skipping row due to format issue: {e}")

        logger.info(f"Loaded {len(self.results)} records from existing data.")

    def _generate_plot(self) -> None:
        """
        Generates and saves a Wattage vs. TFLOPS plot using Matplotlib.
        The Pareto-like point with the best TFLOPS/Watt ratio for each precision
        is highlighted on the plot, for each GPU index.
        """
        if len(self.results) == 0:
            logger.warning("No results to plot.")
            return

        # We'll plot separate lines for each GPU and precision combination.
        # Group results by (gpu_index, precision_mode).
        from collections import defaultdict
        data_map = defaultdict(list)
        # data_map[(gpu_index, precision_mode)] = [(wattage, tflops), ...]
        for r in self.results:
            data_map[(r.gpu_index, r.precision_mode)].append((r.wattage, r.tflops))

        plt.figure(figsize=(10, 8), dpi=300)
        legend_entries = []

        for (gpu_idx, mode), points in data_map.items():
            points.sort(key=lambda x: x[0])
            watts = [p[0] for p in points]
            tflops = [p[1] for p in points]
            label_str = f"GPU {gpu_idx} - {mode}"
            plt.plot(watts, tflops, marker='o', label=label_str)
            legend_entries.append(label_str)

            # Find best TFLOPS/Watt ratio to highlight as a simple "Pareto" proxy
            best_ratio = 0.0
            best_idx: Optional[int] = None
            for i, (w, t) in enumerate(points):
                ratio = t / w if w != 0 else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx is not None:
                plt.scatter(points[best_idx][0],
                            points[best_idx][1],
                            marker='*', s=200, c='red',
                            label=f"{label_str} best ratio")

        plt.title("GPU Wattage vs. TFLOPS (Multiple GPUs & Precision Modes)")
        plt.xlabel("Wattage (W)")
        plt.ylabel("TFLOPS")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.plot_save_path)
        logger.info(f"Plot saved to {self.config.plot_save_path}")


def parse_arguments() -> BenchmarkConfig:
    """
    Parse command-line arguments and return a BenchmarkConfig object.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark GPU TFLOPS vs. wattage and create a plot. "
                    "Optionally, only plot from existing results."
    )
    parser.add_argument("--gpu-index", type=int, nargs="+",
                        default=[0],
                        help="One or more GPU indices to benchmark. Default: 0")
    parser.add_argument("--min-wattage", type=int, default=100,
                        help="Minimum wattage for the test. Default: 100")
    parser.add_argument("--max-wattage", type=int, default=390,
                        help="Maximum wattage for the test. Default: 390")
    parser.add_argument("--wattage-step", type=int, default=10,
                        help="Step size for wattage increments. Default: 10")
    parser.add_argument("--temperature-threshold", type=int, default=55,
                        help="GPU temperature threshold for stable baseline. Default: 55")
    parser.add_argument("--benchmark-duration", type=int, default=60,
                        help="Time in seconds for each benchmark iteration. Default: 60")
    parser.add_argument("--precision-modes", type=str, nargs="+",
                        default=["fp32", "tf32", "bf16", "fp16", "int8", "fp8", "fp4", "int4"],
                        help="List of precision modes to test. E.g. bf16 fp32 tf32 fp8 fp4 int4")
    parser.add_argument("--results-csv", type=str, default="gpu_benchmark_results.csv",
                        help="Path to save or read the CSV results. Default: gpu_benchmark_results.csv")
    parser.add_argument("--results-json", type=str, default="gpu_benchmark_results.json",
                        help="Path to save or read the JSON results. Default: gpu_benchmark_results.json")
    parser.add_argument("--plot-file", type=str, default="tflops_vs_watts.png",
                        help="Path to save the plot. Default: tflops_vs_watts.png")
    parser.add_argument("--baseline-wait-interval", type=int, default=5,
                        help="Seconds to wait between temperature checks. Default: 5")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Matrix size (N x N) for matmul. Default: 8192")
    parser.add_argument("--plot-only", action="store_true",
                        help="If set, only read existing CSV/JSON and generate plot without running benchmarks.")
    parser.add_argument("--existing-results-csv", type=str, default=None,
                        help="Path to an existing CSV file for plot-only mode.")
    parser.add_argument("--existing-results-json", type=str, default=None,
                        help="Path to an existing JSON file for plot-only mode.")

    args = parser.parse_args()

    # Build the config
    config = BenchmarkConfig(
        gpu_indices=args.gpu_index,
        min_wattage=args.min_wattage,
        max_wattage=args.max_wattage,
        wattage_step=args.wattage_step,
        temperature_threshold=args.temperature_threshold,
        benchmark_duration=args.benchmark_duration,
        precision_modes=args.precision_modes,
        results_save_path=Path(args.results_csv),
        json_save_path=Path(args.results_json),
        plot_save_path=Path(args.plot_file),
        baseline_wait_interval=args.baseline_wait_interval,
        matrix_size=args.matrix_size,
        plot_only=args.plot_only,
        existing_results_csv=args.existing_results_csv,
        existing_results_json=args.existing_results_json
    )
    return config


def main() -> None:
    """
    Entry point of the script. Parses arguments, creates config, and runs the benchmark or plot.
    """
    # Configure logging
    logger.remove()  # Remove default logger
    logger.add(lambda msg: print(msg, flush=True), level="INFO")

    config = parse_arguments()
    benchmark = GpuBenchmark(config)

    try:
        benchmark.run()
    except Exception as e:
        logger.error(f"Unexpected error during benchmarking: {e}")


if __name__ == "__main__":
    main()
