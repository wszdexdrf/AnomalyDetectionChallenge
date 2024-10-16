#! /usr/bin/python3
import time
from matplotlib import pyplot as plt
from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np
from generator import Generator
import argparse
import json


class MultithreadedGenerator:
    def __init__(self, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        # Create a SeedSequence object for seeding the random number generators
        self.seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in self.seq.spawn(threads)]

        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    # Generate random numbers using multiple threads
    def generate(self, n, type="normal", params=None, seasonality=None):
        def _normal_fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        def _uniform_fill(random_state, out, first, last):
            random_state.random(out=out[first:last])

        def _translate_normal(values, mean, std):
            return values * std + mean

        def _translate_uniform(values, low, high):
            return values * (high - low) + low

        def _fill(random_state, out, first, last, type):
            if type == "normal":
                _normal_fill(random_state, out, first, last)
            elif type == "uniform":
                _uniform_fill(random_state, out, first, last)
            else:
                raise ValueError("Unknown generator type")

        values = np.empty(n)
        step = n // self.threads
        futures = {}
        for i in range(self.threads):
            args = (
                _fill,
                self._random_generators[i],
                values,
                i * step,
                (i + 1) * step,
                type,
            )
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)
        # translate from standard normal to normal with mean and std
        if params is not None and type == "normal":
            mean = params["mean"]
            std = params["std"]
            values = _translate_normal(values, mean, std)

        # translate from standard uniform to uniform with low and high
        elif params is not None and type == "uniform":
            low = params["low"]
            high = params["high"]
            values = _translate_uniform(values, low, high)

        # Add seasonality if provided
        if seasonality is not None:
            amplitude = seasonality["amplitude"] if "amplitude" in seasonality else 1
            phase = seasonality["phase"] if "phase" in seasonality else 0
            period = seasonality["period"] if "period" in seasonality else 1
            values += amplitude * np.sin(2 * np.pi * np.arange(n) / period + phase)
        return values

    # This function can be used to dynamically change the number of threads
    def set_threads(self, threads):
        old_threads = self.threads
        self.executor.shutdown(False)
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.threads = threads
        # If the number of threads is increased, create new random generators
        # Keep the old random generators for predictability
        if threads > old_threads:
            self._random_generators.extend(default_rng(s) for s in self.seq.spawn(threads - old_threads))

    # Generate random numbers and save them to a file
    # Optimized for an SSD or fast storage with low latency
    def generate_to_file(self, n, type="normal", params=None, seasonality=None, file_path="data.npy"):
        chunk_size = n // self.threads
        futures = []
        # Use a semaphore to limit the number of chunks in memory
        semaphore = multiprocessing.Semaphore(10)

        # Delete any existing file
        open(file_path, "w").close()

        def _save_chunk(chunk_size, type, params, file_path):
            with semaphore:
                data = self.generate(chunk_size, type, params, seasonality)
                with open(file_path, "ab") as f:
                    f.write(data.tobytes())

        with concurrent.futures.ThreadPoolExecutor() as write_executor:
            for i in range(self.threads):
                futures.append(write_executor.submit(_save_chunk, chunk_size, type, params, file_path))
            concurrent.futures.wait(futures)

    # Generate random numbers and save them to a file synchronously
    # Better for HDD or slow storage with high latency
    # Necessary for ordered data streams
    def generate_to_file_sync(self, n, type="normal", params=None, seasonality=None, file_path="data.npy"):
        chunk_size = 10**7
        # Delete any existing file
        open(file_path, "w").close()
        f = open(file_path, "ab")
        print("Writing array of size", n)
        count = 0
        for i in range(0, n, chunk_size):
            data = self.generate(min(chunk_size, n - i), type, params, seasonality)
            count += len(data)
            f.write(data.tobytes())
        f.close()
    def __del__(self):
        self.executor.shutdown(False)


if __name__ == "__main__":

    def demo():
        # Throughput comparison
        def compare_throughput():
            n = 10000000  # Number of random numbers to generate
            num_workers_list = list(range(1, 33))  # Different number of workers to test
            worker_times = []

            # Sampler instance
            sampler = Generator(type="normal", params={"mean": 0, "std": 1})

            # Measure time for Sampler
            start_time = time.time()
            sampler.generate(n)
            sampler_time = time.time() - start_time

            print(f"Sampler throughput: {n / sampler_time} numbers/second")

            # Measure throughput for different number of workers
            rng = MultithreadedGenerator(threads=4)
            for num_workers in num_workers_list:
                start_time = time.time()
                rng.set_threads(num_workers)
                rng.generate_to_file(n)
                worker_time = time.time() - start_time
                worker_times.append(worker_time)
                print(f"Number of workers: {num_workers}, Throughput: {n / worker_time} numbers/second")

            plt.plot(num_workers_list, [n / t for t in worker_times], marker="o")
            plt.axhline(y=n / sampler_time, color="r", linestyle="-", label="Sampler throughput")
            plt.xlabel("Number of workers")
            plt.ylabel("Throughput (numbers/second)")
            plt.title("Throughput comparison")
            plt.show()

        rng = MultithreadedGenerator(seed=42, threads=4)
        rng.set_threads(8)
        samples = rng.generate(1000, type="uniform", params={"low": 0, "high": 1}, seasonality={"amplitude": 0.1, "period": 200})
        plt.plot(samples, marker="o", linestyle="")
        plt.show()

        compare_throughput()

    parser = argparse.ArgumentParser(description="Multithreaded random number generator")
    parser.add_argument("--demo", action="store_true", help="Run the demo function")
    parser.add_argument("--file", type=str, help="File path to save generated numbers")
    parser.add_argument(
        "--type",
        type=str,
        default="normal",
        help="Type of random numbers to generate",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Parameters for the random number generator in JSON format",
    )
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use")
    parser.add_argument("--time", action="store_true", help="Measure and Report time")
    parser.add_argument("--sync", action="store_true", help="Use synchronous file writing")
    parser.add_argument("--seasonality", type=str, help="Seasonality parameters in JSON format")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        rng = MultithreadedGenerator(seed=42, threads=args.threads)
        params = json.loads(args.params) if args.params else None
        seasonality = json.loads(args.seasonality) if args.seasonality else None
        if args.time:
            start_time = time.time()
        if args.file:
            if args.sync:
                rng.generate_to_file_sync(args.num_samples, type=args.type, params=params, file_path=args.file, seasonality=seasonality)
            else:
                rng.generate_to_file(args.num_samples, type=args.type, params=params, file_path=args.file, seasonality=seasonality)
        else:
            samples = rng.generate(args.num_samples, type=args.type, params=params, seasonality=seasonality)
            plt.plot(samples, marker="o", linestyle="")
            plt.show()
        if args.time:
            print(f"Time taken: {time.time() - start_time} seconds, Throughput: {args.num_samples / (time.time() - start_time)} numbers/second")
