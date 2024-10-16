import json
import multiprocessing
import concurrent
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from consumer import Consumer
from parallel_generator import MultithreadedGenerator
import argparse


class MultithreadedConsumer:
    def __init__(self, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)

    # Detect anomalies using multiple threads
    def detect_anomaly(self, data, type="zscore", params={"threshold": 3}):
        def _zscore(data, out, first, last, threshold):
            mean = np.mean(data[first:last])
            std = np.std(data[first:last])
            out[first:last] = np.greater(np.abs((data[first:last] - mean) / std), threshold)

        def _stl(data, out, first, last, params):
            seasonal_period = params.get("seasonal", 7)
            trend = np.convolve(data[first:last], np.ones(seasonal_period) / seasonal_period, mode="valid")
            trend = np.concatenate((trend, np.full(len(data[first:last]) - len(trend), trend[-1])))
            seasonal = data[first:last] - trend
            resid = data[first:last] - trend - seasonal
            threshold = params.get("threshold", 3)
            out[first:last] = np.greater(np.abs(resid), threshold * np.std(resid))

        ## If allowed, we can use statsmodels.tsa.seasonal.STL which is 10x faster
        
        # def _stl(data, out, first, last, params):
        #     stl = STL(data[first:last], seasonal=params.get("seasonal", 7))
        #     result = stl.fit()
        #     resid = result.resid
        #     threshold = params.get("threshold", 3)
        #     out[first:last] = np.greater(np.abs(resid), threshold * np.std(resid))

        def _detect_anomaly(data, out, first, last, type, params):
            if type == "zscore":
                _zscore(data, out, first, last, params["threshold"])
            elif type == "stl":
                _stl(data, out, first, last, params)

        step = min(len(data) // self.threads, 10**7)
        futures = {}
        too_large = False
        if len(data) > 10**8:
            anomaly_mask = np.memmap("anomaly_mask.dat", dtype=bool, mode="w+", shape=(len(data),))
        else:
            anomaly_mask = np.empty(len(data), dtype=bool)
        for i in range(len(data) // step + 1):
            args = (
                _detect_anomaly,
                data,
                anomaly_mask,
                i * step,
                (i + 1) * step,
                type,
                params,
            )
            futures[self.executor.submit(*args)] = i

        concurrent.futures.wait(futures)
        return anomaly_mask

    # This method is used to change the number of threads dynamically
    def set_threads(self, threads):
        self.executor.shutdown(False)
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.threads = threads

    def detect_anomaly_from_file(self, file_path, type="zscore", params={"threshold": 3}):
        # We need order to be preserved when reading from the file
        # Multithreading will not be as effective in this case

        # Check if file size is too large
        file_size = os.path.getsize(file_path)
        too_large = False
        if file_size > 10**8:
            too_large = True

        if too_large:
            data = np.memmap(file_path, dtype="float64", mode="r")
        else:
            data = np.fromfile(file_path, dtype="float64")
        return len(data), self.detect_anomaly(data, type=type, params=params)

    def __del__(self):
        self.executor.shutdown(False)


if __name__ == "__main__":

    def demo():
        def compare_throughput():
            n = 10000000
            generator = MultithreadedGenerator(threads=16)
            data = generator.generate(n, type="normal", params={"mean": 0, "std": 1})
            num_workers_list = list(range(1, 33))  # Different number of workers to test
            worker_times = []

            # Consumer instance
            consumer = Consumer(type="zscore", params={"threshold": 3})

            start_time = time.time()
            consumer.detect_anomaly(data)
            consumer_time = time.time() - start_time

            print(f"Consumer throughput: {n / consumer_time} numbers/second")

            # Measure throughput for different number of workers
            consumer = MultithreadedConsumer()
            for num_threads in num_workers_list:
                consumer.set_threads(num_threads)
                start_time = time.time()
                consumer.detect_anomaly(data)
                worker_times.append(time.time() - start_time)
                print(f"Consumer throughput with {num_threads} threads: {n / worker_times[-1]} numbers/second")

            plt.plot(num_workers_list, [n / t for t in worker_times], marker="o")
            plt.axhline(y=n / consumer_time, color="r", linestyle="--")
            plt.xlabel("Number of threads")
            plt.ylabel("Throughput (numbers/second)")
            plt.title("Throughput comparison")
            plt.show()

        rng = MultithreadedGenerator(seed=42, threads=8)
        data = rng.generate(1000)
        consumer = MultithreadedConsumer(threads=8)
        anomaly = consumer.detect_anomaly(data, type="zscore", params={"threshold": 2})
        # Highlight anomalies
        anomaly_indices = np.where(anomaly)[0]
        plt.scatter(range(len(data)), data, color="blue", s=5, label="Data")
        plt.scatter(anomaly_indices, data[anomaly_indices], color="red", s=50, label="Anomalies")

        # Customize the plot
        plt.title("Data with Anomalies")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

        compare_throughput()

    parser = argparse.ArgumentParser(description="Multithreaded anomaly detection")
    parser.add_argument("--demo", action="store_true", help="Run the demo function")
    parser.add_argument("--file", type=str, help="File path for anomaly detection")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--generator-type", type=str, default="normal", help="Type of data generator")
    parser.add_argument("--generator-params", type=str, default='{"mean": 0, "std": 1}', help="Parameters for data generator in JSON format")
    parser.add_argument("--params", type=str, default='{"threshold": 3}', help="Parameters for anomaly detection in JSON format")
    parser.add_argument("--time", action="store_true", help="Report the time of execution")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use")
    parser.add_argument("--type", type=str, default="zscore", help="Type of anomaly detection method")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.file:
        consumer = MultithreadedConsumer(threads=args.threads)
        start_time = time.time()
        n, anomaly = consumer.detect_anomaly_from_file(args.file, type=args.type, params=json.loads(args.params))
        if args.time:
            print(f"Execution time: {time.time() - start_time} seconds, Throughput: {n / (time.time() - start_time)} numbers/second")
        print(f"Anomalies detected: {np.where(anomaly)[0]}")
    else:
        generator = MultithreadedGenerator(seed=42, threads=8)
        data = generator.generate(args.num_samples, type=args.generator_type, params=json.loads(args.generator_params))
        consumer = MultithreadedConsumer(threads=8)
        start_time = time.time()
        anomaly = consumer.detect_anomaly(data, type=args.type, params=json.loads(args.params))
        if args.time:
            print(f"Execution time: {time.time() - start_time} seconds, Throupughput: {args.num_samples / (time.time() - start_time)} numbers/second")
        print(f"Anomalies detected: {np.where(anomaly)[0]}")
