import multiprocessing
import time

from matplotlib import pyplot as plt
import numpy as np

from parallel_consumer import MultithreadedConsumer
from parallel_generator import MultithreadedGenerator


# This function is used to serially generate random numbers and detect anomalies
# This is done in chunks of size chunk_size
def serial(
    n=10**9,
    chunk_size=3 * 10**7,
    generator_type="normal",
    generator_params={"mean": 0, "std": 1},
    consumer_type="zscore",
    consumer_params={"threshold": 3},
):
    thread_count = multiprocessing.cpu_count() * 2
    generator = MultithreadedGenerator(threads=thread_count)
    consumer = MultithreadedConsumer(threads=thread_count)
    anomaly_indexes = []
    for i in range(0, n, chunk_size):
        data_chunk = generator.generate(min(chunk_size, n - i), type=generator_type, params=generator_params)
        anomalies = consumer.detect_anomaly(data_chunk, type=consumer_type, params=consumer_params)
        anomaly_indexes.extend(np.where(anomalies)[0] + i)
    return anomaly_indexes


if __name__ == "__main__":
    n = 10**8
    chunk_sizes = range(10**6, 5 * 10**7 + 1, 10**6)
    serial_throughputs = []
    for chunk_size in chunk_sizes:
        start_time = time.time()
        _ = serial(n, chunk_size)
        throughput = n / (time.time() - start_time)
        serial_throughputs.append(throughput)
        print(f"Throughput for chunk size {chunk_size}: {throughput} numbers/second")

    plt.plot(chunk_sizes, serial_throughputs)
    plt.xlabel("Chunk size")
    plt.ylabel("Throughput (numbers/second)")
    plt.title("Throughput comparison")
    plt.show()
