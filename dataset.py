import numpy as np

if __name__ == "__main__":
    dataset_directory = "./dataset/"
    dataset_size = 100000
    dataset_chunksize = 64 + 32
    frequency_range = [200.0, 1000.0]
    sampling_frequency = 48000.0

    for i in range(dataset_size):
        f = np.random.rand() * (frequency_range[1] - frequency_range[0]) + frequency_range[0]
        p = np.random.rand() * 2 * np.pi
        x = np.sin(2 * np.pi * f * np.arange(dataset_chunksize) / sampling_frequency + p)
        x = 127.5 * x + 127.5;
        x = x.astype(np.int64)
        np.save(dataset_directory + f"{i}.npy", x)
