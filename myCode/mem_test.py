from joblib import Parallel, delayed
import time

def allocate_memory_block(index, gb_per_job=50):
    allocation_size = gb_per_job * 1024**3
    memory_hog = []
    while True:
        try:
            memory_hog.append(bytearray(allocation_size))
            print(f"[Job {index}] Allocated {len(memory_hog) * gb_per_job} GB total.")
            time.sleep(0.1)
        except MemoryError:
            print(f"[Job {index}] MemoryError! Restarting...")
            memory_hog.clear()
            time.sleep(1)
            return allocate_memory_block(index, gb_per_job)

if __name__ == "__main__":
    n_jobs = 8  # 并行进程数，可根据 CPU 和 RAM 资源调整
    gb_per_job = 50

    Parallel(n_jobs=n_jobs)(
        delayed(allocate_memory_block)(i, gb_per_job) for i in range(n_jobs)
    )