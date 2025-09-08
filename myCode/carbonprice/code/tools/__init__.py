import gc
import sys
import os.path
import threading
import time
import traceback
import functools
import tracemalloc

import pandas as pd
import numpy as np
import psutil
import xarray as xr
import numpy_financial as npf
import matplotlib.patches as patches

from typing import Tuple
from datetime import datetime
from matplotlib import pyplot as plt

import tools.config as config

task_name = config.TASK_NAME
task_dir = f'../../../output/{task_name}/carbon_price'
os.makedirs(task_dir, exist_ok=True)


class LogToFile:
    def __init__(self, log_path, mode: str = 'a'):
        self.log_path_stdout = f"{log_path}_stdout.log"
        self.log_path_stderr = f"{log_path}_stderr.log"
        self.mode = mode

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with open(self.log_path_stdout, self.mode, encoding='utf-8') as file_stdout, open(self.log_path_stderr,
                                                                            self.mode, encoding='utf-8') as file_stderr:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                try:
                    sys.stdout = self.StreamToLogger(file_stdout, original_stdout)
                    sys.stderr = self.StreamToLogger(file_stderr, original_stderr)
                    return func(*args, **kwargs)
                except Exception as e:
                    # Capture the full traceback
                    exc_info = traceback.format_exc()
                    # Log the traceback to stderr log before re-raising the exception
                    sys.stderr.write(exc_info + '\n')
                    raise  # Re-raise the caught exception to propagate it
                finally:
                    # Reset stdout and stderr
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

        return wrapper

    class StreamToLogger(object):
        def __init__(self, file, orig_stream=None):
            self.file = file
            self.orig_stream = orig_stream

        def write(self, buf):
            if buf.strip():  # Check if buf is just whitespace/newline
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_buf = f"{timestamp} - {buf}"
            else:
                formatted_buf = buf  # If buf is just a newline/whitespace, don't prepend timestamp

            # Write to the original stream if it exists
            if self.orig_stream:
                self.orig_stream.write(formatted_buf)

            # Write to the log file
            self.file.write(formatted_buf)
            # self.file.flush()

        def flush(self):
            # Ensure content is written to disk
            self.file.flush()


def log_memory_usage(output_dir=task_dir, mode='a', interval=1, stop_event=None):
    '''
    Log the memory usage of the current process to a file with enhanced accuracy.
    Parameters
        output_dir (str): The directory to save the memory log file.
        mode (str): The mode to open the file. Default is 'a' (append).
        interval (int): The interval in seconds to log the memory usage.
    '''

    with open(f'{output_dir}/mem_log.txt', mode=mode, encoding='utf-8') as file:
        while not stop_event.is_set():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            process = psutil.Process(os.getpid())

            # Get working set memory (most accurate) - ensure consistency across all processes
            memory_info = process.memory_info()

            # Check if working set is available on this system
            has_wset = hasattr(memory_info, 'wset')

            if has_wset:
                wset_memory = memory_info.wset
            else:
                wset_memory = memory_info.rss

            # Include child processes using the SAME metric type
            children = process.children(recursive=True)
            if children:
                for child in children:
                    try:
                        child_memory_info = child.memory_info()
                        if has_wset and hasattr(child_memory_info, 'wset'):
                            wset_memory += child_memory_info.wset
                        else:
                            # Use RSS for consistency if wset not available
                            wset_memory += child_memory_info.rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            # Write working set memory info (most accurate)
            wset_gb = wset_memory / (1024 * 1024 * 1024)

            file.write(f'{timestamp}\t{wset_gb:.3f}\n')
            file.flush()
            time.sleep(interval)


# Enhanced memory monitoring helper functions            
memory_log = []
monitoring = False
monitor_thread = None
baseline_memory = 0  # Store the baseline memory at start


def monitor_memory(interval=0.01):
    """
    Memory monitoring focused on Working Set delta from baseline.
    Runs in a thread, logs memory usage every `interval` seconds.
    """
    process = psutil.Process(os.getpid())

    while monitoring:
        try:
            memory_info = process.memory_info()

            # Check if working set is available and use consistently
            has_wset = hasattr(memory_info, 'wset')

            if has_wset:
                current_wset_mb = memory_info.wset / 1024 ** 2
            else:
                current_wset_mb = memory_info.rss / 1024 ** 2

            # Calculate delta from baseline
            delta_mb = current_wset_mb - baseline_memory

            # Store delta memory info
            memory_log.append({
                'time': time.time(),
                'wset_mb': current_wset_mb,
                'delta_mb': delta_mb
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

        time.sleep(interval)


def start_memory_monitor():
    """
    Start Working Set memory monitoring with baseline measurement.
    Clears previous logs and starts monitoring from current memory usage.
    """
    global monitoring, monitor_thread, baseline_memory

    # Clear previous log
    memory_log.clear()

    # Force garbage collection to get clean baseline
    gc.collect()

    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    has_wset = hasattr(memory_info, 'wset')
    if has_wset:
        baseline_memory = memory_info.wset / 1024 ** 2
    else:
        baseline_memory = memory_info.rss / 1024 ** 2

    # Start monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    print("Delta memory monitoring started")


def stop_memory_monitor():
    """
    Stop memory monitoring and return delta analysis.
    Returns a plot showing only the incremental memory usage.
    """
    global monitoring

    monitoring = False
    if monitor_thread:
        monitor_thread.join()

    if not memory_log:
        print("No memory data collected")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(memory_log)
    df['Time'] = df['time'] - df['time'].min()

    # Delta memory plot (main focus)
    plt.plot(df['Time'], df['delta_mb'])
    plt.xlabel('Time (s)')
    plt.ylabel('Delta Memory (MB)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{task_dir}/memory_delta_plot.png', dpi=300)

    return plt