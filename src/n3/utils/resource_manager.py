import threading
import subprocess
import numpy as np

class ResourceManager:
    def __init__(self, max_parallel=4):
        self.semaphore = threading.Semaphore(max_parallel)

    def __enter__(self):
        self.semaphore.acquire()
        return self

    def __exit__(self, *args):
        self.semaphore.release()

    @staticmethod
    def get_available_gpu():
        """Find GPU with most free memory"""
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,nounits,noheader"
            ]).decode().strip().split('\n')
            return int(np.argmax([int(x) for x in output]))
        except subprocess.CalledProcessError:
            return 0
