import time
from tqdm import tqdm
from threading import Thread, Event
import subprocess as sp
import re
import copy
import numpy as np


T = 'temperature'
U = 'utilization'
M = 'memory'
TM = 'total_memory'
MP = 'memory_percentage'


class GPUStat:
    def __init__(self):
        self._running = Event()
        self._thread = None
        self.result_dict = {}
        self.stat_template = {
            T: [],
            U: [],
            M: [],
            TM: [],
            MP: [],
        }

    def _read_gpustat(self, interval, running):
        GPU_RE = re.compile(r"\[(\d)\][\w\d -]*\| (\d+)'C,[\s]+(\d+) \% \|[\s]+(\d+) \/ (\d+) MB \|[\w\d\s\(\)]*")
        '''
        input string : "[0] GeForce RTX 2080 Ti | 41'C,   0 % |  1119 / 11019 MB | root(1115M)"
        output match : [('0', '41', '0', '1119', '11019')]
        '''

        pts = sp.Popen(["gpustat", '--interval', str(interval)], stdout=sp.PIPE)
        try:
            # Reading loop
            while running.is_set():
                if pts.poll() is not None:
                    continue
                out = pts.stdout
                if out is not None:
                    # Read line process output
                    text = out.readline().decode("utf-8")
                    # Decode line in UTF-8
                    
                    stats = re.findall(GPU_RE, text)
                    if len(stats) > 0:
                        (gpuid, temperature, utilization, memory, total_memory) = stats[0]
                        gpuid = int(gpuid)
                        if gpuid not in self.result_dict:
                            self.result_dict[gpuid] = copy.deepcopy(self.stat_template)
                        self.result_dict[gpuid][T].append(int(temperature))
                        self.result_dict[gpuid][U].append(int(utilization))
                        self.result_dict[gpuid][M].append(int(memory))
                        self.result_dict[gpuid][TM].append(int(total_memory))
                        self.result_dict[gpuid][MP].append(int(memory) / int(total_memory))
        finally:
            # Kill process
            try:
                pts.kill()
            except OSError:
                pass

    def all_stat(self):
        for gpu, stat in self.result_dict.items():
            for name, value in stat.items():    
                print(f"GPU[{gpu}]-{name}: {value}")

    def summary(self, gpuid=None):
        for gpu, stat in self.result_dict.items():
            if gpuid is not None and gpuid != gpu:
                continue
            for name, value in stat.items():
                print("GPU[{}]-{}: {:.1f}, 10th percentile: {:.1f}, 50th percentile: {:.1f}, 90th percentile: {:.1f}".format(gpu, name, 
                    np.average(value), 
                    np.percentile(value, 10),
                    np.percentile(value, 50), 
                    np.percentile(value, 90)))

    def open(self, interval=1):
        """Start reading gpu stats, with default interval 60s.
        """
        if self._thread is not None:
            return False
        # Set timeout
        interval = int(interval * 60)
        # Check if thread or process exist
        self._running.set()
        # Start thread Service client
        self._thread = Thread(target=self._read_gpustat, args=(interval, self._running, ))
        self._thread.start()
        return True

    def close(self, timeout=None):
        # Check if thread and process are already empty
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        return True


if __name__=="__main__":
    gpustat = GPUStat()
    gpustat.open()
    time.sleep(6)
    gpustat.summary()
    gpustat.all_stat()
    gpustat.close()
