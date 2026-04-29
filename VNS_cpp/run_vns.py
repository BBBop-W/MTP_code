import subprocess
import time
import os
import signal

p = subprocess.Popen(["./vns_solver", "m11c11"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
t0 = time.time()
while time.time() - t0 < 10:
    # read non-blocking?
    pass

p.send_signal(signal.SIGINT)
out, _ = p.communicate()
print(out)
