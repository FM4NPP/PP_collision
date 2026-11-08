"""
acts.py: the ACTS worker (scripts/acts/acts_worker.py): one persistent process per configuration, in the ACTS
container, that propagates track states (and refits their hits) for TrackBuilder(use_acts=True).
"""
import atexit
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

ACTS_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                  "scripts", "acts", "acts_worker.py")
_ACTS_WORKERS = {}
# seconds spent starting and running ACTS workers (the eval resets and reports it)
ACTS_TIMING = {"start_s": 0.0, "propagate_s": 0.0, "states": 0, "restarts": 0}

class _ActsWorker:
    """
    One persistent ACTS worker process (scripts/acts/acts_worker.py), started in the background; the first
    propagate waits until it is ready. Its output (ROOT's and ACTS's start-up messages) goes to a log file, not
    the terminal: its failure summary is printed at close, the log's end with an error.
    """
    def __init__(self, command, config):
        self.command = command
        self.ready = self.keep_log = False
        fd, self.log_path = tempfile.mkstemp(prefix="acts_worker_", suffix=".log")
        os.close(fd)
        with open(self.log_path, "ab") as log:      # appending: the worker's writes never overwrite each other
            self.process = subprocess.Popen(list(command) + [ACTS_WORKER_SCRIPT, json.dumps(config)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)

    def log(self, lines=None):
        """The worker's output so far (its last lines with lines)."""
        with open(self.log_path, errors="replace") as f:
            text = f.read().splitlines()
        return "\n".join(text[-lines:] if lines else text)

    def failure(self, what):
        """RuntimeError for a worker that stopped, with the end of its output (the log is kept)."""
        self.keep_log = True
        return RuntimeError(f"ACTS worker {what} (code {self.process.wait()}): {' '.join(self.command)}; the end of "
                            f"its output ({self.log_path}):\n{self.log(40)}")

    def wait_ready(self):
        """Waits for the worker's ready line; the wait counts as start time."""
        t0 = time.perf_counter()
        while not self.ready:           # start-up output on stdout (e.g. ACTS's setup) until the ready line
            line = self.process.stdout.readline()
            if not line:
                raise self.failure("exited during start-up")
            self.ready = line.strip() == b"ACTS_WORKER_READY"
            if not self.ready:
                with open(self.log_path, "ab") as log:
                    log.write(line)
        ACTS_TIMING["start_s"] += time.perf_counter() - t0

    def propagate(self, position, direction, qop, covariance, hit_track, hits):
        """
        States in mm (position, direction (n, 3), qop (n,), covariance (n, 6, 6)) propagated backward to the
        perigee; with the Kalman refit, their hits first (hits (m, 9): position mm, the hit fit's direction there,
        the hit's r phi and z errors (mm) and 1 for a hit in the gas (0: silicon); hit_track (m,): their state's index,
        sorted; m = 0 without the refit). Returns parameters (n, 6), ok (n,),
        covariance (n, 6, 6), refitted (n,).
        """
        import numpy as np
        self.wait_ready()
        t0 = time.perf_counter()
        n = len(qop)
        self.process.stdin.write(struct.pack("<QQ", n, len(hit_track)) + b"".join(
            np.ascontiguousarray(a, "<f8").tobytes() for a in (position, direction, qop, covariance, hit_track, hits)))
        self.process.stdin.flush()
        reply = self.process.stdout.read(n * 338)
        if len(reply) != n * 338:
            raise self.failure("stopped")
        ACTS_TIMING["propagate_s"] += time.perf_counter() - t0
        ACTS_TIMING["states"] += n
        return (np.frombuffer(reply[: n * 48], "<f8").reshape(n, 6).copy(),
                np.frombuffer(reply[n * 336: n * 337], np.uint8).astype(bool),
                np.frombuffer(reply[n * 48: n * 336], "<f8").reshape(n, 6, 6).copy(),
                np.frombuffer(reply[n * 337:], np.uint8).astype(bool))

    def close(self):
        """Ends the worker and prints its failure summary (the lines from "ACTS worker:" on)."""
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        if not os.path.exists(self.log_path):       # closed before
            return
        text = self.log().splitlines()
        start = next((i for i, line in enumerate(text) if line.startswith("ACTS worker:")), len(text))
        if start < len(text):
            sys.stderr.write("\n".join(text[start:]) + "\n")
        if not self.keep_log:
            os.remove(self.log_path)

def _acts_worker(command, config):
    """
    The running worker for this command and config (started on first use, restarted if it died).
    """
    key = (tuple(command), json.dumps(config, sort_keys=True))
    if key not in _ACTS_WORKERS or _ACTS_WORKERS[key].process.poll() is not None:
        if key in _ACTS_WORKERS:
            # a worker that died between batches: its log kept, a new one started
            dead = _ACTS_WORKERS[key]
            dead.keep_log = True
            sys.stderr.write(f"ACTS worker exited (code {dead.process.poll()}) between batches; restarting; its log: "
                             f"{dead.log_path}\n")
            ACTS_TIMING["restarts"] += 1
            dead.close()
        _ACTS_WORKERS[key] = _ActsWorker(command, config)
    return _ACTS_WORKERS[key]

atexit.register(lambda: [worker.close() for worker in _ACTS_WORKERS.values()])
