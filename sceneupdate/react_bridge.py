"""
ReactWorkerBridge - IPC bridge to REACT worker subprocess
"""
import os
import json
import subprocess
import threading
import numpy as np

from config import (REACT_INPUT_PATH, REACT_OUTPUT_PATH,
                     REACT_SHUTDOWN_PATH, REACT_SG_SYNC_PATH,
                     REACT_WORKER_PYTHON, REACT_WORKER_SCRIPT)


class ReactWorkerBridge:
    def __init__(self, sg_json_path):
        self.sg_json_path = sg_json_path
        self.worker_proc = None
        self.last_output_mtime = 0
        self.last_output = {}
        self.current_sg = None
        with open(sg_json_path, 'r') as f:
            self.current_sg = json.load(f)

    def start_worker(self):
        import signal
        try:
            for line in subprocess.check_output(
                ["pgrep", "-f", "react_worker.py"], text=True
            ).strip().split("\n"):
                pid = int(line.strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
        except (subprocess.CalledProcessError, ValueError):
            pass
        for f in [REACT_INPUT_PATH, REACT_OUTPUT_PATH,
                  REACT_SHUTDOWN_PATH, REACT_SG_SYNC_PATH]:
            if os.path.exists(f): os.remove(f)
        import time
        log_file = open("/tmp/react_worker.log", "w")
        self.worker_proc = subprocess.Popen(
            [REACT_WORKER_PYTHON, "-u", REACT_WORKER_SCRIPT, self.sg_json_path],
            stdout=log_file, stderr=log_file, text=True)
        print(f"REACT Worker started (PID: {self.worker_proc.pid})")
        for i in range(600):
            if os.path.exists(REACT_SG_SYNC_PATH):
                print(f"REACT Worker ready (took {i*0.1:.1f}s)")
                return True
            time.sleep(0.1)
            if self.worker_proc.poll() is not None:
                print(f"REACT Worker died! Check /tmp/react_worker.log")
                return False
        print("REACT Worker startup timeout")
        return False

    def send_frame(self, rgb_np, depth_np, robot_pos, robot_quat, robot_yaw):
        if self.worker_proc is None or self.worker_proc.poll() is not None: return
        if rgb_np is None: return
        if getattr(self, '_is_writing', False): return
        self._is_writing = True
        rgb_c = rgb_np.copy()
        depth_c = depth_np.copy() if depth_np is not None else np.zeros((1,1), dtype=np.float32)
        pos_c, quat_c, yaw_v = np.array(robot_pos), np.array(robot_quat), float(robot_yaw)
        def _write():
            try:
                np.savez("/tmp/react_input_tmp", rgb=rgb_c, depth=depth_c,
                         robot_pos=pos_c, robot_quat=quat_c, robot_yaw=np.array(yaw_v))
                os.replace("/tmp/react_input_tmp.npz", REACT_INPUT_PATH)
            except Exception as e: print(f"[REACT] send error: {e}")
            finally: self._is_writing = False
        threading.Thread(target=_write, daemon=True).start()

    def poll_results(self):
        if not os.path.exists(REACT_OUTPUT_PATH): return None
        try: mtime = os.path.getmtime(REACT_OUTPUT_PATH)
        except OSError: return None
        if mtime <= self.last_output_mtime: return None
        self.last_output_mtime = mtime
        try:
            with open(REACT_OUTPUT_PATH, 'r') as f: self.last_output = json.load(f)
        except Exception: return None
        if os.path.exists(REACT_SG_SYNC_PATH):
            try:
                with open(REACT_SG_SYNC_PATH, 'r') as f: self.current_sg = json.load(f)
            except Exception: pass
        return self.last_output

    def get_change_summary(self):
        return self.last_output.get("change_summary", "No changes.")

    def shutdown(self):
        try:
            with open(REACT_SHUTDOWN_PATH, 'w') as f: f.write("stop")
        except: pass
        if self.worker_proc:
            try: self.worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.worker_proc.kill()
            print("REACT Worker stopped.")
