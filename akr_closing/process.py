"""Synchronous child execution with signals forwarded only to the owned process group."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import time


def run_checked(argv,*,cwd:Path,log_path:Path,stop_requested=None):
    if stop_requested is None:
        from . import pipeline
        stop_requested=lambda:pipeline.STOP
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a') as log:
        child=subprocess.Popen(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                if stop_requested():
                    os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                    from .pipeline import SafeStop
                    raise SafeStop('owned subprocess stopped; atomically completed rows are retained')
                time.sleep(.2)
            if child.returncode:raise RuntimeError(f'child exited {child.returncode}; see {log_path}; checkpoints retained')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
    return child.returncode
