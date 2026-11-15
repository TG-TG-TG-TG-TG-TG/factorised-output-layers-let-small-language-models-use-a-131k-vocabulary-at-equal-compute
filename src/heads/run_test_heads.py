"""Runs test_heads.py unmodified under python -I (isolated mode drops the script dir from sys.path,
and the test's own sys.path line points to the chat sandbox /home/claude/work).
JAX_PLATFORMS=cpu python -I src/heads/run_test_heads.py
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
runpy.run_path(os.path.join(HERE, "test_heads.py"), run_name="__main__")
