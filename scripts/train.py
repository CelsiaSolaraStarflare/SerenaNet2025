"""
Training script for SerenaNet.
"""
from __future__ import annotations
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.trainer import main as trainer_main


if __name__ == '__main__':
    trainer_main()
