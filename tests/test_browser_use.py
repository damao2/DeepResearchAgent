import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.tools.auto_browser import AutoBrowserUseTool
from src.models import model_manager

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=False)
    
    tool = AutoBrowserUseTool()

    task = "Open the video: https://www.youtube.com/watch?v=2s7OebVzJjI. Then jump to the 0:30."
    res = asyncio.run(tool.forward(task=task))
    print(res)