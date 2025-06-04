import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.models import model_manager
from src.tools.deep_analyzer import DeepAnalyzerTool

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=False)
    
    deep_analyzer = DeepAnalyzerTool()

    task = """
    Please analyze the attached file or uri and provide a detailed caption.
    """

    response = asyncio.run(deep_analyzer.forward(task=task, source=os.path.join(root, "data/GAIA/2023/validation/6359a0b1-8f7b-499b-9336-840f9ab90688.png")))
    
    print(response)