# DeepResearchAgent

## Introduction

DeepResearchAgent is a hierarchical multi-agent system designed not only for deep research tasks but also for general-purpose task solving. The framework leverages a top-level planning agent to coordinate multiple specialized lower-level agents, enabling automated task decomposition and efficient execution across diverse and complex domains.

## Architecture

The system adopts a two-layer structure:

### 1. Top-Level Planning Agent
- Responsible for understanding, decomposing, and planning the overall workflow for a given task.
- Breaks down tasks into manageable sub-tasks and assigns them to appropriate lower-level agents.
- Dynamically coordinates the collaboration among agents to ensure smooth task completion.

### 2. Specialized Lower-Level Agents
- **Deep Analyzer**
  - Performs in-depth analysis of input information, extracting key insights and potential requirements.
  - Supports analysis of various data types, including text and structured data.
- **Deep Researcher**
  - Conducts thorough research on specified topics or questions, retrieving and synthesizing high-quality information.
  - Capable of generating research reports or knowledge summaries automatically.
- **Browser Use**
  - Automates browser operations, supporting web search, information extraction, and data collection tasks.
  - Assists the Deep Researcher in acquiring up-to-date information from the internet.

## Features
- Hierarchical agent collaboration for complex and dynamic task scenarios
- Extensible agent system, allowing easy integration of additional specialized agents
- Automated information analysis, research, and web interaction capabilities
  

## Updates
- 2025.05.27
  - Updated the available remote API calls to support OpenAI, Anthropic, and Google LLMs.

## TODO List
- [x] Asynchronous feature completed
- [ ] Image Generation Agent to be developed
- [ ] MCP in progress
- [ ] AI4Research Agent to be developed
- [ ] Novel Writing Agent to be developed

## Installation

### Prepare Environment
```
# poetry install environment
conda create -n dra python=3.11
conda activate dra
make install

# (Optional) You can also use requirements.txt to setup the environment
conda create -n dra python=3.11
conda activate dra
make install-requirements

# If you encounter any issues with Playwright during installation, you can install it manually:
pip install playwright
playwright install chromium --with-deps --no-shell
```

### Put `.env` in the root

`.env` should be like
```
PYTHONWARNINGS=ignore # ignore warnings
ANONYMIZED_TELEMETRY=false # disable telemetry
HUGGINEFACE_API_KEY=abcabcabc # your huggingface api key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=abcabcabc # your openai api key
ANTHROPIC_API_BASE=https://api.anthropic.com
ANTHROPIC_API_KEY=abcabcabc # your anthropic api key
GOOGLE_API_BASE=https://generativelanguage.googleapis.com
GOOGLE_API_KEY=abcabcabc # your google api key
```

## Usage

### Deep Researcher for "AI Agent" as an example
```
python examples/run_example.py
```

### GAIA as an example

```
# Download GAIA
mkdir data | cd data
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA

# Run the script in the examples
python examples/run_gaia.py
```

## Experiments
We evaluated our agent on the GAIA validation set and achieved state-of-the-art performance on May 10th.
<p align="center">
  <img src="./docs/gaia_benchmark.png" alt="GAIA Example Result" width="700"/>
</p>

## Acknowledgement
DeepResearchAgent is primarily inspired by the architecture of smolagents. The following improvements have been made:
- The codebase of smolagents has been modularized for better structure and organization.
- The original synchronous framework has been refactored into an asynchronous one.
- The multi-agent setup process has been optimized to make it more user-friendly and efficient.

We would like to express our gratitude to the following open source projects, which have greatly contributed to the development of this work:
- [smolagents](https://github.com/huggingface/smolagents) - A lightweight agent framework.
- [OpenManus](https://github.com/mannaandpoem/OpenManus) - An asynchronous agent framework.
- [browser-use](https://github.com/browser-use/browser-use) - An AI-powered browser automation tool.
- [crawl4ai](https://github.com/unclecode/crawl4ai) - A web crawling library for AI applications.
- [markitdown](https://github.com/microsoft/markitdown) - A tool for converting files to Markdown format.

We sincerely appreciate the efforts of all contributors and maintainers of these projects for their commitment to advancing AI technologies and making them available to the wider community.

## Contribution

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Cite
```
@misc{DeepResearchAgent,
  title =        {`DeepResearchAgent`: A Hierarchical Multi-Agent Framework for General-purpose Task Solving.},
  author =       {Wentao Zhang, Ce Cui, Yang Liu, Bo An},
  howpublished = {\url{https://github.com/SkyworkAI/DeepResearchAgent}},
  year =         {2025}
}
```