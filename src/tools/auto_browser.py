import os
import subprocess
import atexit
import signal
import time
from dotenv import load_dotenv
load_dotenv(verbose=True)

import logging
# 设置 browser_use 模块的日志级别为 DEBUG
logging.getLogger('browser_use').setLevel(logging.DEBUG)
logging.getLogger('browser_use.agent.service').setLevel(logging.DEBUG)
logging.getLogger('browser_use.browser.session').setLevel(logging.DEBUG)
logging.getLogger('browser_use.controller.service').setLevel(logging.DEBUG)
logging.getLogger('browser_use.utils').setLevel(logging.DEBUG)

import asyncio
from browser_use import Agent
from browser_use import BrowserConfig, Browser
from langchain_openai import ChatOpenAI

from src.proxy.local_proxy import PROXY_URL, proxy_env
from src.tools import AsyncTool, ToolResult
from src.tools.browser import Controller, CDP
from src.utils import assemble_project_path
from src.config import config
from src.registry import register_tool

@register_tool("auto_browser_use")
class AutoBrowserUseTool(AsyncTool):
    name = "auto_browser_use"
    description = "A powerful browser automation tool that allows interaction with web pages through various actions. Automatically browse the web and extract information based on a given task."
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task to perform"
            },
        },
        "required": ["task"],
    }
    output_type = "any"

    def __init__(self):
        super(AutoBrowserUseTool, self).__init__()

        self.http_server_path = assemble_project_path("src/tools/browser/http_server")
        self.http_save_path = assemble_project_path("src/tools/browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self._init_pdf_server()
        self.browser_agent = self._init_browser_agent()

    def _init_pdf_server(self):

        server_proc = subprocess.Popen(
            ["python3", "-m", "http.server", "8080"],
            cwd=self.http_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=None
        )

        @atexit.register
        def shutdown_server():
            print("Shutting down http.server...")
            try:
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=5)
            except Exception as e:
                print("Force killing server...")
                server_proc.kill()

    def _init_browser_agent(self):
        """
        Initialize the browser agent with the given configuration.
        """

        if config.use_local_proxy:
            os.environ["HTTP_PROXY"] = PROXY_URL
            os.environ["HTTPS_PROXY"] = PROXY_URL

        controller = Controller(http_save_path=self.http_save_path)

        # Get model_id and api_base from config
        model_id = config.browser_tool.model_id
        api_base = config.browser_tool.api_base

        # Determine API key based on local proxy usage
        if config.use_local_proxy:
            api_key = os.getenv("SKYWORK_API_KEY")
            if not api_base: # If api_base is not set in config, use default SKYWORK_API_BASE
                api_base = os.getenv("SKYWORK_API_BASE")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_base: # If api_base is not set in config, use default OPENAI_API_BASE
                api_base = os.getenv("OPENAI_API_BASE")

        model = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=api_base,
        )

        browser_agent = Agent(
            task="",
            llm=model,
            enable_memory=False,
            controller=controller,
            page_extraction_llm=model,
        )

        return browser_agent

    async def _browser_task(self, task):
        if config.use_local_proxy:
            with proxy_env(PROXY_URL):
                self.browser_agent.add_new_task(task)
                history = await self.browser_agent.run(max_steps=50)
                contents = history.extracted_content()
        else:
            self.browser_agent.add_new_task(task)
            history = await self.browser_agent.run(max_steps=50)
            contents = history.extracted_content()
        return "\n".join(contents)

    async def forward(self, task: str) -> ToolResult:
        """
        Automatically browse the web and extract information based on a given task.

        Args:
            task: The task to perform
        Returns:
            ToolResult with the task result
        """
        result = await self._browser_task(task)
        return ToolResult(output=result, error=None)
