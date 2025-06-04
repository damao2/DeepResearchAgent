from src.tools.tools import AsyncTool, ToolResult
from typing import Any


class FinalAnswerTool(AsyncTool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    parameters = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "any",
                "description": "The final answer to the problem.",
            },
        },
        "required": ["answer"],
    }
    output_type = "any"

    async def forward(self, answer: Any) -> ToolResult:
        result = ToolResult(
            output=answer,
            error=None,
        )
        return result