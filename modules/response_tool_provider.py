from typing import Dict, Any, List

from xaibo.core.protocols import ConversationHistoryProtocol, LLMProtocol, ToolProviderProtocol, ResponseProtocol
from xaibo.core.protocols.tools import ToolResult, Tool

class ResponseToolProvider(ToolProviderProtocol):
    def __init__(self, response: ResponseProtocol):
        self.response = response

    async def list_tools(self) -> List[Tool]:
        """List all available tools provided by this provider"""
        from xaibo.core.models.tools import ToolParameter
        
        return [
            Tool(
                name="respond",
                description="Triggers a response to the user with the provided message",
                parameters={
                    "message": ToolParameter(
                        type="string",
                        description="The message content to send as a response to the user",
                        required=True
                    )
                }
            )
        ]

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with the given parameters"""
        if tool_name != "respond":
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
        
        # Extract the message parameter
        message = parameters.get("message")
        if not message:
            return ToolResult(
                success=False,
                error="Missing required parameter: message"
            )
        
        try:
            # Use the response protocol to send the message
            await self.response.respond_text(message)
            return ToolResult(
                success=True,
                result="Response sent successfully"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to send response: {str(e)}"
            )
