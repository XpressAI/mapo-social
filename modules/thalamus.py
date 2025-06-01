import json
from typing import List, Dict, Any

from xaibo.core.models import LLMMessage
from xaibo.core.protocols import ConversationHistoryProtocol, LLMProtocol, ToolProviderProtocol
from xaibo.core.protocols.tools import ToolResult, Tool

class Thalamus(ToolProviderProtocol):
    """ ToolProviderProtocol wrapper that acts as a "Thalamus"
    
    Thalamus wraps a ToolProviderProtocol implementation and questions if the chosen tool in `execute_tool` is
    appropriate to use. If it comes to the conclusion that it is not, it will share its thoughts with the caller and
    block the execution.
    
    It takes the current conversation and the tool description as a context when making that decision.
    """
    
    def __init__(self, 
                 tool_provider: ToolProviderProtocol, 
                 conversation: ConversationHistoryProtocol, 
                 llm: LLMProtocol,
                 config: Dict[str, str]
                 ):
        """Initialize the Thalamus with the given tool provider, conversation and LLM
    
        Args:
            tool_provider (ToolProviderProtocol): The underlying tool provider that provides and executes tools
            conversation (ConversationHistoryProtocol): The conversation history to use as context
            llm (LLMProtocol): The language model to use for decision making
            config (dict[str, str]): Configuration dictionary for the Thalamus. Must contain:
                prompt (str): The prompt template used for decision making. Must ask to approve tool execution,
                    such that the word "APPROVED" is present in the response. Required placeholders:
                    {tool_name}: Name of the tool being validated
                    {tool_description}: Description of the tool being validated
                    {parameters}: Parameters passed to the tool
        """
        self.tool_provider = tool_provider
        self.conversation = conversation
        self.llm = llm

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        if 'prompt' not in config:
            raise ValueError("Configuration missing required 'prompt' key")

        if not config['prompt'] or not isinstance(config['prompt'], str):
            raise ValueError("Configuration 'prompt' must be a non-empty string")

        required_placeholders = ['{tool_name}', '{tool_description}', '{parameters}']
        missing_placeholders = [p for p in required_placeholders if p not in config['prompt']]
        if missing_placeholders:
            raise ValueError(
                f"Prompt template must contain the following placeholders: {', '.join(missing_placeholders)}")

        if config['prompt'].format(tool_name='', tool_description='', parameters='').strip() == '':
            raise ValueError("Prompt template cannot be empty after placeholder substitution")

        self.prompt = config['prompt']
    
    async def list_tools(self) -> List[Tool]:
        """List all available tools provided by this provider"""
        return await self.tool_provider.list_tools()
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with the given parameters"""
        tools = await self.list_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if not tool:
            return ToolResult(success=False, error=f"Tool {tool_name} not found")

        conversation_context = [x for x in await self.conversation.get_history()]
        
        validation_prompt = self.prompt.format(
            tool_name=tool_name,
            tool_description=tool.description,
            parameters=parameters
        )
        conversation_context.append(
            LLMMessage.user(validation_prompt)
        )

        validation_response = await self.llm.generate(conversation_context)
        if "APPROVED" not in validation_response.content.upper():
            return ToolResult(
                success=False,
                error=f"Tool execution blocked: {validation_response.content}"
            )

        return await self.tool_provider.execute_tool(tool_name, parameters)
