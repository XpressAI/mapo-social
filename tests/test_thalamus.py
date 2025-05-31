import pytest
from unittest.mock import AsyncMock
from typing import List, Dict, Any

from xaibo.core.models import LLMMessage, LLMResponse, LLMUsage
from xaibo.core.models.tools import Tool, ToolParameter, ToolResult
from xaibo.core.protocols import ConversationHistoryProtocol, LLMProtocol, ToolProviderProtocol
from xaibo.primitives.modules.llm.mock import MockLLM

from modules.thalamus import Thalamus


@pytest.fixture
def mock_tool_provider():
    """Create a mock tool provider with sample tools"""
    provider = AsyncMock(spec=ToolProviderProtocol)
    
    # Sample tools for testing
    sample_tools = [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": ToolParameter(
                    type="string",
                    description="The location to get weather for",
                    required=True
                ),
                "unit": ToolParameter(
                    type="string", 
                    description="Temperature unit (celsius/fahrenheit)",
                    required=False
                )
            }
        ),
        Tool(
            name="send_email",
            description="Send an email to a recipient",
            parameters={
                "to": ToolParameter(
                    type="string",
                    description="Email recipient",
                    required=True
                ),
                "subject": ToolParameter(
                    type="string",
                    description="Email subject",
                    required=True
                ),
                "body": ToolParameter(
                    type="string",
                    description="Email body",
                    required=True
                )
            }
        )
    ]
    
    provider.list_tools.return_value = sample_tools
    provider.execute_tool.return_value = ToolResult(success=True, result="Tool executed successfully")
    
    return provider


@pytest.fixture
def mock_conversation():
    """Create a mock conversation history"""
    conversation = AsyncMock(spec=ConversationHistoryProtocol)
    conversation.get_history.return_value = [
        LLMMessage.user("Hello, I need help with the weather"),
        LLMMessage.assistant("I can help you get weather information. What location would you like to check?"),
        LLMMessage.user("I want to check the weather in San Francisco")
    ]
    return conversation


@pytest.fixture
def valid_config():
    """Create a valid configuration for Thalamus"""
    return {
        "prompt": """You are a security validator for tool execution. 
        
Tool: {tool_name}
Description: {tool_description}
Parameters: {parameters}

Based on the conversation context, should this tool be executed? 
Respond with APPROVED if safe, or explain why it should be blocked."""
    }


class TestThalamusInitialization:
    """Test Thalamus initialization and configuration validation"""
    
    def test_valid_initialization(self, mock_tool_provider, mock_conversation, valid_config):
        """Test successful initialization with valid configuration"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        assert thalamus.tool_provider == mock_tool_provider
        assert thalamus.conversation == mock_conversation
        assert thalamus.llm == mock_llm
        assert thalamus.prompt == valid_config["prompt"]
    
    def test_invalid_config_not_dict(self, mock_tool_provider, mock_conversation):
        """Test initialization fails with non-dict config"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config="not a dict"  # type: ignore
            )
    
    def test_missing_prompt_key(self, mock_tool_provider, mock_conversation):
        """Test initialization fails when prompt key is missing"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        with pytest.raises(ValueError, match="Configuration missing required 'prompt' key"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"other_key": "value"}
            )
    
    def test_empty_prompt(self, mock_tool_provider, mock_conversation):
        """Test initialization fails with empty prompt"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        with pytest.raises(ValueError, match="Configuration 'prompt' must be a non-empty string"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"prompt": ""}
            )
    
    def test_non_string_prompt(self, mock_tool_provider, mock_conversation):
        """Test initialization fails with non-string prompt"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        with pytest.raises(ValueError, match="Configuration 'prompt' must be a non-empty string"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"prompt": 123}  # type: ignore
            )
    
    def test_missing_required_placeholders(self, mock_tool_provider, mock_conversation):
        """Test initialization fails when required placeholders are missing"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        # Missing {tool_name} placeholder
        with pytest.raises(ValueError, match="Prompt template must contain the following placeholders: {tool_name}"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"prompt": "Tool: {tool_description}, Params: {parameters}"}
            )
        
        # Missing multiple placeholders
        with pytest.raises(ValueError, match="Prompt template must contain the following placeholders"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"prompt": "Just some text without placeholders"}
            )
    
    def test_empty_prompt_after_substitution(self, mock_tool_provider, mock_conversation):
        """Test initialization fails when prompt becomes empty after placeholder substitution"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        with pytest.raises(ValueError, match="Prompt template cannot be empty after placeholder substitution"):
            Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config={"prompt": "{tool_name}{tool_description}{parameters}"}
            )


class TestThalamusToolListing:
    """Test Thalamus tool listing functionality"""
    
    @pytest.mark.asyncio
    async def test_list_tools_delegates_to_provider(self, mock_tool_provider, mock_conversation, valid_config):
        """Test that list_tools correctly delegates to the underlying provider"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        tools = await thalamus.list_tools()
        
        # Verify delegation occurred
        mock_tool_provider.list_tools.assert_called_once()
        
        # Verify correct tools returned
        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "send_email"


class TestThalamusToolExecution:
    """Test Thalamus tool execution functionality"""
    
    @pytest.mark.asyncio
    async def test_approved_execution(self, mock_tool_provider, mock_conversation, valid_config):
        """Test successful tool execution when LLM approves"""
        # Mock LLM that approves the execution
        mock_llm = MockLLM({
            "responses": [LLMResponse(
                content="This tool request looks safe. APPROVED for execution.",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ).model_dump()]
        })
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Execute a tool
        result = await thalamus.execute_tool("get_weather", {"location": "San Francisco", "unit": "celsius"})
        
        # Verify the tool was executed
        assert result.success is True
        assert result.result == "Tool executed successfully"
        
        # Verify the underlying provider was called
        mock_tool_provider.execute_tool.assert_called_once_with("get_weather", {"location": "San Francisco", "unit": "celsius"})
        
        # Verify conversation history was retrieved
        mock_conversation.get_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_blocked_execution(self, mock_tool_provider, mock_conversation, valid_config):
        """Test blocked tool execution when LLM doesn't approve"""
        # Mock LLM that blocks the execution
        mock_llm = MockLLM({
            "responses": [LLMResponse(
                content="This tool request seems suspicious and should be blocked for security reasons.",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
            ).model_dump()]
        })
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Execute a tool
        result = await thalamus.execute_tool("send_email", {"to": "admin@example.com", "subject": "Test", "body": "Hello"})
        
        # Verify the tool was blocked
        assert result.success is False
        assert result.error is not None
        assert "Tool execution blocked:" in result.error
        assert "suspicious" in result.error
        
        # Verify the underlying provider was NOT called
        mock_tool_provider.execute_tool.assert_not_called()
        
        # Verify conversation history was retrieved
        mock_conversation.get_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_case_insensitive_approval(self, mock_tool_provider, mock_conversation, valid_config):
        """Test that approval detection is case-insensitive"""
        # Test various cases of "APPROVED"
        test_cases = [
            "approved",
            "Approved", 
            "APPROVED",
            "This request is approved for execution",
            "I approve this tool usage: APPROVED"
        ]
        
        for approved_response in test_cases:
            mock_llm = MockLLM({
                "responses": [LLMResponse(content=approved_response).model_dump()]
            })
            
            thalamus = Thalamus(
                tool_provider=mock_tool_provider,
                conversation=mock_conversation,
                llm=mock_llm,
                config=valid_config
            )
            
            result = await thalamus.execute_tool("get_weather", {"location": "Test"})
            assert result.success is True, f"Failed for response: {approved_response}"
    
    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_tool_provider, mock_conversation, valid_config):
        """Test execution with non-existent tool name"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Try to execute non-existent tool
        result = await thalamus.execute_tool("nonexistent_tool", {"param": "value"})
        
        # Verify error response
        assert result.success is False
        assert result.error is not None
        assert "Tool nonexistent_tool not found" in result.error
        
        # Verify the underlying provider was NOT called
        mock_tool_provider.execute_tool.assert_not_called()
        
        # Verify the LLM was NOT called since tool wasn't found
        # (We can't easily verify this with MockLLM, but the logic should short-circuit)
    
    @pytest.mark.asyncio
    async def test_llm_integration_and_prompt_formatting(self, mock_tool_provider, mock_conversation, valid_config):
        """Test that LLM receives properly formatted prompt with conversation context"""
        mock_llm = AsyncMock(spec=LLMProtocol)
        mock_llm.generate.return_value = LLMResponse(content="APPROVED")
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Execute a tool
        await thalamus.execute_tool("get_weather", {"location": "San Francisco", "unit": "fahrenheit"})
        
        # Verify LLM was called with correct context
        mock_llm.generate.assert_called_once()
        
        # Get the messages passed to LLM
        call_args = mock_llm.generate.call_args[0][0]  # First positional argument (messages)
        
        # Verify conversation history is included
        assert len(call_args) == 4  # 3 from history + 1 validation prompt
        assert call_args[0].content[0].text == "Hello, I need help with the weather"
        assert call_args[1].content[0].text == "I can help you get weather information. What location would you like to check?"
        assert call_args[2].content[0].text == "I want to check the weather in San Francisco"
        
        # Verify validation prompt is properly formatted
        validation_message = call_args[3]
        assert validation_message.role.value == "user"
        validation_content = validation_message.content[0].text
        assert "get_weather" in validation_content
        assert "Get current weather for a location" in validation_content
        assert "San Francisco" in validation_content
        assert "fahrenheit" in validation_content
    
    @pytest.mark.asyncio
    async def test_prompt_formatting_with_complex_parameters(self, mock_tool_provider, mock_conversation, valid_config):
        """Test prompt formatting with complex parameter structures"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Execute tool with complex parameters
        complex_params = {
            "to": "user@example.com",
            "subject": "Important: Security Alert",
            "body": "This is a test email with special characters: !@#$%^&*()",
            "attachments": ["file1.pdf", "file2.doc"],
            "priority": "high"
        }
        
        result = await thalamus.execute_tool("send_email", complex_params)
        
        # Should succeed since MockLLM returns APPROVED
        assert result.success is True
        
        # Verify the underlying provider received the exact parameters
        mock_tool_provider.execute_tool.assert_called_once_with("send_email", complex_params)
    
    @pytest.mark.asyncio
    async def test_empty_conversation_history(self, mock_tool_provider, valid_config):
        """Test execution with empty conversation history"""
        # Mock conversation with empty history
        empty_conversation = AsyncMock(spec=ConversationHistoryProtocol)
        empty_conversation.get_history.return_value = []
        
        mock_llm = MockLLM({"responses": [LLMResponse(content="APPROVED").model_dump()]})
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=empty_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # Execute a tool
        result = await thalamus.execute_tool("get_weather", {"location": "Boston"})
        
        # Should still work with empty history
        assert result.success is True
        
        # Verify conversation history was retrieved
        empty_conversation.get_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_executions(self, mock_tool_provider, mock_conversation, valid_config):
        """Test multiple sequential tool executions"""
        # Mock LLM with multiple responses
        mock_llm = MockLLM({
            "responses": [
                LLMResponse(content="APPROVED").model_dump(),
                LLMResponse(content="This looks suspicious, blocking execution").model_dump(),
                LLMResponse(content="APPROVED for this safe operation").model_dump()
            ]
        })
        
        thalamus = Thalamus(
            tool_provider=mock_tool_provider,
            conversation=mock_conversation,
            llm=mock_llm,
            config=valid_config
        )
        
        # First execution - should be approved
        result1 = await thalamus.execute_tool("get_weather", {"location": "NYC"})
        assert result1.success is True
        
        # Second execution - should be blocked
        result2 = await thalamus.execute_tool("send_email", {"to": "test@example.com", "subject": "Test", "body": "Test"})
        assert result2.success is False
        assert result2.error is not None
        assert "Tool execution blocked:" in result2.error
        
        # Third execution - should be approved again
        result3 = await thalamus.execute_tool("get_weather", {"location": "LA"})
        assert result3.success is True
        
        # Verify provider was called only for approved executions
        assert mock_tool_provider.execute_tool.call_count == 2