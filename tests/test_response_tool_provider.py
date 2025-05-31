import pytest
from unittest.mock import AsyncMock
from typing import Dict, Any

from xaibo.core.protocols import ResponseProtocol
from xaibo.core.models.tools import Tool, ToolParameter, ToolResult

from modules.response_tool_provider import ResponseToolProvider


@pytest.fixture
def mock_response_protocol():
    """Create a mock ResponseProtocol for testing"""
    return AsyncMock(spec=ResponseProtocol)


@pytest.fixture
def response_tool_provider(mock_response_protocol):
    """Create a ResponseToolProvider instance with mocked dependency"""
    return ResponseToolProvider(response=mock_response_protocol)


class TestResponseToolProviderInitialization:
    """Test ResponseToolProvider initialization"""
    
    def test_initialization_with_response_protocol(self, mock_response_protocol):
        """Test that ResponseToolProvider can be initialized with a ResponseProtocol"""
        provider = ResponseToolProvider(response=mock_response_protocol)
        
        assert provider.response == mock_response_protocol
    
    def test_response_attribute_properly_set(self, mock_response_protocol):
        """Test that the response attribute is properly set"""
        provider = ResponseToolProvider(response=mock_response_protocol)
        
        assert hasattr(provider, 'response')
        assert provider.response is mock_response_protocol


class TestResponseToolProviderToolListing:
    """Test ResponseToolProvider tool listing functionality"""
    
    @pytest.mark.asyncio
    async def test_list_tools_returns_one_tool(self, response_tool_provider):
        """Test list_tools() returns exactly one tool named 'respond'"""
        tools = await response_tool_provider.list_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "respond"
    
    @pytest.mark.asyncio
    async def test_tool_has_correct_description(self, response_tool_provider):
        """Test the tool has correct description"""
        tools = await response_tool_provider.list_tools()
        tool = tools[0]
        
        assert tool.description == "Triggers a response to the user with the provided message"
    
    @pytest.mark.asyncio
    async def test_tool_has_correct_parameters(self, response_tool_provider):
        """Test the tool has correct parameters"""
        tools = await response_tool_provider.list_tools()
        tool = tools[0]
        
        assert "message" in tool.parameters
        message_param = tool.parameters["message"]
        
        assert isinstance(message_param, ToolParameter)
        assert message_param.type == "string"
        assert message_param.description == "The message content to send as a response to the user"
        assert message_param.required is True
    
    @pytest.mark.asyncio
    async def test_message_parameter_is_required_and_string_type(self, response_tool_provider):
        """Test the message parameter is required and of type string"""
        tools = await response_tool_provider.list_tools()
        tool = tools[0]
        message_param = tool.parameters["message"]
        
        assert message_param.required is True
        assert message_param.type == "string"


class TestResponseToolProviderToolExecution:
    """Test ResponseToolProvider tool execution functionality"""
    
    @pytest.mark.asyncio
    async def test_successful_execution_of_respond_tool(self, response_tool_provider, mock_response_protocol):
        """Test successful execution of 'respond' tool with valid message"""
        test_message = "Hello, this is a test response!"
        
        result = await response_tool_provider.execute_tool("respond", {"message": test_message})
        
        assert result.success is True
        assert result.result == "Response sent successfully"
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_execution_calls_response_respond_text_with_correct_message(self, response_tool_provider, mock_response_protocol):
        """Test execution calls response.respond_text() with correct message"""
        test_message = "Test message for verification"
        
        await response_tool_provider.execute_tool("respond", {"message": test_message})
        
        mock_response_protocol.respond_text.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_unknown_tool_name_returns_error(self, response_tool_provider, mock_response_protocol):
        """Test unknown tool name returns error"""
        result = await response_tool_provider.execute_tool("unknown_tool", {"message": "test"})
        
        assert result.success is False
        assert result.error == "Unknown tool: unknown_tool"
        assert result.result is None
        
        # Verify response.respond_text was not called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_missing_message_parameter_returns_error(self, response_tool_provider, mock_response_protocol):
        """Test missing message parameter returns error"""
        result = await response_tool_provider.execute_tool("respond", {})
        
        assert result.success is False
        assert result.error == "Missing required parameter: message"
        assert result.result is None
        
        # Verify response.respond_text was not called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_empty_message_parameter_returns_error(self, response_tool_provider, mock_response_protocol):
        """Test empty message parameter returns error"""
        result = await response_tool_provider.execute_tool("respond", {"message": ""})
        
        assert result.success is False
        assert result.error == "Missing required parameter: message"
        assert result.result is None
        
        # Verify response.respond_text was not called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_none_message_parameter_returns_error(self, response_tool_provider, mock_response_protocol):
        """Test None message parameter returns error"""
        result = await response_tool_provider.execute_tool("respond", {"message": None})
        
        assert result.success is False
        assert result.error == "Missing required parameter: message"
        assert result.result is None
        
        # Verify response.respond_text was not called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_exception_handling_when_respond_text_raises_exception(self, response_tool_provider, mock_response_protocol):
        """Test exception handling when response.respond_text() raises an exception"""
        test_message = "Test message that will cause an exception"
        test_exception = Exception("Network error occurred")
        
        # Configure mock to raise an exception
        mock_response_protocol.respond_text.side_effect = test_exception
        
        result = await response_tool_provider.execute_tool("respond", {"message": test_message})
        
        assert result.success is False
        assert result.error == "Failed to send response: Network error occurred"
        assert result.result is None
        
        # Verify response.respond_text was called with correct message
        mock_response_protocol.respond_text.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_execution_with_complex_message_content(self, response_tool_provider, mock_response_protocol):
        """Test execution with complex message content including special characters"""
        complex_message = """This is a complex message with:
        - Special characters: !@#$%^&*()
        - Unicode: 🚀 ✨ 🎉
        - Multiple lines
        - JSON-like content: {"key": "value", "number": 42}
        """
        
        result = await response_tool_provider.execute_tool("respond", {"message": complex_message})
        
        assert result.success is True
        assert result.result == "Response sent successfully"
        
        # Verify the exact message was passed to respond_text
        mock_response_protocol.respond_text.assert_called_once_with(complex_message)
    
    @pytest.mark.asyncio
    async def test_execution_with_additional_ignored_parameters(self, response_tool_provider, mock_response_protocol):
        """Test execution ignores additional parameters beyond 'message'"""
        test_message = "Main message content"
        parameters = {
            "message": test_message,
            "extra_param": "should be ignored",
            "another_param": 123
        }
        
        result = await response_tool_provider.execute_tool("respond", parameters)
        
        assert result.success is True
        assert result.result == "Response sent successfully"
        
        # Verify only the message was passed to respond_text
        mock_response_protocol.respond_text.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_executions(self, response_tool_provider, mock_response_protocol):
        """Test multiple sequential tool executions"""
        messages = ["First message", "Second message", "Third message"]
        
        for message in messages:
            result = await response_tool_provider.execute_tool("respond", {"message": message})
            assert result.success is True
            assert result.result == "Response sent successfully"
        
        # Verify all calls were made with correct messages
        assert mock_response_protocol.respond_text.call_count == 3
        call_args_list = mock_response_protocol.respond_text.call_args_list
        
        for i, message in enumerate(messages):
            assert call_args_list[i][0][0] == message
    
    @pytest.mark.asyncio
    async def test_mock_verification_respond_text_not_called_on_error(self, response_tool_provider, mock_response_protocol):
        """Test that response.respond_text() is not called when there are errors"""
        # Test with unknown tool
        await response_tool_provider.execute_tool("unknown", {"message": "test"})
        
        # Test with missing message
        await response_tool_provider.execute_tool("respond", {})
        
        # Test with empty message
        await response_tool_provider.execute_tool("respond", {"message": ""})
        
        # Verify respond_text was never called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_tool_result_structure_on_success(self, response_tool_provider, mock_response_protocol):
        """Test that ToolResult has correct structure on success"""
        result = await response_tool_provider.execute_tool("respond", {"message": "test"})
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.result == "Response sent successfully"
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_tool_result_structure_on_failure(self, response_tool_provider, mock_response_protocol):
        """Test that ToolResult has correct structure on failure"""
        result = await response_tool_provider.execute_tool("unknown", {"message": "test"})
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.result is None
        assert result.error is not None
        assert isinstance(result.error, str)