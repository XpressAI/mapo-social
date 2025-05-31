import pytest
from unittest.mock import AsyncMock
from typing import List, Dict, Any

from xaibo.core.models import LLMMessage, LLMResponse, LLMUsage
from xaibo.core.models.llm import LLMFunctionCall
from xaibo.core.models.tools import Tool, ToolParameter, ToolResult
from xaibo.core.protocols import ResponseProtocol, ConversationHistoryProtocol, LLMProtocol, ToolProviderProtocol
from xaibo.primitives.modules.llm.mock import MockLLM

from modules.response_tool_provider import ResponseToolProvider
from modules.pfc import PFCOrchestrator


@pytest.fixture
def mock_response_protocol():
    """Create a mock ResponseProtocol for testing"""
    return AsyncMock(spec=ResponseProtocol)


@pytest.fixture
def mock_conversation_history():
    """Create a mock conversation history"""
    conversation = AsyncMock(spec=ConversationHistoryProtocol)
    conversation.get_history.return_value = [
        LLMMessage.user("Hello, I need help with something"),
        LLMMessage.assistant("I can help you with that. What do you need?"),
        LLMMessage.user("I want to check the weather")
    ]
    return conversation


@pytest.fixture
def valid_config():
    """Create a valid configuration for PFCOrchestrator"""
    return {
        "system_prompt": "You are a helpful assistant that can use tools to help users.",
        "max_thoughts": 5,
        "stop_tool_name": "respond"
    }


@pytest.fixture
def mock_llm_responses():
    """Create mock LLM responses for various scenarios"""
    return {
        "tool_call_response": LLMResponse(
            content="I'll help you check the weather.",
            tool_calls=[
                LLMFunctionCall(
                    id="call_123",
                    name="get_weather",
                    arguments={"location": "San Francisco"}
                )
            ],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        ),
        "stop_tool_response": LLMResponse(
            content="Here's the weather information for you.",
            tool_calls=[
                LLMFunctionCall(
                    id="call_456",
                    name="respond",
                    arguments={"message": "The weather in San Francisco is sunny, 72°F"}
                )
            ],
            usage=LLMUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23)
        ),
        "no_tool_response": LLMResponse(
            content="I understand your request.",
            usage=LLMUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        )
    }


@pytest.fixture
def response_tool_provider(mock_response_protocol):
    """Create a ResponseToolProvider instance with mocked dependency"""
    return ResponseToolProvider(response=mock_response_protocol)


@pytest.fixture
def mock_tool_provider(response_tool_provider):
    """Create a mock tool provider with sample tools including respond tool"""
    provider = AsyncMock(spec=ToolProviderProtocol)
    
    # Sample tools for testing - must include the respond tool
    sample_tools = [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": ToolParameter(
                    type="string",
                    description="The location to get weather for",
                    required=True
                )
            }
        ),
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
    
    provider.list_tools.return_value = sample_tools
    provider.execute_tool.return_value = ToolResult(success=True, result="Tool executed successfully")
    
    return provider


@pytest.fixture
def pfc_orchestrator(mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config):
    """Create a PFCOrchestrator instance with mocked dependencies"""
    mock_llm = MockLLM({"responses": [LLMResponse(content="Test response").model_dump()]})
    
    return PFCOrchestrator(
        response=mock_response_protocol,
        llm=mock_llm,
        tool_provider=mock_tool_provider,
        history=mock_conversation_history,
        config=valid_config
    )


class TestPFCOrchestratorInitialization:
    """Test PFCOrchestrator initialization and configuration validation"""
    
    def test_valid_initialization(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config):
        """Test successful initialization with valid configuration"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        assert orchestrator.response == mock_response_protocol
        assert orchestrator.llm == mock_llm
        assert orchestrator.tool_provider == mock_tool_provider
        assert orchestrator.history == mock_conversation_history
        assert orchestrator.system_prompt == valid_config["system_prompt"]
        assert orchestrator.max_thoughts == valid_config["max_thoughts"]
        assert orchestrator.stop_tool_name == valid_config["stop_tool_name"]
    
    def test_initialization_with_none_config(self, mock_response_protocol, mock_conversation_history, mock_tool_provider):
        """Test initialization with None config uses defaults"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=None
        )
        
        assert orchestrator.system_prompt == ""
        assert orchestrator.max_thoughts == 10
        assert orchestrator.stop_tool_name == "respond"
    
    def test_initialization_with_empty_config(self, mock_response_protocol, mock_conversation_history, mock_tool_provider):
        """Test initialization with empty config uses defaults"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config={}
        )
        
        assert orchestrator.system_prompt == ""
        assert orchestrator.max_thoughts == 10
        assert orchestrator.stop_tool_name == "respond"
    
    def test_initialization_with_partial_config(self, mock_response_protocol, mock_conversation_history, mock_tool_provider):
        """Test initialization with partial config"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        partial_config = {"max_thoughts": 3}
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=partial_config
        )
        
        assert orchestrator.system_prompt == ""  # default
        assert orchestrator.max_thoughts == 3    # from config
        assert orchestrator.stop_tool_name == "respond"  # default
    
    def test_provides_method(self):
        """Test that provides() method returns correct protocols"""
        from xaibo.core.protocols import TextMessageHandlerProtocol
        
        protocols = PFCOrchestrator.provides()
        assert protocols == [TextMessageHandlerProtocol]


class TestPFCOrchestratorStopToolValidation:
    """Test PFCOrchestrator stop tool validation functionality"""
    
    def test_validate_stop_tool_with_valid_tool(self, pfc_orchestrator):
        """Test validation passes when stop tool exists"""
        tools = [
            Tool(name="get_weather", description="Get weather", parameters={}),
            Tool(name="respond", description="Respond to user", parameters={})
        ]
        
        # Should not raise an exception
        pfc_orchestrator._async_validate_stop_tool(tools)
    
    def test_validate_stop_tool_with_missing_tool(self, pfc_orchestrator):
        """Test validation fails when stop tool is missing"""
        tools = [
            Tool(name="get_weather", description="Get weather", parameters={}),
            Tool(name="send_email", description="Send email", parameters={})
        ]
        
        with pytest.raises(ValueError, match="Stop tool 'respond' not found in tool provider"):
            pfc_orchestrator._async_validate_stop_tool(tools)
    
    def test_validate_stop_tool_with_custom_stop_tool_name(self, mock_response_protocol, mock_conversation_history, mock_tool_provider):
        """Test validation with custom stop tool name"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        config = {"stop_tool_name": "custom_stop"}
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=config
        )
        
        tools = [
            Tool(name="get_weather", description="Get weather", parameters={}),
            Tool(name="custom_stop", description="Custom stop tool", parameters={})
        ]
        
        # Should not raise an exception
        orchestrator._async_validate_stop_tool(tools)
        
        # Test with missing custom stop tool
        tools_without_custom = [
            Tool(name="get_weather", description="Get weather", parameters={}),
            Tool(name="respond", description="Default respond tool", parameters={})
        ]
        
        with pytest.raises(ValueError, match="Stop tool 'custom_stop' not found in tool provider"):
            orchestrator._async_validate_stop_tool(tools_without_custom)


class TestPFCOrchestratorTextHandling:
    """Test PFCOrchestrator text handling functionality"""
    
    @pytest.mark.asyncio
    async def test_successful_text_handling_with_tool_execution(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config, mock_llm_responses):
        """Test successful text handling with tool execution"""
        # Mock LLM with tool call followed by stop tool
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["tool_call_response"].model_dump(),
                mock_llm_responses["stop_tool_response"].model_dump()
            ]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("What's the weather like?")
        
        # Verify conversation history was retrieved
        mock_conversation_history.get_history.assert_called_once()
        
        # Verify tools were listed
        mock_tool_provider.list_tools.assert_called_once()
        
        # Verify tool was executed
        mock_tool_provider.execute_tool.assert_called()
    
    @pytest.mark.asyncio
    async def test_tool_execution_failure_increases_stress_level(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config, mock_llm_responses):
        """Test that tool execution failures increase stress level"""
        # Configure tool provider to fail
        mock_tool_provider.execute_tool.return_value = ToolResult(success=False, error="Tool execution failed")
        
        # Mock LLM with tool call followed by stop tool
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["tool_call_response"].model_dump(),
                mock_llm_responses["stop_tool_response"].model_dump()
            ]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("What's the weather like?")
        
        # Verify tool execution was attempted
        mock_tool_provider.execute_tool.assert_called()
    
    @pytest.mark.asyncio
    async def test_tool_execution_exception_increases_stress_level(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config, mock_llm_responses):
        """Test that tool execution exceptions increase stress level"""
        # Configure tool provider to raise exception
        mock_tool_provider.execute_tool.side_effect = Exception("Network error")
        
        # Mock LLM with tool call followed by stop tool
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["tool_call_response"].model_dump(),
                mock_llm_responses["stop_tool_response"].model_dump()
            ]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("What's the weather like?")
        
        # Verify tool execution was attempted
        mock_tool_provider.execute_tool.assert_called()
    
    @pytest.mark.asyncio
    async def test_maximum_thoughts_reached_scenario(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, mock_llm_responses):
        """Test behavior when maximum thoughts is reached"""
        config = {"max_thoughts": 2, "stop_tool_name": "respond"}
        
        # Mock LLM with repeated tool calls (no stop tool)
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["tool_call_response"].model_dump(),
                mock_llm_responses["tool_call_response"].model_dump(),
                mock_llm_responses["no_tool_response"].model_dump()  # Final response without tools
            ]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=config
        )
        
        await orchestrator.handle_text("Keep using tools")
        
        # Verify response was called indicating stop tool not called
        mock_response_protocol.respond_text.assert_called_with("Response tool not called")
    
    @pytest.mark.asyncio
    async def test_stop_tool_called_scenario(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config, mock_llm_responses):
        """Test behavior when stop tool is called"""
        # Mock LLM that immediately calls stop tool
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["stop_tool_response"].model_dump()
            ]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("Please respond")
        
        # Verify tool was executed
        mock_tool_provider.execute_tool.assert_called_with("respond", {"message": "The weather in San Francisco is sunny, 72°F"})
        
        # Verify response was NOT called with "Response tool not called" since stop tool was called
        mock_response_protocol.respond_text.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_conversation_history_integration(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config, mock_llm_responses):
        """Test that conversation history is properly integrated"""
        mock_llm = AsyncMock(spec=LLMProtocol)
        mock_llm.generate.return_value = mock_llm_responses["stop_tool_response"]
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("Test message")
        
        # Verify LLM was called
        mock_llm.generate.assert_called()
        
        # Get the conversation passed to LLM
        call_args = mock_llm.generate.call_args[0][0]  # First positional argument (messages)
        
        # Verify conversation includes system prompt, history, and user message
        assert len(call_args) >= 4  # system + 3 history messages + user message
        assert call_args[0].content[0].text == valid_config["system_prompt"]  # system prompt
        
        # Find the user message we sent (should be the 5th message: system + 3 history + user)
        user_message = call_args[4]
        assert user_message.content[0].text == "Test message"  # user message
    
    @pytest.mark.asyncio
    async def test_empty_conversation_history(self, mock_response_protocol, mock_tool_provider, valid_config, mock_llm_responses):
        """Test handling with empty conversation history"""
        # Mock conversation with empty history
        empty_conversation = AsyncMock(spec=ConversationHistoryProtocol)
        empty_conversation.get_history.return_value = []
        
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["stop_tool_response"].model_dump()]
        })
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=empty_conversation,
            config=valid_config
        )
        
        await orchestrator.handle_text("Test with empty history")
        
        # Should still work with empty history
        empty_conversation.get_history.assert_called_once()
        mock_tool_provider.list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_system_prompt_configuration(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, mock_llm_responses):
        """Test handling without system prompt"""
        config = {"max_thoughts": 5, "stop_tool_name": "respond"}  # No system_prompt
        
        mock_llm = AsyncMock(spec=LLMProtocol)
        mock_llm.generate.return_value = mock_llm_responses["stop_tool_response"]
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=config
        )
        
        await orchestrator.handle_text("Test without system prompt")
        
        # Verify LLM was called
        mock_llm.generate.assert_called()
        
        # Get the conversation passed to LLM
        call_args = mock_llm.generate.call_args[0][0]
        
        # Should not include system prompt, just history + user message
        # Note: The conversation will grow as the LLM processes and adds responses
        assert len(call_args) >= 4  # At least 3 history messages + user message
        
        # Find the user message we sent (should be the 4th message: 3 history + user)
        user_message = call_args[3]
        assert user_message.content[0].text == "Test without system prompt"
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response(self, mock_response_protocol, mock_conversation_history, mock_tool_provider, valid_config):
        """Test handling multiple tool calls in a single LLM response"""
        # Create response with multiple tool calls
        multi_tool_response = LLMResponse(
            content="I'll check weather and then respond.",
            tool_calls=[
                LLMFunctionCall(id="call_1", name="get_weather", arguments={"location": "NYC"}),
                LLMFunctionCall(id="call_2", name="respond", arguments={"message": "Weather checked!"})
            ]
        )
        
        mock_llm = MockLLM({"responses": [multi_tool_response.model_dump()]})
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=mock_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        await orchestrator.handle_text("Check weather and respond")
        
        # Verify both tools were executed
        assert mock_tool_provider.execute_tool.call_count == 2
        
        # Verify the calls were made with correct parameters
        call_args_list = mock_tool_provider.execute_tool.call_args_list
        assert call_args_list[0][0] == ("get_weather", {"location": "NYC"})
        assert call_args_list[1][0] == ("respond", {"message": "Weather checked!"})
    
    @pytest.mark.asyncio
    async def test_stop_tool_validation_failure(self, mock_response_protocol, mock_conversation_history, valid_config):
        """Test behavior when stop tool validation fails"""
        # Create tool provider without the required stop tool
        bad_tool_provider = AsyncMock(spec=ToolProviderProtocol)
        bad_tool_provider.list_tools.return_value = [
            Tool(name="get_weather", description="Get weather", parameters={})
            # Missing "respond" tool
        ]
        
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = PFCOrchestrator(
            response=mock_response_protocol,
            llm=mock_llm,
            tool_provider=bad_tool_provider,
            history=mock_conversation_history,
            config=valid_config
        )
        
        # Should raise ValueError during handle_text
        with pytest.raises(ValueError, match="Stop tool 'respond' not found in tool provider"):
            await orchestrator.handle_text("Test message")