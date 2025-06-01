import pytest
import json
import tempfile
import os
from unittest.mock import AsyncMock, patch, mock_open
from typing import List, Dict, Any

from xaibo.core.models import LLMMessage, LLMResponse, LLMUsage
from xaibo.core.models.llm import LLMFunctionCall
from xaibo.core.protocols import MemoryProtocol, ConversationHistoryProtocol, TextMessageHandlerProtocol
from xaibo.primitives.modules.llm.mock import MockLLM

from modules.hippo import MemoryOrchestrator


class NoopTextMessageHandler(TextMessageHandlerProtocol):
    """Noop implementation of TextMessageHandlerProtocol for testing"""
    
    def __init__(self):
        self.handle_text_called = False
        self.last_text = None
    
    async def handle_text(self, text: str) -> None:
        """Handle text message - noop implementation that just records the call"""
        self.handle_text_called = True
        self.last_text = text


@pytest.fixture
def mock_memory_protocol():
    """Create a mock MemoryProtocol for testing"""
    memory = AsyncMock(spec=MemoryProtocol)
    
    # Mock search results
    from xaibo.core.protocols.memory import MemorySearchResult
    mock_results = [
        MemorySearchResult(
            memory_id="mem_1",
            content="Previous weather conversation",
            similarity_score=0.8,
            attributes={"summary": "User asked about weather in SF"}
        ),
        MemorySearchResult(
            memory_id="mem_2",
            content="Email discussion",
            similarity_score=0.7,
            attributes={"summary": "User wanted to send an email"}
        )
    ]
    memory.search_memory.return_value = mock_results
    memory.store_memory.return_value = None
    
    return memory


@pytest.fixture
def mock_conversation_history():
    """Create a mock conversation history"""
    conversation = AsyncMock(spec=ConversationHistoryProtocol)
    conversation.get_history.return_value = [
        LLMMessage.user("Hello, I need help with something"),
        LLMMessage.assistant("I can help you with that. What do you need?"),
        LLMMessage.user("I want to check the weather")
    ]
    conversation.clear_history.return_value = None
    conversation.add_message.return_value = None
    return conversation


@pytest.fixture
def noop_doer():
    """Create a noop TextMessageHandlerProtocol for testing"""
    return NoopTextMessageHandler()


@pytest.fixture
def valid_config():
    """Create a valid configuration for MemoryOrchestrator"""
    return {
        "memory_size": 5,
        "mid_term_memory_size": 2,
        "memory_file": "test-memory.json",
        "is_useful_prompt": "Is this memory useful for the current context?",
        "was_useful_prompt": "Was this memory helpful in the conversation?",
        "rewrite_memory_prompt": "Improve this memory for future use.",
        "create_memory_prompt": "Create a memory from this conversation.",
        "create_memory_summary_prompt": "Summarize this memory briefly.",
        "short_term_memory_prefix": "Recent memories:",
        "mid_term_memory_prefix": "Distant memories:"
    }


@pytest.fixture
def mock_llm_responses():
    """Create mock LLM responses for various scenarios"""
    return {
        "useful_memory": LLMResponse(
            content="This memory is useful",
            tool_calls=[
                LLMFunctionCall(
                    id="call_123",
                    name="useful",
                    arguments={}
                )
            ],
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        ),
        "useless_memory": LLMResponse(
            content="This memory is not useful",
            tool_calls=[
                LLMFunctionCall(
                    id="call_456",
                    name="useless",
                    arguments={}
                )
            ],
            usage=LLMUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12)
        ),
        "no_tool_response": LLMResponse(
            content="I cannot determine usefulness",
            usage=LLMUsage(prompt_tokens=6, completion_tokens=3, total_tokens=9)
        ),
        "rewrite_response": LLMResponse(
            content="Improved memory: User prefers weather updates in the morning",
            usage=LLMUsage(prompt_tokens=15, completion_tokens=8, total_tokens=23)
        ),
        "create_memory_response": LLMResponse(
            content="User asked about weather and received helpful information",
            usage=LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        ),
        "summary_response": LLMResponse(
            content="Weather conversation summary",
            usage=LLMUsage(prompt_tokens=12, completion_tokens=6, total_tokens=18)
        )
    }


@pytest.fixture
def temp_memory_file():
    """Create a temporary memory file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_memories = ["Memory 1", "Memory 2", "Memory 3"]
        json.dump(test_memories, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


class TestMemoryOrchestratorInitialization:
    """Test MemoryOrchestrator initialization and configuration validation"""
    
    def test_valid_initialization(self, mock_memory_protocol, mock_conversation_history, noop_doer, valid_config):
        """Test successful initialization with valid configuration"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test response").model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=valid_config
        )
        
        assert orchestrator.llm == mock_llm
        assert orchestrator.mid_term_memory == mock_memory_protocol
        assert orchestrator.doer == noop_doer
        assert orchestrator.history == mock_conversation_history
        assert orchestrator.memory_size == valid_config["memory_size"]
        assert orchestrator.mid_term_memory_size == valid_config["mid_term_memory_size"]
        assert orchestrator.memory_file == valid_config["memory_file"]
        assert orchestrator.is_useful_prompt == valid_config["is_useful_prompt"]
        assert orchestrator.was_useful_prompt == valid_config["was_useful_prompt"]
        assert orchestrator.rewrite_memory_prompt == valid_config["rewrite_memory_prompt"]
        assert orchestrator.create_memory_prompt == valid_config["create_memory_prompt"]
        assert orchestrator.create_memory_summary_prompt == valid_config["create_memory_summary_prompt"]
        assert orchestrator.short_term_memory_prefix == valid_config["short_term_memory_prefix"]
        assert orchestrator.mid_term_memory_prefix == valid_config["mid_term_memory_prefix"]
    
    def test_initialization_with_none_config(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test initialization with None config uses defaults"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=None
        )
        
        assert orchestrator.memory_size == 10
        assert orchestrator.mid_term_memory_size == 3
        assert orchestrator.memory_file == 'short-term-memory.json'
        assert orchestrator.is_useful_prompt == 'Is this memory relevant to the current conversation?'
        assert orchestrator.was_useful_prompt == 'Was this memory useful for the current conversation?'
        assert orchestrator.rewrite_memory_prompt == 'Rewrite this memory to be even more useful the next time you use it.'
        assert orchestrator.create_memory_prompt == 'Create a new memory based on the current conversation.'
        assert orchestrator.create_memory_summary_prompt == 'Summarize this memory in one sentence.'
        assert orchestrator.short_term_memory_prefix == 'These are your most recent memories that are useful for the current conversation:'
        assert orchestrator.mid_term_memory_prefix == 'These are some summaries of more distant memories with their ids that may be useful for the current conversation:'
    
    def test_initialization_with_empty_config(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test initialization with empty config uses defaults"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config={}
        )
        
        assert orchestrator.memory_size == 10
        assert orchestrator.mid_term_memory_size == 3
        assert orchestrator.memory_file == 'short-term-memory.json'
    
    def test_initialization_with_partial_config(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test initialization with partial config"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        partial_config = {"memory_size": 7, "memory_file": "custom.json"}
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=partial_config
        )
        
        assert orchestrator.memory_size == 7  # from config
        assert orchestrator.memory_file == "custom.json"  # from config
        assert orchestrator.mid_term_memory_size == 3  # default
        assert orchestrator.is_useful_prompt == 'Is this memory relevant to the current conversation?'  # default
    
    def test_initialization_loads_existing_memory_file(self, mock_memory_protocol, mock_conversation_history, noop_doer, temp_memory_file):
        """Test initialization loads existing memory file"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        config = {"memory_file": temp_memory_file}
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=config
        )
        
        assert orchestrator.memory_data == ["Memory 1", "Memory 2", "Memory 3"]
    
    def test_initialization_creates_empty_memory_for_nonexistent_file(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test initialization creates empty memory when file doesn't exist"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        config = {"memory_file": "nonexistent-file.json"}
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=config
        )
        
        assert orchestrator.memory_data == []


class TestMemoryOrchestratorMemoryEvaluation:
    """Test MemoryOrchestrator memory evaluation methods"""
    
    @pytest.mark.asyncio
    async def test_is_useful_memory_returns_true_when_useful_tool_called(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _is_useful_memory returns True when useful tool is called"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["useful_memory"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        context = [LLMMessage.user("Test context")]
        result = await orchestrator._is_useful_memory("Test memory", context)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_is_useful_memory_returns_false_when_useless_tool_called(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _is_useful_memory returns False when useless tool is called"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["useless_memory"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        context = [LLMMessage.user("Test context")]
        result = await orchestrator._is_useful_memory("Test memory", context)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_useful_memory_returns_false_when_no_tools_called(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _is_useful_memory returns False when no tools are called"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["no_tool_response"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        context = [LLMMessage.user("Test context")]
        result = await orchestrator._is_useful_memory("Test memory", context)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_was_useful_memory_returns_true_when_useful_tool_called(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _was_useful_memory returns True when useful tool is called"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["useful_memory"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        context = [LLMMessage.user("Test context")]
        result = await orchestrator._was_useful_memory("Test memory", context)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_was_useful_memory_returns_false_when_useless_tool_called(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _was_useful_memory returns False when useless tool is called"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["useless_memory"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        context = [LLMMessage.user("Test context")]
        result = await orchestrator._was_useful_memory("Test memory", context)
        
        assert result is False


class TestMemoryOrchestratorMemoryOperations:
    """Test MemoryOrchestrator memory operations"""
    
    @pytest.mark.asyncio
    async def test_rewrite_memory(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test rewrite_memory method"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["rewrite_response"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        history = [LLMMessage.user("Test history")]
        result = await orchestrator.rewrite_memory("Old memory", history)
        
        assert result == "Improved memory: User prefers weather updates in the morning"
    
    @pytest.mark.asyncio
    async def test_rewrite_memory_handles_empty_content(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test rewrite_memory handles empty LLM response content"""
        empty_response = LLMResponse(
            content="",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=0, total_tokens=5)
        )
        mock_llm = MockLLM({"responses": [empty_response.model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        history = [LLMMessage.user("Test history")]
        result = await orchestrator.rewrite_memory("Old memory", history)
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_create_memory(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test create_memory method"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["create_memory_response"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        history = [LLMMessage.user("Weather conversation")]
        result = await orchestrator.create_memory(history)
        
        assert result == "User asked about weather and received helpful information"
    
    @pytest.mark.asyncio
    async def test_create_memory_handles_empty_content(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test create_memory handles empty LLM response content"""
        empty_response = LLMResponse(
            content="",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=0, total_tokens=5)
        )
        mock_llm = MockLLM({"responses": [empty_response.model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        history = [LLMMessage.user("Test history")]
        result = await orchestrator.create_memory(history)
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_summarize(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test _summarize method"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["summary_response"].model_dump()]
        })
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        result = await orchestrator._summarize("Long memory about weather conversation")
        
        assert result == "Weather conversation summary"
    
    @pytest.mark.asyncio
    async def test_add_memory_appends_when_under_limit(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test add_memory appends when under memory size limit"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history,
                    config={"memory_size": 3}
                )
                
                orchestrator.memory_data = ["Memory 1", "Memory 2"]
                
                await orchestrator.add_memory("New memory")
                
                assert len(orchestrator.memory_data) == 3
                assert orchestrator.memory_data[-1] == "New memory"
                mock_memory_protocol.store_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_add_memory_moves_to_mid_term_when_at_limit(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test add_memory moves oldest memory to mid-term storage when at limit"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["summary_response"].model_dump()]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history,
                    config={"memory_size": 2}
                )
                
                orchestrator.memory_data = ["Memory 1", "Memory 2"]
                
                await orchestrator.add_memory("New memory")
                
                assert len(orchestrator.memory_data) == 2
                assert orchestrator.memory_data == ["Memory 2", "New memory"]
                mock_memory_protocol.store_memory.assert_called_once_with(
                    "Memory 1", 
                    {"summary": "Weather conversation summary", "text": "Memory 1"}
                )


class TestMemoryOrchestratorPersistence:
    """Test MemoryOrchestrator memory persistence functionality"""
    
    @pytest.mark.asyncio
    async def test_save_memories_creates_directory_and_writes_file(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test _save_memories creates directory and writes file"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                with patch('json.dump') as mock_json_dump:
                    orchestrator = MemoryOrchestrator(
                        llm=mock_llm,
                        memory=mock_memory_protocol,
                        doer=noop_doer,
                        history=mock_conversation_history,
                        config={"memory_file": "test/path/memory.json"}
                    )
                    
                    orchestrator.memory_data = ["Memory 1", "Memory 2"]
                    
                    await orchestrator._save_memories()
                    
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                    mock_file.assert_called_once_with("test/path/memory.json", 'w')
                    mock_json_dump.assert_called_once_with(["Memory 1", "Memory 2"], mock_file.return_value.__enter__.return_value)
    
    def test_initialization_handles_json_decode_error(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test initialization handles corrupted JSON file gracefully"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                with pytest.raises(json.JSONDecodeError):
                    MemoryOrchestrator(
                        llm=mock_llm,
                        memory=mock_memory_protocol,
                        doer=noop_doer,
                        history=mock_conversation_history,
                        config={"memory_file": "corrupted.json"}
                    )


class TestMemoryOrchestratorTextHandling:
    """Test MemoryOrchestrator text handling functionality"""
    
    @pytest.mark.asyncio
    async def test_handle_text_full_workflow(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test complete handle_text workflow"""
        # Setup mock LLM with responses for: useful check, was_useful check, rewrite, create_memory
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["useful_memory"].model_dump(),  # _is_useful_memory
                mock_llm_responses["useful_memory"].model_dump(),  # _was_useful_memory
                mock_llm_responses["rewrite_response"].model_dump(),  # rewrite_memory
                mock_llm_responses["create_memory_response"].model_dump()  # create_memory
            ]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history,
                    config={"memory_size": 5}
                )
                
                orchestrator.memory_data = ["Useful memory"]
                
                await orchestrator.handle_text("What's the weather like?")
                
                # Verify doer was called
                assert noop_doer.handle_text_called
                assert noop_doer.last_text == "What's the weather like?"
                
                # Verify history operations
                mock_conversation_history.clear_history.assert_called_once()
                assert mock_conversation_history.add_message.call_count >= 2  # useful memories + mid-term memories
                
                # Verify mid-term memory search
                mock_memory_protocol.search_memory.assert_called_once_with("What's the weather like?", 3)
    
    @pytest.mark.asyncio
    async def test_handle_text_with_no_useful_memories(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test handle_text when no memories are deemed useful"""
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["useless_memory"].model_dump(),  # _is_useful_memory
                mock_llm_responses["create_memory_response"].model_dump()  # create_memory
            ]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history
                )
                
                orchestrator.memory_data = ["Not useful memory"]
                
                await orchestrator.handle_text("Test message")
                
                # Memory should still be in memory_data since it wasn't useful
                assert "Not useful memory" in orchestrator.memory_data
                
                # Verify doer was called
                assert noop_doer.handle_text_called
    
    @pytest.mark.asyncio
    async def test_handle_text_with_empty_memory_data(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test handle_text with empty memory data"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["create_memory_response"].model_dump()]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history
                )
                
                orchestrator.memory_data = []
                
                await orchestrator.handle_text("Test message")
                
                # Verify doer was called
                assert noop_doer.handle_text_called
                
                # Verify new memory was created
                assert len(orchestrator.memory_data) == 1
                assert orchestrator.memory_data[0] == "User asked about weather and received helpful information"
    
    @pytest.mark.asyncio
    async def test_handle_text_memory_context_integration(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test that memory context is properly integrated into conversation history"""
        mock_llm = MockLLM({
            "responses": [
                mock_llm_responses["useful_memory"].model_dump(),  # _is_useful_memory
                mock_llm_responses["useful_memory"].model_dump(),  # _was_useful_memory
                mock_llm_responses["rewrite_response"].model_dump(),  # rewrite_memory
                mock_llm_responses["create_memory_response"].model_dump()  # create_memory
            ]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history,
                    config={
                        "short_term_memory_prefix": "Recent memories:",
                        "mid_term_memory_prefix": "Distant memories:"
                    }
                )
                
                orchestrator.memory_data = ["Useful memory"]
                
                await orchestrator.handle_text("Test message")
                
                # Verify add_message was called with memory contexts
                add_message_calls = mock_conversation_history.add_message.call_args_list
                
                # Should have at least 2 calls: useful memories + mid-term memories
                assert len(add_message_calls) >= 2
                
                # Check that memory contexts were added
                memory_messages = [call[0][0] for call in add_message_calls]
                memory_contents = [msg.content[0].text for msg in memory_messages]
                
                # Should contain both short-term and mid-term memory contexts
                assert any("Recent memories:" in content for content in memory_contents)
                assert any("Distant memories:" in content for content in memory_contents)


class TestMemoryOrchestratorEdgeCases:
    """Test MemoryOrchestrator edge cases and error scenarios"""
    
    @pytest.mark.asyncio
    async def test_handle_text_with_memory_evaluation_errors(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test handle_text when memory evaluation methods raise exceptions"""
        # Mock LLM that raises an exception
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history
                )
                
                orchestrator.memory_data = ["Test memory"]
                
                # Should raise the exception since we're not handling it in the implementation
                with pytest.raises(Exception, match="LLM error"):
                    await orchestrator.handle_text("Test message")
    
    @pytest.mark.asyncio
    async def test_handle_text_with_mid_term_memory_search_error(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test handle_text when mid-term memory search fails"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["create_memory_response"].model_dump()]
        })
        
        # Mock memory protocol to raise exception on search
        mock_memory_protocol.search_memory.side_effect = Exception("Memory search failed")
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history
                )
                
                orchestrator.memory_data = []
                
                # Should raise the exception since we're not handling it in the implementation
                with pytest.raises(Exception, match="Memory search failed"):
                    await orchestrator.handle_text("Test message")
    
    @pytest.mark.asyncio
    async def test_handle_text_with_doer_error(self, mock_memory_protocol, mock_conversation_history, mock_llm_responses):
        """Test handle_text when doer.handle_text raises exception"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["create_memory_response"].model_dump()]
        })
        
        # Mock doer that raises exception
        error_doer = AsyncMock(spec=TextMessageHandlerProtocol)
        error_doer.handle_text.side_effect = Exception("Doer error")
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=error_doer,
                    history=mock_conversation_history
                )
                
                orchestrator.memory_data = []
                
                # Should raise the exception since we're not handling it in the implementation
                with pytest.raises(Exception, match="Doer error"):
                    await orchestrator.handle_text("Test message")
    
    @pytest.mark.asyncio
    async def test_memory_operations_with_whitespace_content(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test memory operations handle whitespace-only content correctly"""
        whitespace_response = LLMResponse(
            content="   \n\t  ",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        )
        mock_llm = MockLLM({"responses": [whitespace_response.model_dump()]})
        
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history
        )
        
        history = [LLMMessage.user("Test history")]
        result = await orchestrator.rewrite_memory("Old memory", history)
        
        # Should strip whitespace and return empty string
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_add_memory_with_empty_string(self, mock_memory_protocol, mock_conversation_history, noop_doer, mock_llm_responses):
        """Test add_memory with empty string"""
        mock_llm = MockLLM({
            "responses": [mock_llm_responses["summary_response"].model_dump()]
        })
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir'):
                orchestrator = MemoryOrchestrator(
                    llm=mock_llm,
                    memory=mock_memory_protocol,
                    doer=noop_doer,
                    history=mock_conversation_history,
                    config={"memory_size": 2}
                )
                
                orchestrator.memory_data = []
                
                await orchestrator.add_memory("")
                
                # Should still add the empty memory
                assert len(orchestrator.memory_data) == 1
                assert orchestrator.memory_data[0] == ""
    
    @pytest.mark.asyncio
    async def test_configuration_edge_cases(self, mock_memory_protocol, mock_conversation_history, noop_doer):
        """Test configuration with edge case values"""
        mock_llm = MockLLM({"responses": [LLMResponse(content="Test").model_dump()]})
        
        # Test with zero memory size
        config = {"memory_size": 0}
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=config
        )
        
        assert orchestrator.memory_size == 0
        
        # Test with negative values (should still work, though not practical)
        config = {"memory_size": -1, "mid_term_memory_size": -1}
        orchestrator = MemoryOrchestrator(
            llm=mock_llm,
            memory=mock_memory_protocol,
            doer=noop_doer,
            history=mock_conversation_history,
            config=config
        )
        
        assert orchestrator.memory_size == -1
        assert orchestrator.mid_term_memory_size == -1