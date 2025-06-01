import os.path
import pathlib
from typing import List

from xaibo.core.models import Tool
from xaibo.core.protocols import TextMessageHandlerProtocol, ResponseProtocol, LLMProtocol, ToolProviderProtocol, \
    ConversationHistoryProtocol, MemoryProtocol
from xaibo.core.models.llm import LLMMessage, LLMOptions, LLMRole, LLMFunctionResult, LLMMessageContentType, LLMMessageContent

import json


class MemoryOrchestrator(TextMessageHandlerProtocol):
    def __init__(self,
                 llm: LLMProtocol,
                 memory: MemoryProtocol,
                 doer: TextMessageHandlerProtocol,
                 history: ConversationHistoryProtocol,
                 config: dict | None = None
                ):
        """Initialize the MemoryOrchestrator."""
        config = config or {}
        self.llm: LLMProtocol = llm
        self.mid_term_memory: MemoryProtocol = memory
        self.doer: TextMessageHandlerProtocol = doer
        self.history: ConversationHistoryProtocol = history

        self.memory_size = config.get('memory_size', 10)
        self.mid_term_memory_size = config.get('mid_term_memory_size', 3)
        self.memory_file = config.get('memory_file', 'short-term-memory.json')

        self.is_useful_prompt = config.get('is_useful_prompt', 'Is this memory relevant to the current conversation?')
        self.was_useful_prompt = config.get('was_useful_prompt', 'Was this memory useful for the current conversation?')
        self.rewrite_memory_prompt = config.get('rewrite_memory_prompt', 'Rewrite this memory to be even more useful the next time you use it.')
        self.create_memory_prompt = config.get('create_memory_prompt', 'Create a new memory based on the current conversation.')
        self.create_memory_summary_prompt = config.get('create_memory_summary_prompt', 'Summarize this memory in one sentence.')

        self.short_term_memory_prefix = config.get('short_term_memory_prefix', 'These are your most recent memories that are useful for the current conversation:')
        self.mid_term_memory_prefix = config.get('mid_term_memory_prefix', 'These are some summaries of more distant memories with their ids that may be useful for the current conversation:')

        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory_data = json.load(f)
        else:
            self.memory_data = []

    async def handle_text(self, text: str) -> None:
        """Handle an incoming text message.

        Args:
            text: The text message to handle
        """
        orig_history = list(await self.history.get_history())

        useful_memory_list = [memory for memory in self.memory_data if await self._is_useful_memory(memory, orig_history)]
        useful_memories = LLMMessage.user(
            self.short_term_memory_prefix+"\n"+
            "\n".join(useful_memory_list)
        )

        search_results = await self.mid_term_memory.search_memory(text, self.mid_term_memory_size)
        mid_results = LLMMessage.user(
            self.mid_term_memory_prefix+"\n"+
            "\n".join(f"{m.id}. {m.attributes['summary']}" for m in search_results)
        )

        await self.history.clear_history()
        await self.history.add_message(useful_memories)
        await self.history.add_message(mid_results)
        for msg in orig_history:
            await self.history.add_message(msg)

        await self.doer.handle_text(text)

        new_history = list(await self.history.get_history())
        for memory in useful_memory_list:
            self.memory_data.remove(memory)
            if await self._was_useful_memory(memory, new_history):
                new_memory = await self.rewrite_memory(memory, new_history)
                await self.add_memory(new_memory)

        # form new memory
        await self.add_memory(await self.create_memory(new_history))


    async def _is_useful_memory(self, memory: str, context: List[LLMMessage]) -> bool:
        result = await self.llm.generate(
            context + [
                LLMMessage.user(self.is_useful_prompt + "\n" + memory),
            ],
            LLMOptions(functions=[
                Tool(name="useful", description="Call this tool if you think the given memory is going to be useful"),
                Tool(name="useless", description="Call this tool if you think the given memory is not going to be useful"),
            ])
        )
        if result.tool_calls is not None:
            return any(result.tool_calls[0].name == "useful" for result in result.tool_calls)
        else:
            return False

    async def _was_useful_memory(self, memory: str, context: List[LLMMessage]) -> bool:
        result = await self.llm.generate(
            context + [
                LLMMessage.user(self.was_useful_prompt + "\n" + memory),
            ],
            LLMOptions(functions=[
                Tool(name="useful", description="Call this tool if you think the given memory was useful"),
                Tool(name="useless", description="Call this tool if you think the given memory was not useful"),
            ])
        )
        if result.tool_calls is not None:
            return any(result.tool_calls[0].name == "useful" for result in result.tool_calls)
        else:
            return False

    async def rewrite_memory(self, memory: str, history: List[LLMMessage]) -> str:
        result = await self.llm.generate(
            history + [
                LLMMessage.user(self.rewrite_memory_prompt),
                LLMMessage.user(memory)
            ]
        )
        return result.content.strip()

    async def create_memory(self, history: list[LLMMessage]) -> str:
        result = await self.llm.generate(
            history + [
                LLMMessage.user(self.create_memory_prompt),
            ]
        )
        return result.content.strip()

    async def add_memory(self, new_memory: str) -> None:
        if len(self.memory_data) >= self.memory_size:
            last_memory = self.memory_data.pop()
            await self.mid_term_memory.store_memory(last_memory, {"summary": await self._summarize(last_memory), "text": last_memory} )

        self.memory_data.append(new_memory)
        await self._save_memories()

    async def _summarize(self, memory: str) -> str:
        result = await self.llm.generate([
            LLMMessage.system(self.create_memory_summary_prompt),
            LLMMessage.user(memory)
        ])
        return result.content.strip()

    async def _save_memories(self) -> None:
        pathlib.Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory_data, f)