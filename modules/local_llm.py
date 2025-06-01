import os
import json
import logging
from typing import List, Optional, AsyncIterator, Dict, Any
import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from xaibo.core.protocols.llm import LLMProtocol
from xaibo.core.models.llm import LLMMessage, LLMMessageContentType, LLMOptions, LLMResponse, LLMFunctionCall, LLMUsage, LLMRole


logger = logging.getLogger(__name__)


class LocalLLM(LLMProtocol):
    """Implementation of LLMProtocol using Hugging Face transformers with configurable models"""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the Local LLM client.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - torch_dtype: PyTorch data type. Default is "auto".
                - device_map: Device mapping strategy. Default is "auto".
                - max_new_tokens: Maximum number of new tokens to generate. Default is 512.
                - Any additional keys will be passed as arguments to the model.generate() method.
            model_name: The Hugging Face model ID to use for generation.
                       Default is "Qwen/Qwen2.5-3B-Instruct".
        """
        config = config or {}
        
        # Use the model_name parameter, but allow config to override for backward compatibility
        self.model_name = config.get('model_name', 'Qwen/Qwen2.5-3B-Instruct')
        self.torch_dtype = config.get('torch_dtype', "auto")
        self.device_map = config.get('device_map', "auto")
        self.max_new_tokens = config.get('max_new_tokens', 512)
        
        # Initialize model and tokenizer
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
        
        # Store any additional parameters as default kwargs for generation
        self.default_kwargs = {k: v for k, v in config.items() 
                              if k not in ['model_name', 'torch_dtype', 'device_map', 'max_new_tokens']}
    
    def _prepare_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert our messages to chat format compatible with the loaded model"""
        prepared_messages = []
        
        for msg in messages:
            if msg.role == LLMRole.FUNCTION:
                # Handle function calls and results
                if msg.tool_calls:
                    # Convert tool calls to assistant message with function call info
                    tool_call_text = ""
                    for tool_call in msg.tool_calls:
                        tool_call_text += f"Function call: {tool_call.name}({json.dumps(tool_call.arguments)})\n"
                    
                    prepared_messages.append({
                        "role": "assistant",
                        "content": tool_call_text.strip()
                    })
                elif msg.tool_results:
                    # Convert tool results to user message
                    result_text = ""
                    for result in msg.tool_results:
                        result_text += f"Function result: {result.content}\n"
                    
                    prepared_messages.append({
                        "role": "user", 
                        "content": result_text.strip()
                    })
                else:
                    logger.warning("Malformed function message - missing both tool_calls and tool_results")
            else:
                # Handle text content (images not supported in this implementation)
                content = ""
                for c in msg.content:
                    if c.type == LLMMessageContentType.TEXT:
                        content += c.text
                    else:
                        logger.warning("Image content not supported in Qwen implementation")
                
                # Map roles to Qwen format
                role_mapping = {
                    LLMRole.SYSTEM: "system",
                    LLMRole.USER: "user", 
                    LLMRole.ASSISTANT: "assistant"
                }
                
                prepared_messages.append({
                    "role": role_mapping.get(msg.role, "user"),
                    "content": content
                })
            
        return prepared_messages
    
    def _prepare_functions(self, options: LLMOptions) -> Optional[str]:
        """Prepare function calling information as text (since most models don't have native function calling)"""
        if not options.functions:
            return None
            
        function_descriptions = []
        for tool in options.functions:
            params_desc = []
            for param_name, param in tool.parameters.items():
                param_desc = f"{param_name} ({param.type})"
                if param.description:
                    param_desc += f": {param.description}"
                if param.required:
                    param_desc += " [required]"
                if param.default is not None:
                    param_desc += f" [default: {param.default}]"
                params_desc.append(param_desc)
            
            function_desc = f"Function: {tool.name}\nDescription: {tool.description}\nParameters: {', '.join(params_desc)}"
            function_descriptions.append(function_desc)
        
        return "\n\n".join(function_descriptions)

    async def generate(
        self,
        messages: List[LLMMessage],
        options: Optional[LLMOptions] = None
    ) -> LLMResponse:
        """Generate a response using the loaded model"""
        options = options or LLMOptions()
        
        try:
            # Prepare messages
            chat_messages = self._prepare_messages(messages)
            
            # Add function information if available
            function_info = self._prepare_functions(options)
            if function_info:
                # Add function information as system message
                chat_messages.insert(0, {
                    "role": "system",
                    "content": f"You have access to the following functions. When you need to call a function, respond with 'Function call: function_name({{\"param\": \"value\"}})'\n\n{function_info}"
                })
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Prepare generation kwargs
            generation_kwargs = {
                "max_new_tokens": options.max_tokens or self.max_new_tokens,
                "do_sample": True,
                "temperature": options.temperature or 0.7,
                "top_p": options.top_p or 0.9,
                **self.default_kwargs,
                **options.vendor_specific
            }
            
            # Remove None values
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            
            # Run generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            generated_ids = await loop.run_in_executor(
                None,
                lambda: self.model.generate(**model_inputs, **generation_kwargs)
            )
            
            # Extract only the new tokens
            new_tokens = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode response
            response_text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            # Parse potential function calls
            tool_calls = None
            if "Function call:" in response_text:
                # Simple function call parsing (this could be made more robust)
                try:
                    import re
                    pattern = r"Function call: (\w+)\(({.*?})\)"
                    matches = re.findall(pattern, response_text)
                    if matches:
                        tool_calls = []
                        for i, (func_name, args_str) in enumerate(matches):
                            try:
                                args = json.loads(args_str)
                                tool_calls.append(LLMFunctionCall(
                                    id=f"call_{i}",
                                    name=func_name,
                                    arguments=args
                                ))
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse function arguments: {args_str}")
                except Exception as e:
                    logger.warning(f"Failed to parse function calls: {str(e)}")
            
            # Calculate approximate usage (since transformers doesn't provide exact counts)
            input_tokens = model_inputs.input_ids.shape[1]
            output_tokens = len(new_tokens[0])
            usage = LLMUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
            
            return LLMResponse(
                content=response_text,
                tool_calls=tool_calls,
                usage=usage,
                vendor_specific={"model": self.model_name}
            )
            
        except Exception as e:
            logger.error(f"Error generating response from {self.model_name}: {str(e)}")
            raise
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        options: Optional[LLMOptions] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response using the loaded model"""
        options = options or LLMOptions()
        
        try:
            # Prepare messages
            chat_messages = self._prepare_messages(messages)
            
            # Add function information if available
            function_info = self._prepare_functions(options)
            if function_info:
                chat_messages.insert(0, {
                    "role": "system",
                    "content": f"You have access to the following functions. When you need to call a function, respond with 'Function call: function_name({{\"param\": \"value\"}})'\n\n{function_info}"
                })
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Prepare generation kwargs
            generation_kwargs = {
                "max_new_tokens": options.max_tokens or self.max_new_tokens,
                "do_sample": True,
                "temperature": options.temperature or 0.7,
                "top_p": options.top_p or 0.9,
                **self.default_kwargs,
                **options.vendor_specific
            }
            
            # Remove None values
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
            
            # For streaming, we'll simulate it by generating the full response and yielding chunks
            # Note: True streaming would require a more complex implementation with model.generate() streaming
            loop = asyncio.get_event_loop()
            generated_ids = await loop.run_in_executor(
                None,
                lambda: self.model.generate(**model_inputs, **generation_kwargs)
            )
            
            # Extract only the new tokens
            new_tokens = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode response
            response_text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            
            # Simulate streaming by yielding chunks
            chunk_size = 10  # Characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield chunk
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
                     
        except Exception as e:
            logger.error(f"Error generating streaming response from {self.model_name}: {str(e)}")
            raise