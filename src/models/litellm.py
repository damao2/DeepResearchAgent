import warnings
import json # Add json import
import uuid # Add uuid import
import re # Add re import
from typing import Dict, List, Optional, Any

from src.models.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             ChatMessageToolCall, # Import ChatMessageToolCall
                             ChatMessageToolCallDefinition # Import ChatMessageToolCallDefinition
                             )
from src.models.message_manager import (
    MessageManager
)

class LiteLLMModel(ApiModel):
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_base=None,
        api_key=None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        http_client=None,
        **kwargs,
    ):
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        self.http_client = http_client

        self.message_manager = MessageManager(model_id=model_id)

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """Create the LiteLLM client."""
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        # Set global litellm API base and key if provided
        if self.api_base:
            litellm.api_base = self.api_base
        if self.api_key:
            litellm.api_key = self.api_key
        
        # Explicitly set global api_type to openai for custom OpenAI-compatible endpoints
        # This might help litellm correctly identify the provider when api_base is custom
        if self.api_base and "ai.gitee.com" in self.api_base:
            litellm.api_type = "openai"

            # Attempt to add model mapping for litellm
            # This tells litellm that 'Qwen2.5-72B-Instruct' should be treated as an openai model
            # with the specified api_base.
            try:
                litellm.add_model(
                    "openai", # provider
                    "Qwen2.5-72B-Instruct", # model name as used by the user
                    api_base=self.api_base,
                    api_key=self.api_key,
                    api_type="openai"
                )
            except Exception as e:
                # This might fail if the model is already added or due to other reasons.
                # We'll just print a warning.
                print(f"Warning: litellm.add_model failed: {e}")

        return litellm

    def _prepare_completion_kwargs(
            self,
            messages: List[Dict[str, str]],
            stop_sequences: Optional[List[str]] = None,
            grammar: Optional[str] = None,
            tools_to_call_from: Optional[List[Any]] = None,
            custom_role_conversions: Optional[Dict[str, str]] = None,
            convert_images_to_image_urls: bool = False,
            http_client=None,
            timeout: Optional[int] = 300,
            **kwargs,
    ) -> Dict:
        # Add debug prints for self.model_id and self.api_base
        print(f"DEBUG (prepare_kwargs): self.model_id = {self.model_id}")
        print(f"DEBUG (prepare_kwargs): self.api_base = {self.api_base}")

        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=self.flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle timeout
        if timeout is not None:
            completion_kwargs["timeout"] = timeout

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [self.message_manager.get_tool_json_schema(tool,
                                   model_id=self.model_id) for tool in tools_to_call_from],
                    "tool_choice": "auto", # Changed from "required" to "auto"
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)
        
        # Special handling for Qwen2.5-72B-Instruct with ai.gitee.com
        # Convert model_id to litellm's required format if it's the problematic one
        # This must be done AFTER kwargs.update(kwargs) to ensure it's not overridden
        if self.model_id == "Qwen2.5-72B-Instruct" and self.api_base and "ai.gitee.com" in self.api_base:
            completion_kwargs["model"] = "openai/Qwen2.5-72B-Instruct"
        else:
            # Ensure model is set if not handled by special case
            completion_kwargs["model"] = self.model_id


        # Add debug print after kwargs update and special handling
        print(f"DEBUG (prepare_kwargs): completion_kwargs['model'] after all updates = {completion_kwargs.get('model')}")

        if http_client:
            completion_kwargs['client'] = http_client

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(completion_kwargs)

        return completion_kwargs

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            http_client=self.http_client,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        # Explicitly pass model, api_base, api_key to litellm.completion
        # This might help litellm correctly identify the provider
        model_id_for_completion = completion_kwargs.pop("model")
        api_base_for_completion = completion_kwargs.pop("api_base", None)
        api_key_for_completion = completion_kwargs.pop("api_key", None)

        # Add a print statement to debug the model_id
        print(f"DEBUG: model_id_for_completion = {model_id_for_completion}")
        #print(f"DEBUG: api_base_for_completion = {api_base_for_completion}")
       # print(f"DEBUG: api_key_for_completion = {'<hidden>' if api_key_for_completion else 'None'}")
       # print(f"DEBUG: remaining completion_kwargs = {completion_kwargs}")

        response = self.client.completion(
            model=model_id_for_completion,
            api_base=api_base_for_completion,
            api_key=api_key_for_completion,
            api_type="openai", # Explicitly set api_type for custom OpenAI-compatible endpoints
            **completion_kwargs
        )

        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        # Extract message content and tool calls from the response
        message_content = response.choices[0].message.content
        message_tool_calls = response.choices[0].message.tool_calls

        # Initialize ChatMessage with role and content
        first_message = ChatMessage.from_dict(
            {"role": response.choices[0].message.role, "content": message_content},
            raw=response,
        )

        # If litellm already parsed tool_calls, use them
        if message_tool_calls:
            first_message.tool_calls = []
            for tool_call in message_tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass # If it's a string but not valid JSON, keep it as is or handle as error
                first_message.tool_calls.append(
                    ChatMessageToolCall(
                        id=str(uuid.uuid4()),
                        type="function",
                        function=ChatMessageToolCallDefinition(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )
        elif message_content and isinstance(message_content, str):
            # Attempt to parse tool calls from content if tool_calls field is empty
            # This handles cases where the model puts tool call info in the content string
            try:
                # Use regex to find a JSON object that looks like a tool call
                # This regex looks for a JSON object containing "name" and "arguments" keys
                # It's a heuristic and might need refinement based on actual model output patterns
                tool_call_pattern = r"\{.*?\"name\":\s*\"([^\"]+)\".*?\"arguments\":\s*(\{.*?\})\}"
                match = re.search(tool_call_pattern, message_content, re.DOTALL)

                if match:
                    tool_name = match.group(1)
                    arguments_str = match.group(2)
                    arguments = json.loads(arguments_str)

                    first_message.tool_calls = [
                        ChatMessageToolCall(
                            id=str(uuid.uuid4()),
                            type="function",
                            function=ChatMessageToolCallDefinition(
                                name=tool_name,
                                arguments=arguments,
                            ),
                        )
                    ]
                    # Optionally, remove the tool call text from the content
                    # This part is tricky as the content might contain other useful text
                    # For now, let's keep the content as is, or set it to None if only tool call was present
                    # If the entire content was just the tool call, clear it.
                    if message_content.strip() == match.group(0).strip():
                        first_message.content = None
                    else:
                        # Remove the matched tool call string from the content
                        first_message.content = message_content.replace(match.group(0), "").strip()

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse tool call JSON from content string: {e}")
            except Exception as e:
                print(f"Warning: An unexpected error occurred while parsing tool call from content: {e}")

        return self.postprocess_message(first_message, tools_to_call_from)

    def parse_tool_calls(self, raw_output: Any) -> List[ChatMessageToolCall]: # Change return type
        """
        Parses tool calls from the raw output of the LiteLLM model.
        This method is required by the smolagents framework for tool call parsing.
        """
        tool_calls = []
        # Check if tool_calls exist in the message object
        if hasattr(raw_output.choices[0].message, 'tool_calls') and raw_output.choices[0].message.tool_calls:
            for tool_call in raw_output.choices[0].message.tool_calls:
                # Ensure tool_call.function.arguments is a dictionary
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # If it's a string but not valid JSON, keep it as is or handle as error
                        pass 

                tool_calls.append(
                    ChatMessageToolCall( # Use ChatMessageToolCall
                        id=str(uuid.uuid4()), # Generate a unique ID
                        type="function",
                        function=ChatMessageToolCallDefinition( # Use ChatMessageToolCallDefinition
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )
        return tool_calls
