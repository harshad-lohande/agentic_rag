# src/agentic_rag/app/hf_chat.py

from typing import List, Optional

from pydantic import PrivateAttr
from huggingface_hub import InferenceClient

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class HFChatInference(BaseChatModel):
    """
    LangChain ChatModel that uses huggingface_hub.InferenceClient chat.completions.
    Works with providers/models that only expose the conversational route.
    """

    # Pydantic fields (must be provided to BaseChatModel via super().__init__)
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 1.0
    timeout: int = 60
    token: Optional[str] = None

    # Non-validated/private attr for the underlying HF client
    _client: InferenceClient = PrivateAttr(default=None)

    def __init__(
        self,
        model: str,
        token: Optional[str] = None,
        timeout: int = 60,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
    ):
        # IMPORTANT: pass fields to BaseChatModel (Pydantic v2)
        super().__init__(
            model=model,
            token=token,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        # Create the HF Inference client
        self._client = InferenceClient(model=model, token=token, timeout=timeout)

    @property
    def _llm_type(self) -> str:
        return "hf_chat_inference"

    @staticmethod
    def _to_chat_dicts(messages: List[BaseMessage]) -> List[dict]:
        out: List[dict] = []
        for m in messages:
            role = "user"
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            # Fallback: leave role as "user" for other types
            out.append({"role": role, "content": m.content})
        return out

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        payload = self._to_chat_dicts(messages)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            # Note: HF chat.completions may not universally support `stop` across providers; omit for safety.
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])
