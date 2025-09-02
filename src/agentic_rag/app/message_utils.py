# src/agentic_rag/app/message_utils.py

from typing import List, Optional, Union, Any, Dict, Callable
from langchain_core.messages import BaseMessage, AIMessage


def _msg_type(msg: Union[BaseMessage, Dict[str, Any]]) -> Optional[str]:
    """
    Return normalized message type across BaseMessage and dict-shaped messages.
    
    Args:
        msg: Message object or dict
        
    Returns:
        "human" | "ai" | "system" | None
    """
    if isinstance(msg, BaseMessage):
        return msg.type
    elif isinstance(msg, dict):
        return msg.get("type")
    return None


def get_message_content(msg: Union[BaseMessage, Dict[str, Any]]) -> str:
    """
    Extract message content regardless of message shape.
    
    Args:
        msg: Message object or dict
        
    Returns:
        Message content as string, or empty string if not found
    """
    if isinstance(msg, BaseMessage):
        return msg.content if hasattr(msg, 'content') else ""
    elif isinstance(msg, dict):
        return msg.get("content", "")
    return ""


def get_last_message_content(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> str:
    """
    Safely get the content of the last message.
    
    Args:
        messages: List of messages
        
    Returns:
        Content of last message, or empty string if no messages
    """
    if not messages:
        return ""
    return get_message_content(messages[-1])


def is_human_message(msg: Union[BaseMessage, Dict[str, Any]]) -> bool:
    """
    Check if message is from a human user.
    
    Args:
        msg: Message object or dict
        
    Returns:
        True if message is from human
    """
    return _msg_type(msg) == "human"


def is_ai_message(msg: Union[BaseMessage, Dict[str, Any]]) -> bool:
    """
    Check if message is from AI assistant.
    
    Args:
        msg: Message object or dict
        
    Returns:
        True if message is from AI assistant
    """
    return _msg_type(msg) == "ai"


def get_last_human_message_content(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> str:
    """
    Scan backward for the last human message content.
    
    Args:
        messages: List of messages
        
    Returns:
        Content of last human message, or empty string if not found
    """
    for msg in reversed(messages):
        if is_human_message(msg):
            return get_message_content(msg)
    return ""


def get_last_ai_message_content(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> str:
    """
    Scan backward for the last assistant message content.
    
    Args:
        messages: List of messages
        
    Returns:
        Content of last AI message, or empty string if not found
    """
    for msg in reversed(messages):
        if is_ai_message(msg):
            return get_message_content(msg)
    return ""


def get_last_completed_turn_messages(messages: List[Union[BaseMessage, Dict[str, Any]]]) -> List[Union[BaseMessage, Dict[str, Any]]]:
    """
    Return the most recent completed turn as [human_msg, ai_msg] by scanning backward.
    
    This function looks for the most recent completed human-AI conversation turn,
    ignoring any current in-progress human message at the end.
    
    Args:
        messages: List of messages
        
    Returns:
        List containing [human_message, ai_message] for the last completed turn,
        or empty list if no completed turn is found
    """
    if len(messages) < 2:
        return []
    
    # Start from the end, but skip current human message if it's the last one
    end_idx = len(messages)
    if messages and is_human_message(messages[-1]):
        end_idx = len(messages) - 1
    
    # Look for the most recent AI message
    ai_msg = None
    ai_idx = -1
    for i in range(end_idx - 1, -1, -1):
        if is_ai_message(messages[i]):
            ai_msg = messages[i]
            ai_idx = i
            break
    
    if ai_msg is None:
        return []
    
    # Look for the human message that precedes this AI message
    human_msg = None
    for i in range(ai_idx - 1, -1, -1):
        if is_human_message(messages[i]):
            human_msg = messages[i]
            break
    
    if human_msg is None:
        return []
    
    return [human_msg, ai_msg]


def replace_last_assistant_message(content: str, additional_kwargs: Optional[Dict[str, Any]] = None) -> Callable[[List[BaseMessage], List[BaseMessage]], List[BaseMessage]]:
    """
    Create a custom reducer function to REPLACE the last assistant message.
    
    This returns a function that can be used as a reducer in LangGraph to properly
    replace the last AI message instead of appending a new one.
    
    Args:
        content: New content for the assistant message
        additional_kwargs: Optional additional parameters for the message
        
    Returns:
        A reducer function that replaces the last AI message
    """
    if additional_kwargs is None:
        additional_kwargs = {}
    
    def replace_reducer(existing_messages: List[BaseMessage], new_messages: List[BaseMessage]) -> List[BaseMessage]:
        """Custom reducer that replaces the last AI message."""
        # Start with existing messages
        result = existing_messages.copy()
        
        # Find the last AI message index
        last_ai_idx = -1
        for i in range(len(result) - 1, -1, -1):
            if is_ai_message(result[i]):
                last_ai_idx = i
                break
        
        # Create the replacement message
        replacement_message = AIMessage(content=content, **additional_kwargs)
        
        if last_ai_idx >= 0:
            # Replace the last AI message
            result[last_ai_idx] = replacement_message
        else:
            # No AI message found, append normally
            result.append(replacement_message)
        
        return result
    
    return replace_reducer


def create_replacement_message(content: str, additional_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a replacement message update for LangGraph nodes.
    
    This creates a simpler structure that can be handled by the graph to replace
    the last assistant message. The actual replacement logic should be implemented
    at the graph level or through custom state management.
    
    Args:
        content: New content for the assistant message
        additional_kwargs: Optional additional parameters for the message
        
    Returns:
        Dict with the new message and replacement flag
    """
    if additional_kwargs is None:
        additional_kwargs = {}
    
    # Create new AI message
    new_message = AIMessage(content=content, **additional_kwargs)
    
    # Return with replacement instruction
    return {
        "messages": [new_message],
        "_replace_last_ai": True
    }