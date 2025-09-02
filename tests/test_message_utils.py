# tests/test_message_utils.py

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agentic_rag.app.message_utils import (
    _msg_type,
    get_message_content,
    get_last_message_content,
    is_human_message,
    is_ai_message,
    get_last_human_message_content,
    get_last_ai_message_content,
    get_last_completed_turn_messages,
    create_replacement_message,
)


class TestMessageUtils:
    def test_msg_type_with_base_message(self):
        """Test _msg_type with BaseMessage objects."""
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there")
        system_msg = SystemMessage(content="System prompt")
        
        assert _msg_type(human_msg) == "human"
        assert _msg_type(ai_msg) == "ai"
        assert _msg_type(system_msg) == "system"
    
    def test_msg_type_with_dict(self):
        """Test _msg_type with dict-shaped messages."""
        human_dict = {"type": "human", "content": "Hello"}
        ai_dict = {"type": "ai", "content": "Hi there"}
        empty_dict = {}
        
        assert _msg_type(human_dict) == "human"
        assert _msg_type(ai_dict) == "ai"
        assert _msg_type(empty_dict) is None
    
    def test_get_message_content_with_base_message(self):
        """Test get_message_content with BaseMessage objects."""
        human_msg = HumanMessage(content="Hello world")
        ai_msg = AIMessage(content="Hi there!")
        
        assert get_message_content(human_msg) == "Hello world"
        assert get_message_content(ai_msg) == "Hi there!"
    
    def test_get_message_content_with_dict(self):
        """Test get_message_content with dict-shaped messages."""
        human_dict = {"type": "human", "content": "Hello world"}
        ai_dict = {"type": "ai", "content": "Hi there!"}
        empty_dict = {}
        
        assert get_message_content(human_dict) == "Hello world"
        assert get_message_content(ai_dict) == "Hi there!"
        assert get_message_content(empty_dict) == ""
    
    def test_get_last_message_content(self):
        """Test get_last_message_content."""
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Second message"),
            {"type": "human", "content": "Last message"}
        ]
        
        assert get_last_message_content(messages) == "Last message"
        assert get_last_message_content([]) == ""
    
    def test_is_human_message(self):
        """Test is_human_message."""
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi")
        human_dict = {"type": "human", "content": "Hello"}
        ai_dict = {"type": "ai", "content": "Hi"}
        
        assert is_human_message(human_msg) is True
        assert is_human_message(ai_msg) is False
        assert is_human_message(human_dict) is True
        assert is_human_message(ai_dict) is False
    
    def test_is_ai_message(self):
        """Test is_ai_message."""
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi")
        human_dict = {"type": "human", "content": "Hello"}
        ai_dict = {"type": "ai", "content": "Hi"}
        
        assert is_ai_message(human_msg) is False
        assert is_ai_message(ai_msg) is True
        assert is_ai_message(human_dict) is False
        assert is_ai_message(ai_dict) is True
    
    def test_get_last_human_message_content(self):
        """Test get_last_human_message_content."""
        messages = [
            HumanMessage(content="First human"),
            AIMessage(content="AI response"),
            HumanMessage(content="Second human"),
            AIMessage(content="Another AI response"),
        ]
        
        assert get_last_human_message_content(messages) == "Second human"
        
        # Test with mixed message types
        mixed_messages = [
            {"type": "human", "content": "Dict human"},
            AIMessage(content="AI response"),
        ]
        assert get_last_human_message_content(mixed_messages) == "Dict human"
        
        # Test with no human messages
        ai_only = [AIMessage(content="Only AI")]
        assert get_last_human_message_content(ai_only) == ""
    
    def test_get_last_ai_message_content(self):
        """Test get_last_ai_message_content."""
        messages = [
            HumanMessage(content="Human message"),
            AIMessage(content="First AI"),
            HumanMessage(content="Another human"),
            AIMessage(content="Last AI"),
        ]
        
        assert get_last_ai_message_content(messages) == "Last AI"
        
        # Test with mixed message types
        mixed_messages = [
            HumanMessage(content="Human message"),
            {"type": "ai", "content": "Dict AI"},
        ]
        assert get_last_ai_message_content(mixed_messages) == "Dict AI"
        
        # Test with no AI messages
        human_only = [HumanMessage(content="Only human")]
        assert get_last_ai_message_content(human_only) == ""
    
    def test_get_last_completed_turn_messages(self):
        """Test get_last_completed_turn_messages."""
        # Normal conversation with completed turn
        messages = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
            AIMessage(content="Answer 2"),
            HumanMessage(content="Current question"),  # In-progress
        ]
        
        result = get_last_completed_turn_messages(messages)
        assert len(result) == 2
        assert get_message_content(result[0]) == "Question 2"
        assert get_message_content(result[1]) == "Answer 2"
        
        # Test with conversation ending in AI message
        messages_end_ai = [
            HumanMessage(content="Question 1"),
            AIMessage(content="Answer 1"),
            HumanMessage(content="Question 2"),
            AIMessage(content="Answer 2"),
        ]
        
        result = get_last_completed_turn_messages(messages_end_ai)
        assert len(result) == 2
        assert get_message_content(result[0]) == "Question 2"
        assert get_message_content(result[1]) == "Answer 2"
        
        # Test with mixed message types
        mixed_messages = [
            {"type": "human", "content": "Dict question"},
            AIMessage(content="BaseMessage answer"),
            HumanMessage(content="Current question"),
        ]
        
        result = get_last_completed_turn_messages(mixed_messages)
        assert len(result) == 2
        assert get_message_content(result[0]) == "Dict question"
        assert get_message_content(result[1]) == "BaseMessage answer"
        
        # Test with insufficient messages
        short_messages = [HumanMessage(content="Only one")]
        assert get_last_completed_turn_messages(short_messages) == []
        
        # Test with no completed turns
        no_completion = [
            HumanMessage(content="Question 1"),
            HumanMessage(content="Question 2"),
        ]
        assert get_last_completed_turn_messages(no_completion) == []
    
    def test_create_replacement_message(self):
        """Test create_replacement_message."""
        result = create_replacement_message("New content")
        
        # Should return a command structure
        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "New content"
        assert result["_replace_last_ai"] is True
        
        # Test with additional kwargs
        result_with_kwargs = create_replacement_message(
            "New content", {"metadata": {"source": "test"}}
        )
        assert isinstance(result_with_kwargs, dict)
        assert result_with_kwargs["messages"][0].content == "New content"
        assert result_with_kwargs["_replace_last_ai"] is True