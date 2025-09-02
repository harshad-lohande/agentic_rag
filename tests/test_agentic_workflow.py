# tests/test_agentic_workflow.py

from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Mock settings before importing the module
with patch('agentic_rag.config.Settings') as mock_settings_class:
    mock_settings = Mock()
    mock_settings.CROSS_ENCODER_MODEL_LARGE = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    mock_settings_class.return_value = mock_settings
    
    from agentic_rag.app.agentic_workflow import smart_retrieval_and_rerank


class TestAgenticWorkflow:
    @patch('agentic_rag.app.agentic_workflow.create_retriever')
    @patch('agentic_rag.app.agentic_workflow.HuggingFaceCrossEncoder')
    @patch('agentic_rag.app.agentic_workflow.get_llm')
    @patch('agentic_rag.app.agentic_workflow.ChatPromptTemplate')
    def test_smart_retrieval_and_rerank_with_transformed_query(
        self, mock_prompt_template, mock_get_llm, mock_cross_encoder, mock_create_retriever
    ):
        """Test smart_retrieval_and_rerank when transformed_query is available."""
        # Mock retriever and client
        mock_client = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        mock_create_retriever.return_value = (mock_retriever, mock_client)
        
        # Mock cross-encoder
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.score.return_value = [0.9, 0.7]
        mock_cross_encoder.return_value = mock_cross_encoder_instance
        
        # Create state with transformed query
        state = {
            "transformed_query": "What is the test query?",
            "messages": [
                HumanMessage(content="What is the test?"),
                AIMessage(content="This is a response")
            ],
            "initial_documents": []  # Should not be used in new implementation
        }
        
        # Call the function
        result = smart_retrieval_and_rerank(state)
        
        # Assertions
        assert "documents" in result
        assert "retrieval_success" in result
        assert "is_web_search" in result
        assert result["retrieval_success"] is True
        assert result["is_web_search"] is False
        assert len(result["documents"]) == 2
        
        # Verify that retriever was called with the transformed query
        mock_retriever.invoke.assert_called_once_with("What is the test query?")
        
        # Verify client was closed
        mock_client.close.assert_called_once()
        
        # Verify that inline transformation was not used
        mock_get_llm.assert_not_called()

    @patch('agentic_rag.app.agentic_workflow.create_retriever')
    @patch('agentic_rag.app.agentic_workflow.HuggingFaceCrossEncoder')
    @patch('agentic_rag.app.agentic_workflow.get_llm')
    @patch('agentic_rag.app.agentic_workflow.ChatPromptTemplate')
    @patch('agentic_rag.app.agentic_workflow.format_messages_for_llm')
    def test_smart_retrieval_and_rerank_with_inline_transformation(
        self, mock_format_messages, mock_prompt_template, mock_get_llm, 
        mock_cross_encoder, mock_create_retriever
    ):
        """Test smart_retrieval_and_rerank when inline transformation is needed."""
        # Mock retriever and client
        mock_client = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Test document", metadata={"source": "test"})
        ]
        mock_create_retriever.return_value = (mock_retriever, mock_client)
        
        # Mock cross-encoder
        mock_cross_encoder_instance = Mock()
        mock_cross_encoder_instance.score.return_value = [0.8]
        mock_cross_encoder.return_value = mock_cross_encoder_instance
        
        # Mock LLM chain for inline transformation
        mock_llm = Mock()
        mock_result = Mock()
        mock_result.content = "Transformed inline query"
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        mock_get_llm.return_value = mock_llm
        mock_prompt_template.from_template.return_value.__or__ = Mock(return_value=mock_chain)
        
        mock_format_messages.return_value = "User: What is test?"
        
        # Create state without transformed query
        state = {
            "messages": [
                HumanMessage(content="What is test?"),
                AIMessage(content="This is a response")
            ]
        }
        
        # Call the function
        result = smart_retrieval_and_rerank(state)
        
        # Assertions
        assert result["retrieval_success"] is True
        assert result["is_web_search"] is False
        assert len(result["documents"]) == 1
        
        # Verify inline transformation was used
        mock_get_llm.assert_called_once_with(fast_model=True)
        mock_format_messages.assert_called_once()
        
        # Verify client was closed
        mock_client.close.assert_called_once()

    @patch('agentic_rag.app.agentic_workflow.create_retriever')
    def test_smart_retrieval_and_rerank_no_documents_found(self, mock_create_retriever):
        """Test smart_retrieval_and_rerank when no documents are found."""
        # Mock retriever to return empty list
        mock_client = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_create_retriever.return_value = (mock_retriever, mock_client)
        
        # Create state
        state = {
            "transformed_query": "Test query",
            "messages": [HumanMessage(content="Test")]
        }
        
        # Call the function
        result = smart_retrieval_and_rerank(state)
        
        # Assertions
        assert result["documents"] == []
        assert result["retrieval_success"] is False
        assert result["is_web_search"] is False
        
        # Verify client was closed
        mock_client.close.assert_called_once()