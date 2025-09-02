import pytest
from agentic_rag.app import ingestion
from agentic_rag.config import settings

def test_ingestion_pipeline(mocker, tmp_path):
    """
    Tests the full ingestion pipeline from file to vector store.
    Mocks the embedding model and the Weaviate client.
    """
    # 1. Arrange: Create a dummy data file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    p = data_dir / "test_doc.txt"
    p.write_text("This is the content of our test document for the integration test.")
    
    #
    # 2. Mock external dependencies
    #
    # Mock the settings to use our temporary data directory
    mocker.patch.object(settings, 'DATA_TO_INDEX', str(data_dir))
    
    # Mock the embedding model so we don't download it during the test
    mocker.patch('agentic_rag.app.ingestion.HuggingFaceEmbeddings')
    
    # Mock Weaviate connection and vector store addition
    mock_connect_to_local = mocker.patch('agentic_rag.app.ingestion.weaviate.connect_to_local')
    mock_weaviate_vector_store = mocker.patch('agentic_rag.app.ingestion.WeaviateVectorStore')

    # 3. Act: Run the main ingestion function
    ingestion.ingest_documents()

    # 4. Assert: Check if our mocks were called as expected
    mock_connect_to_local.assert_called_once()
    mock_weaviate_vector_store.return_value.add_documents.assert_called_once()
    
    # Optional: A more detailed assertion on what was passed to add_documents
    call_args = mock_weaviate_vector_store.return_value.add_documents.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].page_content == "This is the content of our test document for the integration test."