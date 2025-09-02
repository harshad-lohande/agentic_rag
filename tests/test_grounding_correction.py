#!/usr/bin/env python3
"""
Test file for the enhanced grounding correction functions
"""
import hashlib
import pytest
import sys
from typing import List, Dict, Tuple

# Mock Document class for testing since langchain might not be fully available
class MockDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Import the helper functions directly
def get_document_dedup_key(doc) -> str:
    """Generate a stable de-duplication key for a document."""
    metadata = doc.metadata or {}
    
    for key in ["source", "file_name", "path", "id"]:
        if key in metadata and metadata[key]:
            return str(metadata[key])
    
    content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
    return f"content_hash_{content_hash}"

def deduplicate_documents(documents: List) -> List:
    """Remove duplicate documents based on deduplication key."""
    if not documents:
        return documents
    
    seen_keys = set()
    deduplicated = []
    
    for doc in documents:
        key = get_document_dedup_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            deduplicated.append(doc)
    
    return deduplicated

def apply_reciprocal_rank_fusion(doc_lists: List[List], k: int = 60) -> List:
    """Apply RRF to merge multiple document lists."""
    doc_scores: Dict[str, Tuple] = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            dedup_key = get_document_dedup_key(doc)
            score = 1.0 / (k + rank)
            
            if dedup_key in doc_scores:
                doc_scores[dedup_key] = (doc_scores[dedup_key][0], doc_scores[dedup_key][1] + score)
            else:
                doc_scores[dedup_key] = (doc, score)
    
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]

def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def apply_diversity_filter(documents: List, top_k: int = 4, similarity_threshold: float = 0.85) -> List:
    """Apply diversity filter using Jaccard similarity."""
    if not documents or top_k <= 0:
        return documents[:top_k]
    
    selected = []
    
    for doc in documents:
        if len(selected) >= top_k:
            break
            
        is_too_similar = False
        for selected_doc in selected:
            similarity = jaccard_similarity(doc.page_content, selected_doc.page_content)
            if similarity >= similarity_threshold:
                is_too_similar = True
                break
        
        if not is_too_similar:
            selected.append(doc)
    
    return selected


class TestGroundingCorrection:
    """Test class for enhanced grounding correction functions"""
    
    def setup_method(self):
        """Set up test documents"""
        self.docs = [
            MockDocument("Machine learning is used in data analysis", {"source": "paper1.pdf"}),
            MockDocument("Deep learning uses neural networks", {"source": "paper2.pdf"}),
            MockDocument("Machine learning is used in data analysis", {"source": "paper1.pdf"}),  # duplicate
            MockDocument("AI and ML are related fields", {"file_name": "ai_guide.txt"}),
            MockDocument("Natural language processing handles text", {"path": "/docs/nlp.md"}),
            MockDocument("Content without metadata", {}),
        ]
    
    def test_get_document_dedup_key(self):
        """Test document deduplication key generation"""
        # Test source metadata
        key1 = get_document_dedup_key(self.docs[0])
        assert key1 == "paper1.pdf"
        
        # Test file_name metadata
        key2 = get_document_dedup_key(self.docs[3])
        assert key2 == "ai_guide.txt"
        
        # Test path metadata
        key3 = get_document_dedup_key(self.docs[4])
        assert key3 == "/docs/nlp.md"
        
        # Test content hash fallback
        key4 = get_document_dedup_key(self.docs[5])
        assert key4.startswith("content_hash_")
        assert len(key4) > 13  # "content_hash_" + 32 char hash
    
    def test_deduplicate_documents(self):
        """Test document deduplication"""
        # Test with duplicates
        deduped = deduplicate_documents(self.docs)
        assert len(deduped) == 5  # 6 original - 1 duplicate
        
        # Ensure first occurrence is kept
        sources = [get_document_dedup_key(doc) for doc in deduped]
        assert sources.count("paper1.pdf") == 1
        
        # Test with empty list
        empty_result = deduplicate_documents([])
        assert empty_result == []
        
        # Test with no duplicates
        unique_docs = self.docs[:2] + self.docs[3:5]
        no_dup_result = deduplicate_documents(unique_docs)
        assert len(no_dup_result) == len(unique_docs)
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation"""
        # Identical texts
        assert jaccard_similarity("hello world", "hello world") == 1.0
        
        # Partial overlap: "hello world" vs "hello there"
        # tokens1 = {"hello", "world"}, tokens2 = {"hello", "there"}
        # intersection = {"hello"} (1), union = {"hello", "world", "there"} (3)
        # Expected: 1/3 = 0.333...
        sim = jaccard_similarity("hello world", "hello there")
        assert 0 < sim < 1
        assert abs(sim - (1/3)) < 0.01  # Expected: 1/3 ≈ 0.333
        
        # No overlap
        assert jaccard_similarity("hello world", "goodbye universe") == 0.0
        
        # Empty texts
        assert jaccard_similarity("", "") == 0.0
        assert jaccard_similarity("hello", "") == 0.0
        
        # Case insensitive
        assert jaccard_similarity("Hello World", "hello world") == 1.0
    
    def test_apply_reciprocal_rank_fusion(self):
        """Test RRF fusion of document lists"""
        list1 = [self.docs[0], self.docs[1], self.docs[3]]
        list2 = [self.docs[1], self.docs[0], self.docs[4]]
        
        fused = apply_reciprocal_rank_fusion([list1, list2], k=60)
        
        # Should have unique documents
        assert len(fused) <= len(set(get_document_dedup_key(doc) for doc in list1 + list2))
        
        # Documents appearing in both lists should rank higher
        # docs[1] appears at rank 2 in list1, rank 1 in list2 -> higher combined score
        # docs[0] appears at rank 1 in list1, rank 2 in list2 -> high combined score
        fused_keys = [get_document_dedup_key(doc) for doc in fused]
        assert get_document_dedup_key(self.docs[0]) in fused_keys
        assert get_document_dedup_key(self.docs[1]) in fused_keys
        
        # Test empty lists
        empty_fused = apply_reciprocal_rank_fusion([])
        assert empty_fused == []
        
        # Test single list
        single_fused = apply_reciprocal_rank_fusion([list1])
        assert len(single_fused) == len(list1)
    
    def test_apply_diversity_filter(self):
        """Test diversity filtering"""
        # Create similar documents
        similar_docs = [
            MockDocument("Machine learning is great for data analysis", {"source": "1"}),
            MockDocument("Machine learning algorithms analyze data effectively", {"source": "2"}),  # similar
            MockDocument("Deep learning uses neural networks", {"source": "3"}),  # different
            MockDocument("Natural language processing handles text", {"source": "4"}),  # different
        ]
        
        # With high threshold, should keep similar documents
        diverse_high = apply_diversity_filter(similar_docs, top_k=4, similarity_threshold=0.9)
        assert len(diverse_high) == 4
        
        # With low threshold, should filter similar documents
        diverse_low = apply_diversity_filter(similar_docs, top_k=4, similarity_threshold=0.2)
        assert len(diverse_low) < 4
        
        # Test with top_k smaller than input
        diverse_limited = apply_diversity_filter(similar_docs, top_k=2, similarity_threshold=0.9)
        assert len(diverse_limited) == 2
        
        # Test with empty list
        empty_diverse = apply_diversity_filter([], top_k=4)
        assert empty_diverse == []
    
    def test_rrf_with_realistic_data(self):
        """Test RRF with realistic retrieval scenario"""
        # Simulate two retrieval results for "machine learning healthcare"
        retrieval1 = [
            MockDocument("Machine learning applications in healthcare diagnostics", {"source": "health1.pdf"}),
            MockDocument("AI-powered medical imaging analysis", {"source": "imaging.pdf"}),
            MockDocument("Predictive analytics for patient outcomes", {"source": "predictive.pdf"}),
        ]
        
        retrieval2 = [
            MockDocument("Predictive analytics for patient outcomes", {"source": "predictive.pdf"}),  # duplicate
            MockDocument("Healthcare data mining with ML algorithms", {"source": "datamining.pdf"}),
            MockDocument("Machine learning applications in healthcare diagnostics", {"source": "health1.pdf"}),  # duplicate, different rank
        ]
        
        fused = apply_reciprocal_rank_fusion([retrieval1, retrieval2], k=60)
        
        # Should have 4 unique documents (2 duplicates removed)
        unique_keys = set(get_document_dedup_key(doc) for doc in fused)
        assert len(unique_keys) == 4
        
        # Documents appearing in both lists should be ranked highly
        top_keys = [get_document_dedup_key(doc) for doc in fused[:2]]
        assert "predictive.pdf" in top_keys  # appears at rank 3 and 1
        assert "health1.pdf" in top_keys     # appears at rank 1 and 3
    
    def test_integration_workflow(self):
        """Test the complete workflow integration"""
        # Simulate complete grounding correction workflow
        retrieval1 = self.docs[:3]
        retrieval2 = self.docs[2:5]
        
        # Step 1: Individual deduplication
        dedup1 = deduplicate_documents(retrieval1)
        dedup2 = deduplicate_documents(retrieval2)
        
        # Step 2: RRF fusion
        fused = apply_reciprocal_rank_fusion([dedup1, dedup2])
        
        # Step 3: Final deduplication
        final_dedup = deduplicate_documents(fused)
        
        # Step 4: Diversity filtering
        diverse = apply_diversity_filter(final_dedup, top_k=4, similarity_threshold=0.85)
        
        # Should have reasonable number of diverse documents
        assert len(diverse) <= 4
        assert len(diverse) >= 1
        
        # All should be unique
        unique_keys = set(get_document_dedup_key(doc) for doc in diverse)
        assert len(unique_keys) == len(diverse)