#!/usr/bin/env python3
"""
Test script to validate the vector similarity fix.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentic_rag.app.semantic_cache import SemanticCache


def test_score_normalization():
    """Test the score normalization function with different values."""
    
    print("🧪 TESTING SCORE NORMALIZATION")
    print("=" * 50)
    
    cache = SemanticCache()
    
    # Test cases based on user's reported data
    test_cases = [
        # (raw_score, expected_description)
        (1.0, "Exact match raw score (user's data)"),
        (0.95, "Very high similarity"),
        (0.5, "Medium similarity/distance"),
        (0.1, "Low distance (high similarity)"),
        (0.0, "Perfect distance (identical vectors)"),
        (2.0, "Maximum cosine distance"),
    ]
    
    print("Testing distance mode normalization:")
    print("-" * 40)
    
    for raw_score, description in test_cases:
        normalized = cache._normalize_similarity_score(raw_score)
        print(f"Raw: {raw_score:4.1f} -> Normalized: {normalized:4.3f} ({description})")
    
    print()
    print("Expected behavior:")
    print("- Raw score 1.0 (exact match) should give high similarity (~1.0)")
    print("- Raw score 0.0 (perfect distance) should give similarity 1.0")
    print("- Raw score 2.0 (max distance) should give similarity 0.0")


if __name__ == "__main__":
    test_score_normalization()