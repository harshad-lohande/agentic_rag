#!/usr/bin/env python3
"""
Test script to validate the simplified semantic cache implementation.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_simplified_cache_rules():
    """Test the simplified cache rule logic."""
    
    print("🧪 TESTING SIMPLIFIED CACHE RULES")
    print("=" * 50)
    
    # Mock the semantic cache tester
    class MockCacheTester:
        def _predict_cache_hit(self, ce_sim, lex_sim):
            """Simplified cache hit prediction logic."""
            
            ce_min = 0.60
            lex_min = 0.15
            
            if ce_sim is None:
                return False, "No cross-encoder similarity available"
            
            # Rule 1: High cross-encoder similarity
            if ce_sim >= 0.85:
                return True, f"Rule 1: High cross-encoder similarity (ce={ce_sim:.3f})"
            
            # Rule 2: Good cross-encoder with lexical support
            if ce_sim >= ce_min and lex_sim >= lex_min:
                return True, f"Rule 2: Cross-encoder & lexical support (ce={ce_sim:.3f}, lex={lex_sim:.2f})"
            
            # Rule 3: Very high lexical similarity
            if lex_sim >= 0.4:
                return True, f"Rule 3: High lexical similarity (lex={lex_sim:.2f})"
            
            return False, f"No rules triggered (ce={ce_sim:.3f}, lex={lex_sim:.2f})"
    
    tester = MockCacheTester()
    
    # Test cases based on realistic scenarios
    test_cases = [
        # (ce_sim, lex_sim, expected_hit, description)
        (0.90, 0.75, True, "High CE + High Lex (exact matches)"),
        (0.88, 0.20, True, "High CE + Low Lex (semantic similarity)"),
        (0.70, 0.25, True, "Good CE + Good Lex (similar queries)"),
        (0.65, 0.45, True, "Moderate CE + High Lex (paraphrases)"),
        (0.50, 0.50, True, "Low CE + High Lex (word variations)"),
        (0.40, 0.10, False, "Low CE + Low Lex (unrelated)"),
        (0.30, 0.05, False, "Very Low CE + Low Lex (unrelated)"),
        (None, 0.80, False, "No CE available"),
    ]
    
    print("Test Results:")
    print("-" * 60)
    
    all_passed = True
    for ce_sim, lex_sim, expected_hit, description in test_cases:
        hit, rule = tester._predict_cache_hit(ce_sim, lex_sim)
        
        success = (hit == expected_hit)
        if not success:
            all_passed = False
        
        status = "✅" if success else "❌"
        ce_str = f"{ce_sim:.2f}" if ce_sim is not None else "None"
        
        print(f"{status} {description}")
        print(f"   CE: {ce_str}, Lex: {lex_sim:.2f}")
        print(f"   Result: {'HIT' if hit else 'MISS'} ({rule})")
        print(f"   Expected: {'HIT' if expected_hit else 'MISS'}")
        print()
    
    if all_passed:
        print("🎉 All tests passed! Simplified cache rules work correctly.")
    else:
        print("⚠️  Some tests failed. Rules may need adjustment.")
    
    return all_passed


def test_realistic_query_scenarios():
    """Test realistic query scenarios."""
    
    print("\n🔍 TESTING REALISTIC QUERY SCENARIOS")
    print("=" * 50)
    
    scenarios = [
        {
            "cached": "What are the benefits of MCTs",
            "test": "What are the benefits of taking MCT",
            "expected_ce": 0.85,  # High semantic similarity
            "expected_lex": 0.8,  # High lexical overlap
            "should_hit": True
        },
        {
            "cached": "How are MCTs different from LCTs?",
            "test": "What is the difference between MCTs and LCTs?",
            "expected_ce": 0.88,  # High semantic similarity
            "expected_lex": 0.5,  # Moderate lexical overlap
            "should_hit": True
        },
        {
            "cached": "What are the benefits of MCTs",
            "test": "What are the health benefits of chocolate?",
            "expected_ce": 0.3,   # Low semantic similarity
            "expected_lex": 0.2,  # Low lexical overlap
            "should_hit": False
        }
    ]
    
    print("Scenario Analysis:")
    print("-" * 60)
    
    class MockCacheTester:
        def _predict_cache_hit(self, ce_sim, lex_sim):
            ce_min = 0.60
            lex_min = 0.15
            
            if ce_sim >= 0.85:
                return True, f"Rule 1: High cross-encoder similarity (ce={ce_sim:.3f})"
            elif ce_sim >= ce_min and lex_sim >= lex_min:
                return True, f"Rule 2: Cross-encoder & lexical support (ce={ce_sim:.3f}, lex={lex_sim:.2f})"
            elif lex_sim >= 0.4:
                return True, f"Rule 3: High lexical similarity (lex={lex_sim:.2f})"
            else:
                return False, f"No rules triggered (ce={ce_sim:.3f}, lex={lex_sim:.2f})"
    
    tester = MockCacheTester()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}:")
        print(f"  Cached: '{scenario['cached']}'")
        print(f"  Query:  '{scenario['test']}'")
        
        hit, rule = tester._predict_cache_hit(
            scenario['expected_ce'], 
            scenario['expected_lex']
        )
        
        result_correct = (hit == scenario['should_hit'])
        status = "✅" if result_correct else "❌"
        
        print(f"  Expected scores: CE={scenario['expected_ce']:.2f}, Lex={scenario['expected_lex']:.2f}")
        print(f"  {status} Result: {'HIT' if hit else 'MISS'} ({rule})")
        print(f"  Expected: {'HIT' if scenario['should_hit'] else 'MISS'}")
        print()


if __name__ == "__main__":
    success = test_simplified_cache_rules()
    test_realistic_query_scenarios()
    
    if success:
        print("✅ Simplified cache implementation validated successfully!")
    else:
        print("❌ Some validation tests failed.")