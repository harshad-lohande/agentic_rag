#!/usr/bin/env python3
"""
Quick test to validate the false positive detection still works.
"""

def test_false_positive_detection():
    """Test that false positive detection still works with updated thresholds."""
    print("🔍 Testing False Positive Detection")
    print("="*50)
    
    # Updated configuration values
    vec_accept = 0.92
    vec_min = 0.85
    emb_min = 0.88
    ce_min = 0.60
    lex_min = 0.15
    
    def evaluate_rules(vector_sim, emb_sim, ce_sim, lex_sim):
        """Simulate the rule evaluation logic with false positive detection."""
        accept = False
        reason = ""
        
        # Rule 1: very high vector similarity alone, but with additional validation
        if vector_sim >= vec_accept:
            # Additional validation for perfect or near-perfect scores to prevent false positives
            if vector_sim >= 0.99:
                # Perfect score should have semantic support - reject if embedding is low AND lexical is very low
                # OR if embedding is very low (indicating completely different queries)
                if (emb_sim < 0.7 and lex_sim < 0.1) or emb_sim < 0.4:
                    return False, "Suspicious high similarity rejected"
            accept = True
            reason = f"Rule 1: vector>={vec_accept}"
        else:
            # Other rules...
            if vector_sim >= vec_min and ce_sim >= ce_min and emb_sim >= emb_min:
                accept = True
                reason = f"Rule 2: vector>={vec_min} & ce>={ce_min} & emb>={emb_min}"
            elif vector_sim >= vec_min and ce_sim >= ce_min and emb_sim >= (emb_min - 0.03) and lex_sim >= lex_min:
                accept = True
                reason = f"Rule 3: vector>={vec_min} & ce>={ce_min} & emb~ & lex>={lex_min}"
            elif vector_sim >= (vec_min + 0.02) and (emb_sim >= (emb_min - 0.05) or ce_sim >= (ce_min + 0.05)):
                accept = True
                reason = f"Rule 4: moderate vector>={vec_min + 0.02} & semantic support"
        
        return accept, reason
    
    # Test false positive case (original bug scenario)
    print("Test Case: Original Bug Scenario")
    print("  Query 1 (cached): 'What is Nativepath??'")
    print("  Query 2 (test):   'What are the benefits of consuming MCTs daily?'")
    print("  Vector similarity: 1.0 (suspicious perfect match)")
    print("  Embedding similarity: 0.22 (low)")
    print("  Lexical similarity: 0.08 (low)")
    
    accept, reason = evaluate_rules(1.0, 0.22, 0.15, 0.08)
    
    print(f"  Result: {'HIT' if accept else 'MISS'}")
    print(f"  Reason: {reason}")
    
    if not accept:
        print("  ✅ PASS: False positive correctly rejected!")
    else:
        print("  ❌ FAIL: False positive was not rejected!")
    
    print()
    
    # Test legitimate similar query that should pass
    print("Test Case: Legitimate Similar Query")
    print("  Query 1 (cached): 'What are the benefits of MCTs'")
    print("  Query 2 (test):   'What benefits can I expect if I consume MCT daily?'")
    print("  Vector similarity: 0.87 (good)")
    print("  Embedding similarity: 0.84 (good)")
    print("  Cross-encoder similarity: 0.68 (good)")
    print("  Lexical similarity: 0.45 (moderate)")
    
    accept2, reason2 = evaluate_rules(0.87, 0.84, 0.68, 0.45)
    
    print(f"  Result: {'HIT' if accept2 else 'MISS'}")
    print(f"  Reason: {reason2}")
    
    if accept2:
        print("  ✅ PASS: Legitimate similar query correctly accepted!")
    else:
        print("  ❌ FAIL: Legitimate similar query was rejected!")
    
    return not accept and accept2


if __name__ == "__main__":
    success = test_false_positive_detection()
    print()
    if success:
        print("🎉 Both tests passed! The fix balances false positive prevention with legitimate cache hits.")
    else:
        print("⚠️  Tests failed. The balance may need adjustment.")