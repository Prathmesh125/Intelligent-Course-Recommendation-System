"""Test the new fallback mechanism"""
import pandas as pd
from recommender import keyword_search
from query_engine import understand_query

# Simulate what happens when live search fails
def test_fallback(query_text):
    print(f"\n{'='*70}")
    print(f"Testing fallback for: '{query_text}'")
    print('='*70)
    
    # Step 1: Extract topic using query understanding
    query_info = understand_query(query_text)
    search_topic = query_info.get("topic", query_text)
    print(f"Extracted topic: '{search_topic}'")
    
    # Step 2: Try keyword search
    local_results = keyword_search(search_topic, top_n=5)
    print(f"Keyword search results: {len(local_results)} courses")
    
    # Step 3: If empty, try individual words
    if local_results.empty and len(search_topic.split()) > 1:
        words = [w for w in search_topic.split() if len(w) > 3]
        print(f"Trying fallback with first word: '{words[0] if words else 'none'}'")
        if words:
            local_results = keyword_search(words[0], top_n=5)
            print(f"Single-word search results: {len(local_results)} courses")
    
    # Step 4: Show results
    if not local_results.empty:
        print(f"\n✓ SUCCESS - Found {len(local_results)} courses:")
        for i, row in local_results.head(3).iterrows():
            print(f"  {i+1}. {row['course_title'][:65]}")
    else:
        print("\n✗ FAILED - No courses found")
    
    return local_results

# Test cases
test_queries = [
    "i want to learn web devlopment",  # User's actual query (with typo)
    "python programming",
    "machine learning",
    "cloud computing",
    "chemistry basics",
]

for q in test_queries:
    result = test_fallback(q)
