"""Test the improved fallback with recommend() function"""
from recommender import recommend
from query_engine import understand_query

def test_improved_fallback(query_text):
    print(f"\n{'='*70}")
    print(f"Testing: '{query_text}'")
    print('='*70)
    
    # Extract topic
    query_info = understand_query(query_text)
    search_topic = query_info.get("topic", query_text)
    print(f"Extracted topic: '{search_topic}'")
    
    # Primary search with TF-IDF
    results = recommend(search_topic, top_n=5)
    
    # Filter low scores
    if not results.empty:
        results = results[results["similarity_score"] > 0.01]
    
    print(f"Results: {len(results)} courses (similarity > 0.01)")
    
    if not results.empty:
        print(f"\n✓ SUCCESS - Top courses:")
        for idx, row in results.head(5).iterrows():
            print(f"  {idx+1}. [{row['similarity_score']:.3f}] {row['course_title'][:60]}")
    else:
        print("\n✗ No relevant courses found")
        # Fallback to single keyword
        words = [w for w in search_topic.split() if len(w) > 3]
        if words:
            print(f"Trying fallback with: '{words[0]}'")
            results = recommend(words[0], top_n=5)
            if not results.empty:
                results = results[results["similarity_score"] > 0.01]
                print(f"Fallback found {len(results)} courses")
                for idx, row in results.head(3).iterrows():
                    print(f"  {idx+1}. [{row['similarity_score']:.3f}] {row['course_title'][:60]}")
    
    return results

# Test cases
test_queries = [
    "i want to learn web devlopment",
    "python programming for beginners",
    "machine learning basics",
    "react frontend development",
    "data science with python",
]

for q in test_queries:
    test_improved_fallback(q)
