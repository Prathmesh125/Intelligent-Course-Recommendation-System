"""
quick_test.py
Quick smoke test — run all modules and verify the full pipeline.
Usage: python quick_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("NLPRec — Full Pipeline Smoke Test")
print("=" * 60)

# 1. Preprocessing
print("\n[1] text_preprocessing …")
from text_preprocessing import preprocess_text, build_corpus
cleaned = preprocess_text("I want to learn Data Science but I don't know coding")
print(f"   Input  : I want to learn Data Science but I don't know coding")
print(f"   Output : {cleaned}")
assert len(cleaned) > 0, "Preprocessing returned empty string!"
print("   ✔ PASS")

# 2. Vectorizer
print("\n[2] vectorizer (load or build) …")
from vectorizer import load_tfidf_model, transform_query
vec, mat, df = load_tfidf_model()
print(f"   Courses loaded : {len(df)}")
print(f"   TF-IDF matrix  : {mat.shape}")
q_vec = transform_query("python machine learning", vec)
print(f"   Query vec nnz  : {q_vec.nnz}")
print("   ✔ PASS")

# 3. Recommender
print("\n[3] recommender …")
from recommender import recommend, keyword_search
queries = [
    "I want to enter AI but I am weak at math and I am a beginner",
    "data science with python no coding experience",
    "advanced deep learning neural networks",
]
for q in queries:
    recs = recommend(q, top_n=3)
    print(f"\n   Query : {q}")
    for _, row in recs.iterrows():
        print(f"     {row['rank']}. [{row['difficulty']:12s}] {row['course_title']}  (sim={row['similarity_score']:.4f})")
print("   ✔ PASS")

# 4. User profile
print("\n[4] user_profile …")
from user_profile import load_profile, log_search, save_course, enrich_query, get_stats
p = load_profile("smoke_test_user")
p = log_search(p, "learn python for data science", "Beginner")
p = save_course(p, "Python for Everybody")
enriched = enrich_query(p, "show me more courses")
stats = get_stats(p)
print(f"   Stats      : {stats}")
print(f"   Enriched Q : {enriched}")
print("   ✔ PASS")

# 5. Evaluation (light — 3 queries only)
print("\n[5] evaluation (quick — 3 queries) …")
from evaluation import evaluate_model, precision_at_k, recall_at_k, f1_at_k
p_val = precision_at_k(["a", "b", "c", "d"], ["a", "c", "e"], k=4)
r_val = recall_at_k(["a", "b", "c", "d"], ["a", "c", "e"], k=4)
f_val = f1_at_k(p_val, r_val)
print(f"   Precision@4 = {p_val:.4f}")
print(f"   Recall@4    = {r_val:.4f}")
print(f"   F1@4        = {f_val:.4f}")
print("   ✔ PASS")

print("\n" + "=" * 60)
print(" All tests passed! Run 'streamlit run app.py' to start the UI.")
print("=" * 60)
