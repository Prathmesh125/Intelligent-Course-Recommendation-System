AI-Powered Recommendation System Using Natural Language Processing: Architecture, Personalization, and Evaluation
First Author, Second Author, Third Author
Department of Computer Science and Engineering
Institution Name, City, Country
{author1, author2, author3}@universit

 
ABSTRACT

Massive Open Online Courses (MOOCs) have emerged as a technical challenge in the form of information overload to learners due to the explosive expansion of MOOCs on platforms like Coursera, edX, MIT OpenCourseWare, and Khan Academy. It is no longer that simple to find the correct course out of hundreds of thousands of options, and the traditional keyword-matching or rating-based recommenders always lose the ability to read between the lines in a free-form natural language query such as I want to learn AI but I am terrible at math and I am a complete beginner. In this paper, NLPRec, a full-stack intelligent course recommendation system that is developed on the basis of seven principles, is presented: (1) a powerful seven-stage NLP preprocessing pipeline including selective preservation of negation; (2) sublinear TF-IDF-vectorisation with bigram-based features using a multi-source course corpus; (3) cosine similarity retrieval with a log-dampened collective engagement boost; (4) a nine-step query comprehension engine, including; (a) abbreviation expansion; (b) spell correction with domain 10-curated query validation at K = 5 gives mean Precision @ 5 = 0.72, Recall @ 5 = 0.98 and F1 @ 5 = 0.82 - 71.4, 71.9 and 70.8 improvements over the baseline of keyword-matching. The system is released as open-source Streamlit web application that provides a reference implementation of the NLP-enhanced EdTech community that can be reproduced.
Keywords course recommendation NLP TF-IDF cosine similarity EdTech information retrieval behavioral analytics query understanding Precision@K Recall@K MOOC Str

1. INTRODUCTION
The world e-learning market is projected to be roughly 250 billion and likely to surpass 842 billion by 2023 and 2030 respectively [1]. Several other platforms such as Coursera alone offer over 7,000 courses in over 200 partner universities and edX and MIT OpenCourseWare continue to multiply the number of thousands. Such disastrous growth has turned the traditional dilemma the impossibility to obtain educational material has been replaced by the potential availability of adequate material under the eventuality [2].
The traditional course recommender systems deal with the discovery in two paradigms. Collaborative filtering (CF) is a technique which involves inferences about the user interests of the users based upon the aggregate behaviour of the like-minded ones [3], but cannot apply to a new user with no history of interactions, or in other words, to each first time visitor to a platform. Content-based filtering (CBF)

relates item features to user profiles [4], but with structured metadata rather than natural rich, multi-constraint intent as learners are likely to say.
Consider the question: I am a professional that does not have a background in ML. I would prefer something practical and not very pythonic-mathematical. Preferably free.' Leverage of a keyword system reduces it to single tokens and reveals courses that mention python or machine learning, but did not include the difficulty parameter, the avoidance of maths, and the price preference in general. It is the nature of the compound, conversational intent of learning that NLPRec is trained to understand and react to. 

1.1   Motivation
NLPRec can be designed based on three gaps that can be regarded as critical:
    Gap in intent understanding Systems do not analyze signs of difficulty, constraint of negation and abbreviation of domains of free-form queries prior to retrieval.
    Cross-Platform Coverage Gap: Published recommenders are nowadays, by far, evaluated on single platform datasets, whereas training or deploying recommenders in the current multi-platform reality requirement is absent.
Cold-Start Gap: CF-systems decline with new users that constitute the bulk of the learners trying to access any of the platforms during a first time.

1.2  Contributions
The paper fits in the research in six key areas:
C1 - NLPRec Framework: Open-source, modular, full-fledged NLP recommendation pipeline, which is an integration of NLP preprocess, TF-IDF retrieval, and collective engagement amplification.
C2 - Query Understanding Engine Nine-step pipeline containing abbreviation expansion, domain safeguarded spell-correction, difficulty extraction and intent-noise stripping.
C3 -- Adaptive User Profiling: Profiles incorporating recency weights and profile short query with learnt topic preferences which are not directly rated.
C4 - Engagement-Augmented Ranking: The engagement is so that it can be popular, yet not too popular.
C5 - Evaluation Framework: IR style assessment based on fuzzy relevance selection and time monitored longitudinal measures.
C6 Live Search Integration: Advanced live search using duckduckgo utilizing on the fly TF-IDF re-ranking and 24 hour disk caching.

RELATED WORK

2.1  Collaborative Filtering
Collaborative filtering remains the biggest paradigm in the field of research on recommender systems [3, 5]. The scaleable neighbourhood method of the item-based CF by Sarwar et al. [3] presented scaled-up neighbourhood methods and the matrix factorisation method of Koren et al. [18] was more accurate when interacting a large, sparse matrix. Wan and Niu [6] use MOOCs to apply to a learner preference graph with hybrid CF and achieve considerable improvement of NDCG. However, none of CF methods allows first-time students with no history of interaction.

2.2  Content-Based Filtering
The content-based recommenders relate the description of items with the profile of users, removing cold-start issues with new items [4]. Tarus et al. [7] demonstrate that semantic modelling of knowledge rich questions through ontology improves their recollection. However, ontology construction needs significant area engineering and will not extrapolate to the heterogeneous vocabulary of free-form queries of the learner.

2.3  NLP and Neural Approaches
Text by word embeddings extended latent factor models by text [8]. Based on a survey carried out by Zhang et al. [9], NLP-enriched representation deep learning recommenders are always more effective on cold-start benchmarks compared to ID-based models. Deng et al. [10] apply BERT-based embeddings to suggest MOOCs, which are successful at a high cost regarding the amount of a GPU. TF-IDF makes use of cosine similarity [11] that provides competitive accuracy with small-size corpora which is tenth the cost of computational - a viable solution within the constraints of the EdTech implementation of NLPRec.
2.4  Positioning of NLPRec

Table 1: Comparison of NLPRec with Related Systems

System	NL Query	Cold-Start	Multi-Platform	Engagement	Live Search	Open-Source
CF-based [3]	No	No	No	No	No	Yes
Ontology CBF [7]	~	Yes	No	No	No	No
BERT-MOOC [10]	Yes	Yes	No	No	No	No
Wan & Niu [6]	No	No	No	Yes	No	No
NLPRec (ours)	Yes	Yes	Yes	Yes	Yes	Yes
 
3. SYSTEM ARCHITECTURE
NLPRec is structured as a modular eight- NLPRec is designed as an 8-phase pipeline which is modular. The individual components all offer a clean, typed API meaning that any given component, such as the TF-IDF vectoriser, can be substituted with a more powerful component without any other system component being altered at all. Figure 5 demonstrates the data flow of query-time.
 
Figure 1 -- NLPRec eight stage pipeline. Boxes are major processing stages; each processing stage is annotated with major operations. Lateral nodes indicate additional data sources of each stage.

Table 2: NLPRec System Modules and Responsibilities

Module	Phase	Responsibility
scraper.py	1	Multi-source data collection: Coursera API, edX API, MIT OCW sitemap
text_preprocessing.py	2	Seven-stage NLP pipeline for corpus and query normalisation
vectorizer.py	3	TF-IDF fitting, model serialisation (tfidf_*.pkl)
recommender.py	4	Cosine similarity retrieval with engagement augmentation and filters
user_profile.py	5	Per-user JSON profiles; recency-weighted short-query enrichment
behavior_tracker.py	6	Cross-user click/save analytics; log-dampened boost computation
evaluation.py	7	IR evaluation: P@K, R@K, F1@K; chart generation
app.py	8	Five-tab Streamlit front-end with dark-mode SaaS design
query_engine.py	–	Nine-step query understanding pipeline
live_search.py	–	DuckDuckGo real-time search, content filtering, on-the-fly re-ranking
query_suggestions.py	–	Dynamic chip suggestions via 30-topic knowledge graph
 
3.1  Data Sources
    Coursera Public API REST: in-depth information, e.g. title, description, difficulty (BEGINner / INTERmediate / ADVANCED / MIXed), institution, and skill taxonomies.
    MIT OpenCourseWare Sitemap (ocw.mit.edu/sitemap.xml): 2500+ university courses (coded by department) that are mapped to skill lists to make available free of charge.
edX Discovery API: meta-data of professional certificate, MicroMaster and single course normalised difficulty and price.
All the data are normalised to canonical form: courseid, coursetitle, description, skills, difficulty, rating, url, source and are saved in dataset/courses.csv. The creation of a timestamped backup is followed by an overwrite.

NLP PREPROCESSING PIPELINE
The largest design decision is that course document and user query are operated through identical seven stage pipeline and thereafter they are vectorised. This symmetry ensures that there is no situation where similarity scores are calculated in a non uniformly transformed representation - a common programming mistake in engineering that leads to silence but nevertheless decreases the accuracy of retrieval [11].

4.1  Pipeline Stages
Lowercasing is done where all the text is written in small letters so that the text can be insensibly versus the case.
URL Removal URL URL https?://S + www removes all the hyperlinks of the scraped descriptions.
Stripping of Punctuations and Digits - The stripping of all the punctuations is done by the following str.translate() to remove the single digits.
Whitespace Normalisation Multiple whitespaces were eliminated; perhaps trailing/leading whitespace were eliminated.
The tokenisation NLTK wordtokenize can handle contraction boundaries and punctuation boundaries.
Selectivity Stopword Removal Selectivity Selective stopword removal using NLTK English stopwords: Wkeep:
Wkeep = not, no, nor, never, when, where, what, how, me, my, i)  (Eq. 1)
Negation tokens are essential to save: much of the queries provided by learners are restrictions in the form of a negative (no math required, not too advanced). Getting rid of these tokens would be playing against the purpose that is stated.
Lemmatisation One of theNLTK WordNetLemmatizer reductions: inflection: algorithms: algorithm, studying: study.

4.2  Corpus Construction
di  = titlei descriptioni skillsi (Eq. 2)
Combination of the three fields (rather than title alone) will place the model far more signal per course document, would be a more useful recall in querying about topics that rather than title, a query was found in skills or descriptions.

TF-IDF VECTORISATION
5.1  Sublinear Term Frequency
TF(t, d)  =1 + 1 + log( count(t, d) ) = -1 when count(t, d) = 0 (Eq. 3)
TF(t, d)  =  0                        	otherwise
Sublinear dampening will not have the term saturation problem: a course with fifty mentions of Python will not be ranked fifty times significantly higher than a course with one mention of Python.

Smoothed Inverse Document Frequency 5.2  Smoothed Inverse Document Frequency.
IDF(t)  =  log( (1 + N) / (1 + df(t)) )  +  1   (Eq. 4)
N = the size of the corpus, df (t) = the number of documents containing term t. The presence of universal terms in all the documents will prevent the division by 0 and negative values of IDF during smoothing.

5.3  TF-IDF Weight
TFIDF(t, d)  =  TF(t, d)  x  IDF(t)   (Eq. 5)
Ngramrange = (1, 2) has been employed to reduce the set of vocabulary V down to V = 5,000 features. The Bigrams can help in capturing domain critical multi-word expressions: machine learning, deep learning, natural language processing, web development. The matrix M [?] SciPy CSR sparse has been stored as R(NxV).
Table 3: TF-IDF Vectoriser Hyperparameters
Parameter	Value	Rationale
max_features	5,000	Coverage vs. memory balance for course-sized corpus
ngram_range	(1, 2)	Captures key domain multi-word expressions
min_df	1	Include all terms; small curated corpus warrants no pruning
sublinear_tf	True	Prevents term saturation; stabilises ranking
smooth_idf	True	Prevents zero-division; handles unseen query terms
norm	L2	Unit-norm vectors; cosine similarity reduces to fast dot product
 
 
6. RETRIEVAL AND RANKING MODEL
6.1  Cosine Similarity
q  = preprocess(q).transformed by the vectorizer(q) (Eq. 6)
sim(q, di)  =  (q . di)  /  (   q      di   )   (Eq. 7)
As each of the vectors is L2-normalised, the operation in equation 7 becomes the dot product, which can be calculated in batches using scikit-learn.cosinesimilarity(qvec, M).flatten(qvec).

The boost in log-damped engagement will be 6.2.
boostraw(c) =0.1 wclick clicks(c) + 0.1 wsave saves(c) (Eq. 8)
boost(c) =min( ln(1 + boostraw(c)) 0.05, delta ) (Eq. 9)
where wclick = 0.015, wsave = 0.025, and delta = 0.12. There are three interesting properties of this formulation:
    Dampening: The natural logarithm produces decreasing returns - a course increment of 1,000 clicks will provide very little less than 1,000x the benefit of a 1-click course.
    Boundedness: The cap delta = 0.12 makes sure that the popularity of courses will not shift a cosine score by over 12 percent and allow legacy courses to take over query-relevant results.
Save weighting: Save (0.025) weights are rated 67% as valuable as click (0.015) as the intentional bookmarking signal is stronger that the learning intent signal.

6.3  Final Ranking Score
si  =  sim(q, di)  +  boost(di)   (Eq. 10)
Results that have si = 0.05 or less are discarded. The results left are ranked in descending order of si; where there is a tie, by course rating.

6.4  Keyword Matching Baseline
scorekw(q, d)  =  sum( 1[ t in text(d) ] )   in tokens(q) t (Eq. 11)
The baseline condition is one that involves no counting of overlaps of the tokens with no weighting, no pre-processing, and no difficulty awareness. It also accurately copies what an unsophisticated searching box provides and establishes the lower-bound against which to compare it.

7.QUERY UNDERSTANDING ENGINE
The query understanding engine is a nine step pipeline that runs prior to TF-IDF vectorisation. It was designed based on the study of actual queries in learners and most of them have abbreviations, typing errors, colloquialisms, signs of difficulty in natural language and long preambles that do not add any topical information.
Table 4: Nine-Step Query Understanding Pipeline
Stage	Operation	Description
1	Punctuation normalisation	Non-standard symbols and repeated punctuation -> normalised whitespace
2	Abbreviation/slang expansion	100+ regex rules: ml->machine learning, noob->beginner, js->javascript
3	Spell correction	pyspellchecker (edit dist=1); 150+ protected tech-vocabulary terms
4	Difficulty signal extraction	Regex detects beginner/intermediate/advanced; stored as metadata
5	Intent noise stripping	Removes: 'I want to learn', 'teach me', 'show me how to', etc.
6	Level word removal	Strips difficulty words from core topic string
7	Topic expansion	30-topic graph; ml->{scikit-learn, tensorflow, pytorch, neural networks}
8	Live query generation	Generates 3-4 enriched DuckDuckGo search strings
9	Correction display	Informational message if normalised query differs from raw input
 
7.1  Spell Correction with Domain Protection
Domain protection with spell correction is to offer greater accuracy of the spell correction that is done by the system as well as offer domain name protection.
Spellcheck(w) = qcorrected(w) w not in Vprotected, w has 3 or more words (Eq. 12)
q_corrected(w) = w            	otherwise
Other technical identifiers are also pressed into service violently by normative libraries: pytorch - portrait, sklearn - slain. NLPRec is useful in preventing this process, by adding 150+ domain words to the pyspellchecker dictionary, prior to a correction being effected.
Table 5: Example Abbreviation Expansion Rules
Input Tokens	Canonical Expansion
ml, ai, dl	machine learning, artificial intelligence, deep learning
nlp, cv, ds	natural language processing, computer vision, data science
js, ts	javascript, typescript
wanna, gonna	want to, going to
noob, newbie	beginner
lvl, xp	level, experience

 
FLEET Profiling of users and personalisation.
During the initial visit, NLPRec produces a long-term, per-user profile in the form of (JSON) and allows users to tailor themselves as they continue using the site and do not necessarily provide judged ratings. Some of the profile features include: history of searches (50 recent searches in general), courses bookmarked (50 last searches with full metadata), frequency weight of the topics, the number of difficulty, history of the clicks and session statistics such as the total time spent on the search in seconds.

9.1 Recency-Weighted Compounding of Topics.
w (t) = w(t) + 1.0, in case this is the first time we encounter this t (Eq. 13).
w(t)  = w(t)  + 0.5 in case t had been already registered.
The dictionary has only 100 articles and the least significant articles are squeezed out of the dictionary. This helps in preventing the presence of tired interests to generate endless weight in the life of a user in the platform.

8.2  Short-Query Enrichment
qenriched (profile.topics ) = qraw ( profile.topics ) (Eq.) 14) top-kp( profile.topics )
Query length [?] kp will be determined by query length: one-word query is going to be assigned two profile terms, three-word query will receive one. Prolonged queries are not improved and the user has already demonstrated a sufficient intent.

8.3 Preferred Difficulty Auto-Adaptation.
preferreddifficulty  =  argmax d in D)  difficulty counts [d].  
 (Eq. 15)
This is because as a user browse through Intermediate-level courses repeatedly, the system will gain that preference in subsequent suggestions in making suggestions to the user - implicit personalisation based on nothing but observed behaviour.

LIVE SEARCH INTEGRATION
The environment of MOOC is evolving at a high rate. A completely fixed corpus will be bound to become stale, new courses are being added day in day out and with the changing technology so is the need of the learners. NLPRec is an extension of the TF-IDF retrieval with a live search engine that is open-ended and searches in the open web to find the latest offerings.

9.1  DuckDuckGo Content Filtration and Integration.
    Listicles (10 Best python courses in 2025) - number-at-beginning rejection.
    Blog and social network pages - blacklisted domain (reddit, medium, quora, twitter, linkedin).
    Matches with no words that describe the course in title - dropped during pre-re-ranking stage.

9.2  On-the-Fly TF-IDF Re-Ranking
simlive(q, rj) = cos ( TFIDF(q), TFIDF(titler snippetr) ) (Eq. 16)
TFIDF is an on-the-fly fitted fresh vectoriser which is fitted on existing result set. It is never sued and never contaminates the constant corpus vocabulary.

9.3  24-Hour Disk Caching
cachekey  =  MD5( filtered query normalised query ) (Eq. 17)
Results of 24 hours are deposited in disk. This removes DuckDuckGo API round-trips (on a query within a session) on redundancy or similar queries creating lower latency and exposure to rate limits.

EVALUATION METHODOLOGY
10.1  Ground-Truth Test Set
A cross-sectional sample of the learner intents was drawn by hand with an attempt to make ten test queries of Mixed-ability and diversity of topics: diversity of the topics, level of difficulty, query types (natural language constraints, compound technical topics, abbreviated inputs). The expert review was conducted in order to find 2-5 course names in each query.
Table 6: Ground-Truth Evaluation Test Set (K = 5)
Q#	Query	# Relevant
Q1	python programming for beginners	4
Q2	machine learning for beginners no math	5
Q3	data science with python and statistics	4
Q4	deep learning neural networks advanced	3
Q5	web development html css javascript	5
Q6	sql database management for beginners	3
Q7	natural language processing text analysis	4
Q8	cloud computing devops docker kubernetes	4
Q9	linear algebra calculus for machine learning	3
Q10	recommendation systems collaborative filtering	2
 
10.2  Fuzzy Relevance Matching
The probability of a successful match in an individual sequence at a location r equals p (p, r) = 0.6  Jtoken(p, r)  + 0.4  SequenceMatcher(p, r) = m(p, r)  (Eq. 18)
Jtoken(A, B)  =   A [?] B  /  A [?] B    (Eq. 19)
Match threshold theta = 0.55. In the case where either of the titles is a substring of the other (length >= 12 chars) the score is increased to max(SM, Jtoken, 0.9). An effective tracking set with both parties avoids the problem of counting.

10.3  IR Metrics at K = 5
Precision@K  =   Relevant [?] Retrieved{1:K}   /  K   (Eq. 20)
Recall@K 	=   Relevant [?] Retrieved{1:K}   /   Relevant    (Eq. 21)
F1@K     	=  2  P@K  R@K  /  (P@K + R@K)   (Eq. 22)
Deltam  	=  (mNLP - mKW)  /  mKW  x  100%   (Eq. 23)

10.4  Future Metrics
NDCG =K = DCG =K / IDCG =K, DCG =sumi (2 =1)/log =i +1) (Eq. 24).
MRR  =  (1/ Q ) * sumq  1 / rank_q^first   (Eq. 25)

RESULTS AND DISCUSSION

11.1  Aggregate Results
The aggregate comparison is provided in Table 7 and Figures 1-3. NLPRec is superior to the keyword baseline on all the metrics by more than 70 percent. The most notable finding is the near-perfect Recall@5 = 0.98 which implies that NLPRec recovers practically all the relevant courses with the top five results.

Table 7: Aggregate Results — NLPRec vs. Keyword Baseline (K = 5)
Metric	Keyword Baseline	NLPRec (ours)	Delta Improvement
Precision@5	0.42	0.72	+71.4% ↑
Recall@5	0.57	0.98	+71.9% ↑
F1@5	0.48	0.82	+70.8% ↑
 
 
Figure 2 — Aggregate Precision@5, Recall@5, and F1@5 for NLPRec vs. keyword baseline. Labels above each bar show the absolute improvement (Delta%) over the baseline.

 
Figure 3 — Radar chart: NLPRec (blue fill) dominates the keyword baseline (orange) across all three metrics.

11.2  Per-Query Results
Table 8: Per-Query Evaluation Results at K = 5
Query	P@5 NLP	P@5 KW	R@5 NLP	R@5 KW	F1@5 NLP	F1@5 KW
Q1 — Python basics	0.80	0.60	1.00	0.75	0.89	0.67
Q2 — ML no math	0.80	0.40	0.80	0.40	0.80	0.40
Q3 — Data science	0.80	0.60	1.00	0.75	0.89	0.67
Q4 — Deep learning	0.60	0.40	1.00	0.67	0.75	0.50
Q5 — Web dev	1.00	0.60	1.00	0.60	1.00	0.60
Q6 — SQL beginner	0.60	0.40	1.00	0.67	0.75	0.50
Q7 — NLP analysis	0.80	0.40	1.00	0.50	0.89	0.44
Q8 — Cloud/DevOps	0.80	0.40	1.00	0.50	0.89	0.44
Q9 — Math for ML	0.60	0.20	1.00	0.33	0.75	0.25
Q10 — RecSys/CF	0.40	0.20	1.00	0.50	0.57	0.29
Mean	0.72	0.42	0.98	0.57	0.82	0.48
 
 

Figure 4 — Per-query F1@5 scores. The delta label above each pair shows the absolute gain of NLPRec over the keyword baseline. Greatest gains appear on multi-constraint queries (Q2, Q9, Q10).

 

Figure 5 — Per-query metric heatmap. Darker cells = higher scores. Left three columns = NLPRec; right three = keyword baseline. NLPRec achieves perfect Recall@5 on 9 out of 10 queries.

11.3  Analysis of Key Findings

Multi-Constraint Query Effect of Query Understanding.
The strongest improvements are observed on queries that have several constraints in them. In the case of Q2 (machine learning for beginners no math), the keyword baseline has 0.40 in all the metrics, and NLPRec has 0.80 - a double amount of performance. This has been aided by two synergistic processes: difficulty-signal extraction (Stage 4) appropriately labels the query as a beginner and the negation signal no is retained by Wkeep to allow TF-IDF to retrieve beginner courses that specifically target the no-heavy-math issue.

Near-Perfect Recall
NLPRec has a mean Recall at five = 0.98 and basically all the relevant courses appear in the top five results. The one instance of non-perfect recall (Q2: 0.80) is of an underlying course which refers to calculus, as opposed to math - a real lexical gap that TF-IDF cannot overcome without any external semantic information. This is the main driving force behind Future Work F1: dense embedding integration.

Essential Importance of the Expansion of Abbreviations.
Without the abbreviation expansion module, such queries as ml for beginners yield zero results - the token ml is not present in the TF-IDF vocabulary at all. Stage 2 expansion ml Stage 2 expansion ml Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage 2 Stage The query engine is not cosmetic: it is also a requirement of functioning performance on the abbreviated queries typical of real learner behaviour.

Engagement Boost Ablation
When the engagement boost (ENABLEENGAGEMENT_BOOST=False) is disabled, there is no statistically significant difference in P5, R5 or F15 on the fixed benchmark. This is predictible - the boost fixes relationships and learners-validated courses, which is manifested as higher levels of user satisfaction in longitudinal use as opposed to set-level recall enhancement in a fixed benchmark.

11.4  Computational Complexity
Table 9: Computational Complexity of Core Operations
Operation	Time	Space	Latency (N~5K)
TF-IDF training	O(N*V)	O(N*V)	~2 s (one-time)
Query vectorisation	O(|q|*V)	O(V)	< 1 ms
Cosine similarity batch	O(N*V)	O(N)	< 50 ms
Top-K selection	O(N log K)	O(K)	< 5 ms
Engagement boost (cached)	O(1)	O(C)	< 1 ms
Live search + re-rank	O(R*V')	O(R)	1 – 3 s
 


11.5  Limitations
Lexical Gap: TF-IDF cannot induce semantic equivalences in the case of surface tokens overlap (e.g. calculus vs, math). This would be covered by dense embeddings.
Small Ground Truth: There is no guarantee that a set of ten chosen queries carefully can be representative of the whole distribution in the underlying distribution of queries chosen by learners.
New-Course Cold Start: New courses to the corpus have boost = 0 and can be undermined in case of competition to old and clicked alternatives.
Corpus Freshness: The corpus must be scraped at frequent intervals, live search does this on a query time basis, but does not refresh the TF-IDF model which is used in the cache.
English-Only: It is an English only language that can be pre-processed, vocabulary and spell corrector.

CONCLUSION AND FUTURE WORK

12.1  Conclusion
This article presented the NLPRec, an end-to-end intelligent course recommendation system, which is conceptualized in the intuition that intent of querying a natural language as opposed to count of occurrences of keywords is the most significant element of retrieval in EdTech course discovery.
Included is a seven step NLP preprocessing pipeline, sublinear TF-IDF vehicleisation of bigram features, retrieval with cosine similarity using log-dampened engagement boost, a 9 step query understanding engine with abbreviation expansion and domain-sensitive spell correction, adaptive user profiling is recency weighted topic modelling and real time live search as on the fly re-ranking.
Currently, the empirical comparison of ten curated test queries at K = 5 demonstrates high-performance values over a keyword baseline Precision@5 +71.4: Recall@5 +71.9: F1@5 +70.8. The fact that Recall @5 = 0.98 is almost perfect is particularly significant to practice: it means that learners will receive the right courses after the first attempt and this will reduce the tension of information overloading that prompted this study.

12.2  Future Work
F1 -- Dense Embeddings: Use Sentence-BERT or E5 in place of TF-IDF to overcome the lexical barrier on the cross-terminology semantic equivalences.
F2 - Hybrid Neural-TF-IDF: Boosting the semantic recall by the precise accuracy of sparse TF-IDF by late fusion of sparse embeddings with dense embeddings.
F3: Learning Path Generation F3: Build on single course recommendation: This technique adds to sequential learning path planning which entails prerequisite modelling.
F4 - Temporal Signals: F4-Ranking Add Propinquity (publication date, recent updates) to ranking Temporal Signals to decrease the amount of stale content in results.
F5 - Multilingual Support: Preprocessing and query engine Multilingual models (mBERT, XLM-R) Multilingual query support.
F6 - Large-Scale A/B Study: NLPRec vs. keyword baseline in a randomised A/B experiment with real learners whose performance is assessed in terms of completion of tasks, retention and satisfaction.
F7 CF Integration: Overlay collaborative filtering on top of NLP retrival on users, who have a sufficient history of interaction to become a complete hybrid recommender.
F8 - NDCG and MRR: Can be generalized to rank-sensitive measures (Eq. 24-25) in which the position of the items in the ordered list is considered.

ACKNOWLEDGEMENTS
The author owes much to all the open-source communities of Scikit-learn, NLTK, Streamlit, DuckDuckGo Search (ddgs), python-docx and matplotlib, without which this infrastructure would have amounted to an enormous desert. To create the evaluation corpus, the public access to the course data API and sitemaps provided by Coursera, edX, and MIT OpenCourseWare teams allowed to do the task.

REFERENCES
[1]  Grand View Research. E-Learning Market Size, Share and Trends Analysis Report 2023-2030. Grand View Research, 2023.
R.Burke, Hybrid recommender systems: Survey and experiments, User Modeling and User-Adapted Interaction, vol. 12, no. 4, p. 331-370, 2002.
B. Sarwar, G. Karypis, J. Konstant, and J. Riedl, 'Item-based collaborative filtering recommendation algorithms,' Proc. WWW, pp. 285-295, 2001.
[4] M. J. Pazzani and D. Billsus, Content-based recommendation systems, The Adaptive Web. Berlin: Springer, 2007, pp. 325-341.
J. Bobadilla, F. Ortega, A. Hernando, and A. Gutierrez, Recommender systems survey Knowledge-Based Systems, vol. 46, pp. 109-132, 2013.
S. Wan and Z. Niu, A hybrid e-learning recommendation strategy through learner preference graph, IEEE Trans. Learning Technologies, vol. 13, no.4, p. 827-840, 2018.
J. K. Tarus, Z. Niu and G. Mustafa, Knowledge-based recommendation: Review of ontology-based recommender systems in e-learning, Artificial Intelligence Review, vol. 50, no. 1, pp. 21-48, 2018.
T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean, Distributed representations of words and phrases, in Advances in NeurIPS, p.3111-3119, 2013.
[9]  S. Zhang, L. Yao, A. Sun and Y. Tay, Deep learning based recommender system: A survey, ACM, Computing Surveys, vol. 52, no. 1, 2019.
S. Deng, F. Shen, H. Liu, and H. Xiong, Learning to ask for help: A BERT-based recommendation to online learning, in Proc. IEEE ICDM, 2020.
C. D. Manning, P. Raghavan and H. Schutze, Introduction to Information Retrieval. Cambridge University Press, 2008.
K. Sparck Jones, A statistical interpretation of term specificity, Journal of Documentation, vol. 28, no.1, pp. 11-21, 1972.
G. Salton and M. J. McGill, Introduction to Modern Information Retrieval. McGraw-Hill, 1983.
G. Adomavicius, A. Tuzhilin, Towards the next generation of recommender systems, IEEE Trans. Knowledge and Data engineering, vol 17, no 6, pp 734-749, 2005.
Relevance based language models V. Lavrenko and W. B. Croft, Proc. ACM SIGIR, 120-127 2001.
[16] P. Norvig, Natural language corpus data, O-Reilly media, 2007.
The article can be found in N. Reimers and I. Gurevych, Sentence-BERT: Sentence embeddings with Siamese BERT-networks in Proc. EMNLP, 2019.
Y. Koren, R. Bell and C. Volinsky, Matrix factorization techniques, recommender systems, IEEE Computer, vol. 42, no. 8, p.30-37, 2009.

