import json
from sentence_transformers import SentenceTransformer
import scipy.spatial

def result(model_paper, corpus, pretrained_file):
	embedder = SentenceTransformer(pretrained_file)
	corpus_embeddings = embedder.encode(corpus)

	queries = []
	for key in model_paper:
		queries.append(model_paper[key]['a'])
	query_embeddings = embedder.encode(queries)

	summaries = []

	max_prob = 0
	max_query = ''
	sentence = ''
	
	i = 0
	for query, query_embedding in zip(queries, query_embeddings):
		i += 1
		distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
	
		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])

		summary = {
			'model_answer' : query,
			'student_answer' : corpus[results[0][0]],
			# 'similarity_score' : 1 - results[0][1],
			'marks' : model_paper[str(i)]['m'],
			'marks_given' : (1 - results[0][1]) * model_paper[str(i)]['m']
		}
		summaries.append(summary)
	return summaries
