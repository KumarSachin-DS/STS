# sts_model.py
from sentence_transformers import SentenceTransformer, util

class SemanticSimilarity:
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        self.model = SentenceTransformer(model_name)

    def get_similarity_score(self, text1: str, text2: str) -> float:
        # Encode the sentences into vector embeddings
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        # Compute cosine similarity
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        # Clip to [0,1] range (cosine can be -1 to 1)
        return max(0.0, min(1.0, (score + 1) / 2))  # normalize -1..1 to 0..1


# Uncomment to run it

# if __name__ == "__main__": 
#     sts = SemanticSimilarity()
#     text1 = "Climate change is a global crisis."
#     text2 = "Global warming threatens ecosystems worldwide."
#     print("Similarity Score:", sts.get_similarity_score(text1, text2))
