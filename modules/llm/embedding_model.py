from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

sentences = [
    "Dziekanat jest otwarty w godzinach 8-16",
    "Plan zajec na semestr letni zostanie opublikowany w marcu",
    "Kiedy zostanie opublikowany plan zajec na semestr letni?",
    "W jakich godzinach jest otwarty dziekanat?",
]
embeddings = model.encode(sentences)
print(f"{embeddings.shape = }")

similarities = model.similarity(embeddings, embeddings)
print(f"{similarities.shape = }")
print("\n" * 3)
print(similarities)
