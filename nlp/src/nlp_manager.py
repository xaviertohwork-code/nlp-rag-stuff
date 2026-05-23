"""Manages the NLP model."""
import math
import os
import nltk
import tiktoken
import torch
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

os.environ["TIKTOKEN_CACHE_DIR"] = "/workspace/models/tiktoken"
nltk.data.path.insert(0, "/workspace/models/nltk")

EMBEDDING_MODEL = "/workspace/models/bge-small/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANKER_MODEL  = "/workspace/models/gte-reranker/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c"
READER_MODEL    = "/workspace/models/modernbert-squad2/models--kiddothe2b--ModernBERT-base-squad2/snapshots/327d7b52f1023f23dc6962672f91257b2878fc88"

enc = tiktoken.get_encoding("cl100k_base")


def semantic_chunk(text, max_tokens=200, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks, current_sentences, current_token_count = [], [], 0
    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        if current_token_count + sentence_tokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = current_sentences[-overlap_sentences:]
            current_token_count = sum(len(enc.encode(s)) for s in current_sentences)
        current_sentences.append(sentence)
        current_token_count += sentence_tokens
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query, top_k=50):
        query_terms = query.lower().split()
        scores = []
        for idx, doc_tokens in enumerate(self.tokenized):
            doc_len = len(doc_tokens)
            tf = Counter(doc_tokens)
            score = 0.0
            for term in query_terms:
                if term not in self.df:
                    continue
                idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
                tf_norm = (tf[term] * (self.k1 + 1)) / (tf[term] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                score += idf * tf_norm
            scores.append(score)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                 "score": scores[i], "text": self.chunks[i]["text"]} for i in top_indices]


class DenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        texts = [c["text"] for c in chunks]
        self.embeddings = self.model.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def search(self, query, top_k=50):
        query_embedding = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                 "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in top_indices]


class RRF:
    def __init__(self, bm25, dense, k=60):
        self.bm25 = bm25
        self.dense = dense
        self.k = k

    def search(self, query, top_k=20):
        bm25_results = self.bm25.search(query, top_k=50)
        dense_results = self.dense.search(query, top_k=50)
        rrf_scores = {}
        chunk_map = {}
        for rank, r in enumerate(bm25_results):
            key = (r["source"], r["chunk_id"])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (self.k + rank + 1)
            chunk_map[key] = r
        for rank, r in enumerate(dense_results):
            key = (r["source"], r["chunk_id"])
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (self.k + rank + 1)
            chunk_map[key] = r
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"source": key[0], "chunk_id": key[1], "score": score,
                 "text": chunk_map[key]["text"]} for key, score in ranked]


class NLPManager:
    loaded = False

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cross-encoder reranker
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)

        # Extractive QA reader
        self.reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL).to(device)
        self.reader_model.eval()
        self.reader_device = device

        self.rrf = None
        self.all_chunks = []

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        all_chunks = []
        for doc in documents:
            doc_id = doc["id"]
            text = doc["document"]
            for i, chunk_text in enumerate(semantic_chunk(text)):
                all_chunks.append({"source": doc_id, "chunk_id": i, "text": chunk_text})
        bm25 = BM25(all_chunks)
        dense = DenseRetriever(all_chunks)
        self.rrf = RRF(bm25, dense)
        self.all_chunks = all_chunks
        self.loaded = True

    def _get_context(self, question, rrf_top_k=20, rerank_top_k=5, score_threshold=0.0):
        rrf_results = self.rrf.search(question, top_k=rrf_top_k)
        pairs = [(question, r["text"]) for r in rrf_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, rrf_results), key=lambda x: x[0], reverse=True)
        # Filter low-confidence chunks, fall back to top-1 if all below threshold
        top = ranked[:rerank_top_k]
        filtered = [r for s, r in top if s > score_threshold]
        return filtered if filtered else [ranked[0][1]]

    def qa(self, question: str) -> dict[str, list[str] | str]:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[dict[str, list[str] | str]]:
        results = []
        for question in questions:
            retrieved = self._get_context(question)
            doc_ids = list(dict.fromkeys(r["source"] for r in retrieved))[:3]
            context = " ".join([r["text"] for r in retrieved])
            try:
                inputs = self.reader_tokenizer(
                    question, context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.reader_device)
                with torch.no_grad():
                    outputs = self.reader_model(**inputs)
                start = outputs.start_logits.argmax()
                end = outputs.end_logits.argmax() + 1
                answer = self.reader_tokenizer.convert_tokens_to_string(
                    self.reader_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
                )
                if not answer.strip():
                    answer = retrieved[0]["text"][:200] if retrieved else ""
            except Exception:
                answer = retrieved[0]["text"][:200] if retrieved else ""
            results.append({"documents": doc_ids, "answer": answer})
        return results
