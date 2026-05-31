"""Manages the NLP model. 0.818 extractive config + BATCHED reranker.

The reranker is essential for chunk quality (dropping it -> 0.477 acc) but slow
when called per-question (~682ms/q on T4, unbatched). This version collects ALL
questions' candidate pairs and reranks them in ONE batched GPU call, then reads.
Same accuracy as 0.818, far better speed.
"""
import math
import re
import os
import nltk
import tiktoken
import torch
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import pathlib
_SRC = pathlib.Path(__file__).parent
_MODELS = pathlib.Path("/workspace/models") if pathlib.Path("/workspace/models").exists() else _SRC.parent / "models"

os.environ["TIKTOKEN_CACHE_DIR"] = str(_MODELS / "tiktoken")
nltk.data.path.insert(0, str(_MODELS / "nltk"))

EMBEDDING_MODEL = str(_MODELS / "bge-small/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
RERANKER_MODEL  = str(_MODELS / "gte-reranker/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c")
READER_MODEL    = str(_MODELS / "modernbert-squad2/models--kiddothe2b--ModernBERT-base-squad2/snapshots/327d7b52f1023f23dc6962672f91257b2878fc88")

# reranker batch size for the big cross-encoder call (tune for T4 VRAM)
_RERANK_BATCH = int(os.environ.get("RERANK_BATCH", "128"))

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
            texts, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True
        )

    def search(self, query, top_k=50):
        query_embedding = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                 "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in top_indices]

    def search_batch(self, queries, top_k=50):
        """Encode all queries at once, then score each. Faster than per-query."""
        q_emb = self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True,
                                  batch_size=128, show_progress_bar=False)
        out = []
        for qe in q_emb:
            scores = self.embeddings @ qe
            top = np.argsort(scores)[::-1][:top_k]
            out.append([{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                         "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in top])
        return out


class RRF:
    def __init__(self, bm25, dense, k=60):
        self.bm25 = bm25
        self.dense = dense
        self.k = k

    def _fuse(self, bm25_results, dense_results, top_k):
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

    def search(self, query, top_k=20):
        return self._fuse(self.bm25.search(query, top_k=50),
                          self.dense.search(query, top_k=50), top_k)

    def search_batch(self, queries, top_k=20):
        dense_all = self.dense.search_batch(queries, top_k=50)
        out = []
        for q, dn in zip(queries, dense_all):
            bm = self.bm25.search(q, top_k=50)
            out.append(self._fuse(bm, dn, top_k))
        return out


class NLPManager:
    loaded = False

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL).to(device)
        self.reader_model.eval()
        self.reader_device = device
        self.rrf = None
        self.all_chunks = []
        print(f"[retrieval] batched reranker, batch={_RERANK_BATCH}")

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

    def _read_one(self, question, retrieved):
        context = " ".join([r["text"] for r in retrieved])
        try:
            inputs = self.reader_tokenizer(
                question, context, return_tensors="pt", truncation=True, max_length=512
            ).to(self.reader_device)
            with torch.no_grad():
                outputs = self.reader_model(**inputs)
            start = outputs.start_logits.argmax()
            end = outputs.end_logits.argmax() + 1
            tokens = self.reader_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
            answer = self.reader_tokenizer.convert_tokens_to_string(tokens)
            answer = answer.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "")
            answer = answer.lstrip("?").lstrip(".").strip()
            answer = " ".join(answer.split()).strip()
            if not answer.strip():
                answer = retrieved[0]["text"][:200] if retrieved else ""
        except Exception:
            answer = retrieved[0]["text"][:200] if retrieved else ""
        return answer

    def qa(self, question: str) -> dict[str, list[str] | str]:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str], rrf_top_k=20, rerank_top_k=5, score_threshold=-1.0):
        # 1. retrieval for all questions (dense encoded in one batch)
        rrf_lists = self.rrf.search_batch(questions, top_k=rrf_top_k)

        # 2. collect ALL (question, chunk) pairs into one list, remember spans
        all_pairs = []
        spans = []  # (start_idx, end_idx) into all_pairs for each question
        for q, rrf_results in zip(questions, rrf_lists):
            s = len(all_pairs)
            for r in rrf_results:
                all_pairs.append((q, r["text"]))
            spans.append((s, len(all_pairs)))

        # 3. ONE batched reranker call over every pair
        if all_pairs:
            all_scores = self.reranker.predict(all_pairs, batch_size=_RERANK_BATCH)
        else:
            all_scores = []

        # 4. per question: pick top reranked chunks, then read
        results = []
        for (q, rrf_results, (s, e)) in zip(questions, rrf_lists, spans):
            scores = all_scores[s:e]
            ranked = sorted(zip(scores, rrf_results), key=lambda x: x[0], reverse=True)
            top = ranked[:rerank_top_k]
            filtered = [r for sc, r in top if sc > score_threshold]
            retrieved = filtered if filtered else ([ranked[0][1]] if ranked else [])
            doc_ids = list(dict.fromkeys(r["source"] for r in retrieved))[:3]
            answer = self._read_one(q, retrieved) if retrieved else ""
            results.append({"documents": doc_ids, "answer": answer})
        return results
