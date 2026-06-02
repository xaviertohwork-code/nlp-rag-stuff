"""NLP manager: proven retrieval stack + distilled Qwen-0.5B generative reader.

Retrieval is UNCHANGED from the 0.818 config (BM25 + dense + RRF + gte reranker,
rerank top-5) — it has 0.982 recall and the reranker is essential for chunk
quality (dropping it -> 0.477). Only the READER is swapped: instead of the
extractive ModernBERT head, we generate the answer with the distilled student,
batched across all questions in the request for speed.

Student loads with plain transformers (no unsloth needed at inference).
"""
import math, re, os
import nltk
import tiktoken
import torch
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

import pathlib
_SRC = pathlib.Path(__file__).parent
_MODELS = pathlib.Path("/workspace/models") if pathlib.Path("/workspace/models").exists() else _SRC.parent / "models"

os.environ["TIKTOKEN_CACHE_DIR"] = str(_MODELS / "tiktoken")
nltk.data.path.insert(0, str(_MODELS / "nltk"))

EMBEDDING_MODEL = str(_MODELS / "bge-small/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
RERANKER_MODEL  = str(_MODELS / "gte-reranker/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c")


def _find_student():
    """Auto-detect the merged student model dir under models/."""
    for cand in ["student_merged", "student"]:
        p = _MODELS / cand
        if p.exists():
            # could be the dir itself or a HF cache layout
            if (p / "config.json").exists():
                return str(p)
            snaps = list(p.glob("snapshots/*"))
            if snaps:
                return str(snaps[0])
    return str(_MODELS / "student_merged")


READER_MODEL = _find_student()
_MAX_NEW = int(os.environ.get("MAX_NEW", "48"))

PROMPT_TMPL = (
    "Answer the question using only the context. Reply with the answer only, "
    "as briefly as possible, including any units or qualifiers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

enc = tiktoken.get_encoding("cl100k_base")


def semantic_chunk(text, max_tokens=200, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks, cur, cnt = [], [], 0
    for s in sentences:
        st = len(enc.encode(s))
        if cnt + st > max_tokens and cur:
            chunks.append(" ".join(cur)); cur = cur[-overlap_sentences:]
            cnt = sum(len(enc.encode(x)) for x in cur)
        cur.append(s); cnt += st
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks; self.k1 = k1; self.b = b; self.N = len(chunks)
        self.tokenized = [c["text"].lower().split() for c in chunks]
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N
        self.df = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query, top_k=50):
        qt = query.lower().split(); scores = []
        for doc_tokens in self.tokenized:
            dl = len(doc_tokens); tf = Counter(doc_tokens); score = 0.0
            for term in qt:
                if term not in self.df: continue
                idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
                tn = (tf[term] * (self.k1 + 1)) / (tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                score += idf * tn
            scores.append(score)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                 "score": scores[i], "text": self.chunks[i]["text"]} for i in top]


class DenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self.embeddings = self.model.encode([c["text"] for c in chunks], batch_size=128,
                                            show_progress_bar=False, normalize_embeddings=True,
                                            convert_to_numpy=True)

    def search(self, query, top_k=50):
        qe = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ qe
        top = np.argsort(scores)[::-1][:top_k]
        return [{"source": self.chunks[i]["source"], "chunk_id": self.chunks[i]["chunk_id"],
                 "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in top]


class RRF:
    def __init__(self, bm25, dense, k=60):
        self.bm25 = bm25; self.dense = dense; self.k = k

    def search(self, query, top_k=20):
        bm = self.bm25.search(query, top_k=50); dn = self.dense.search(query, top_k=50)
        sc = {}; cm = {}
        for rank, r in enumerate(bm):
            key = (r["source"], r["chunk_id"]); sc[key] = sc.get(key, 0) + 1/(self.k+rank+1); cm[key] = r
        for rank, r in enumerate(dn):
            key = (r["source"], r["chunk_id"]); sc[key] = sc.get(key, 0) + 1/(self.k+rank+1); cm[key] = r
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"source": k[0], "chunk_id": k[1], "score": v, "text": cm[k]["text"]} for k, v in ranked]


class NLPManager:
    loaded = False

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        if self.reader_tokenizer.pad_token is None:
            self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token
        self.reader_tokenizer.padding_side = "left"
        self.reader_model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL, torch_dtype=torch.float16).to(device).eval()
        self.rrf = None; self.all_chunks = []
        print(f"[reader] distilled Qwen-0.5B generative, max_new={_MAX_NEW}")

    def load_corpus(self, documents):
        all_chunks = []
        for doc in documents:
            for i, ct in enumerate(semantic_chunk(doc["document"])):
                all_chunks.append({"source": doc["id"], "chunk_id": i, "text": ct})
        self.rrf = RRF(BM25(all_chunks), DenseRetriever(all_chunks))
        self.all_chunks = all_chunks; self.loaded = True

    def _get_context(self, question, rrf_top_k=20, rerank_top_k=5, score_threshold=-1.0):
        rrf_results = self.rrf.search(question, top_k=rrf_top_k)
        pairs = [(question, r["text"]) for r in rrf_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, rrf_results), key=lambda x: x[0], reverse=True)
        top = ranked[:rerank_top_k]
        filtered = [r for s, r in top if s > score_threshold]
        return filtered if filtered else [ranked[0][1]]

    def qa(self, question):
        return self.qa_batch([question])[0]

    def qa_batch(self, questions):
        # 1. retrieve context per question (unchanged proven stack)
        contexts, doc_id_lists = [], []
        for q in questions:
            retrieved = self._get_context(q)
            doc_id_lists.append(list(dict.fromkeys(r["source"] for r in retrieved))[:3])
            contexts.append(" ".join(r["text"] for r in retrieved))

        # 2. build prompts (cap CONTEXT so the question + "Answer:" always survive
        # truncation -- otherwise the model only sees context and parrots it).
        CTX_CHARS = 2000
        prompts = [PROMPT_TMPL.format(context=c[:CTX_CHARS], question=q)
                   for q, c in zip(questions, contexts)]
        answers = []
        B = 16
        for i in range(0, len(prompts), B):
            chunk = prompts[i:i+B]
            enc_in = self.reader_tokenizer(chunk, return_tensors="pt", padding=True,
                                           truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                out = self.reader_model.generate(**enc_in, max_new_tokens=_MAX_NEW, do_sample=False,
                                                 pad_token_id=self.reader_tokenizer.pad_token_id)
            gen = out[:, enc_in["input_ids"].shape[1]:]
            answers.extend(self.reader_tokenizer.batch_decode(gen, skip_special_tokens=True))

        return [{"documents": d, "answer": a.strip()} for d, a in zip(doc_id_lists, answers)]
