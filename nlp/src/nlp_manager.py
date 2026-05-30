"""Manages the NLP model.

PATCHED for local A/B testing. Two improvements behind env flags:

  READER_VARIANT = current | trained | deberta   (default: current)
  PER_PASSAGE    = 0 | 1                          (default: 0)

Defaults reproduce the exact 0.818 baseline. Set flags before running the
harness, e.g. in PowerShell:

  $env:READER_VARIANT="trained"; $env:PER_PASSAGE="1"
  python eval_harness.py --data nlp.jsonl --documents documents --semantic-model ...

A/B matrix (record the COMP column each run):
  current / 0   -> baseline (~0.818)
  trained / 0   -> head fix alone
  current / 1   -> per-passage alone
  trained / 1   -> both
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

# ---- reader: DeBERTa-v3-base-squad2 (trained head) ----------------------
# Baked into /workspace/models in Docker; ../models locally. Update <HASH> to
# match the snapshot folder name you copied in.
READER_MODEL = str(_MODELS / "deberta-v3-squad2/models--deepset--deberta-v3-base-squad2/snapshots/eea39c60cc305c2e4a9504f5ff117294bebb42db")
_TRUST_REMOTE = False

# decoding config — plain argmax won the local A/B; layers added nothing.
_PER_PASSAGE = False
_ANSWER_CLEANUP = False
_MAX_SPAN = 0
# -------------------------------------------------------------------------

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
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(
            READER_MODEL, trust_remote_code=_TRUST_REMOTE
        )
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(
            READER_MODEL, trust_remote_code=_TRUST_REMOTE
        ).to(device)
        self.reader_model.eval()
        self.reader_device = device
        self.rrf = None
        self.all_chunks = []
        print("[reader] DeBERTa-v3-base-squad2 (local)")

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

    def _get_context(self, question, rrf_top_k=20, rerank_top_k=5, score_threshold=-1.0):
        rrf_results = self.rrf.search(question, top_k=rrf_top_k)
        pairs = [(question, r["text"]) for r in rrf_results]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, rrf_results), key=lambda x: x[0], reverse=True)
        top = ranked[:rerank_top_k]
        filtered = [r for s, r in top if s > score_threshold]
        return filtered if filtered else [ranked[0][1]]

    # ---- answer cleanup -----------------------------------------------------
    # When ANSWER_CLEANUP=0: EXACT 0.818 logic (do not change).
    # When ANSWER_CLEANUP=1: additionally strip markdown/structure noise that
    # the harness showed is polluting otherwise-correct spans (leading ---,
    # ###, **, table pipes, section labels). Conservative: only edge-strips
    # structural tokens and removes inline ** markers; never merges words.
    @staticmethod
    def _clean_answer(answer):
        answer = answer.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "")
        answer = answer.lstrip("?").lstrip(".").strip()
        answer = " ".join(answer.split()).strip()
        if not _ANSWER_CLEANUP:
            return answer

        import re as _re
        # remove inline bold/italic markers (content kept)
        a = answer.replace("**", "").replace("__", "")
        # strip leading structural noise repeatedly: ---, ###, ##, #, |, bullets,
        # and stray leading punctuation left behind
        lead = _re.compile(r'^\s*(?:-{2,}|#{1,6}|\|+|[*\u2022]+|[:;,.\-\u2014\u2013]+)\s*')
        prev = None
        while prev != a:
            prev = a
            a = lead.sub("", a).strip()
        # strip trailing structural noise similarly
        trail = _re.compile(r'\s*(?:-{2,}|#{1,6}|\|+)\s*$')
        prev = None
        while prev != a:
            prev = a
            a = trail.sub("", a).strip()
        # collapse whitespace again and drop a dangling trailing markdown header
        a = " ".join(a.split()).strip()
        return a if a else answer

    @staticmethod
    def _best_span(start_logits, end_logits, max_span):
        """Proper SQuAD-style span decode: among (start, end) pairs with
        0 <= end-start < max_span, return the pair maximizing start+end logit.
        Far better than naive argmax(start)->argmax(end), which produces
        sprawling spans that bury the answer in surrounding text."""
        import torch as _t
        n = start_logits.shape[0]
        # take top-k starts and ends to keep it cheap
        k = min(20, n)
        top_start = _t.topk(start_logits, k).indices.tolist()
        top_end = _t.topk(end_logits, k).indices.tolist()
        best = (float("-inf"), 0, 0)
        for s in top_start:
            for e in top_end:
                if e < s or (e - s) >= max_span:
                    continue
                score = float(start_logits[s] + end_logits[e])
                if score > best[0]:
                    best = (score, s, e)
        return best[1], best[2], best[0]

    def _read_span(self, question, context):
        """Single read over a context string. Naive argmax when _MAX_SPAN==0
        (baseline 0.818 logic), bounded best-span when _MAX_SPAN>0."""
        inputs = self.reader_tokenizer(
            question, context,
            return_tensors="pt", truncation=True, max_length=512
        ).to(self.reader_device)
        with torch.no_grad():
            outputs = self.reader_model(**inputs)
        if _MAX_SPAN > 0:
            s, e, _ = self._best_span(outputs.start_logits[0], outputs.end_logits[0], _MAX_SPAN)
            end = e + 1
            start = s
        else:
            start = int(outputs.start_logits.argmax())
            end = int(outputs.end_logits.argmax()) + 1
        tokens = self.reader_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        answer = self.reader_tokenizer.convert_tokens_to_string(tokens)
        return self._clean_answer(answer)

    def _read_per_passage(self, question, retrieved):
        """Read each reranked chunk independently; keep the highest-scoring span.
        Returns (answer, source_doc_id)."""
        best_answer, best_score, best_source = "", float("-inf"), None
        for r in retrieved:
            inputs = self.reader_tokenizer(
                question, r["text"],
                return_tensors="pt", truncation=True, max_length=512
            ).to(self.reader_device)
            with torch.no_grad():
                outputs = self.reader_model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            if _MAX_SPAN > 0:
                start, end, score = self._best_span(start_logits, end_logits, _MAX_SPAN)
            else:
                start = int(start_logits.argmax())
                end_window = end_logits.clone()
                end_window[:start] = torch.finfo(end_window.dtype).min
                end = int(end_window.argmax())
                score = float(start_logits[start] + end_logits[end])
            tokens = self.reader_tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][start:end + 1]
            )
            answer = self._clean_answer(
                self.reader_tokenizer.convert_tokens_to_string(tokens)
            )
            if answer and score > best_score:
                best_answer, best_score, best_source = answer, score, r["source"]
        return best_answer, best_source

    def qa(self, question: str) -> dict[str, list[str] | str]:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[dict[str, list[str] | str]]:
        results = []
        for question in questions:
            retrieved = self._get_context(question)
            doc_ids = list(dict.fromkeys(r["source"] for r in retrieved))[:3]
            try:
                if _PER_PASSAGE:
                    answer, best_source = self._read_per_passage(question, retrieved)
                    if best_source:
                        doc_ids = [best_source] + [d for d in doc_ids if d != best_source]
                        doc_ids = doc_ids[:3]
                else:
                    context = " ".join([r["text"] for r in retrieved])
                    answer = self._read_span(question, context)
                if not answer.strip():
                    answer = retrieved[0]["text"][:200] if retrieved else ""
            except Exception:
                answer = retrieved[0]["text"][:200] if retrieved else ""
            results.append({"documents": doc_ids, "answer": answer})
        return results
