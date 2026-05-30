from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import nltk, os, tiktoken

# Embedding model
SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder="models/bge-small")

# Reranker
CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", cache_folder="models/gte-reranker")

# Reader
AutoTokenizer.from_pretrained("kiddothe2b/ModernBERT-base-squad2", cache_dir="models/modernbert-squad2")
AutoModelForQuestionAnswering.from_pretrained("kiddothe2b/ModernBERT-base-squad2", cache_dir="models/modernbert-squad2")

# Reader (DeBERTa-v3 - trained QA head, replaces ModernBERT)
AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2", cache_dir="models/deberta-v3-squad2")
AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2", cache_dir="models/deberta-v3-squad2")

# NLTK and tiktoken
nltk.download("punkt", download_dir="models/nltk")
nltk.download("punkt_tab", download_dir="models/nltk")
os.environ["TIKTOKEN_CACHE_DIR"] = "models/tiktoken"
tiktoken.get_encoding("cl100k_base")

print("All models downloaded!")

