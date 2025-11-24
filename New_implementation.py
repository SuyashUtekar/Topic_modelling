# Complete updated pipeline - in-memory (no files written)
# Requirements:
# pip install pdfplumber nltk sentence-transformers scikit-learn hdbscan bertopic
# python -m nltk.downloader punkt averaged_perceptron_tagger

import re
import unicodedata
from collections import defaultdict, Counter
import json
import numpy as np
import pdfplumber
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
from bertopic import BERTopic

# --------------------- Config / Paths ---------------------
PDF_PATH = "/mnt/data/10MC02072019.pdf"   # <- your uploaded file path
# Page ranges to extract (0-indexed) derived from: pages 3-30,49-50,66-68,70-71
SELECTED_PAGE_INDICES = (
    list(range(2, 30)) +      # pages 3–30
    list(range(48, 50)) +     # pages 49–50
    list(range(65, 68)) +     # pages 66–68
    list(range(69, 71))       # pages 70–71
)

EMB_MODEL = "all-MiniLM-L6-v2"
PCA_DIM = 50                  # tune: 50/80/100
HDB_MIN_CLUSTER_SIZE = 8      # tune smaller -> more clusters
HDB_MIN_SAMPLES = 1
REDUCE_OUTLIER_PROB_THRESH = 0.35  # optional threshold for treating low-prob members as weak

# --------------------- Utils ---------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def is_heading_like(line: str) -> bool:
    line = line.strip()
    if len(line) < 4:
        return True
    # Title-case short lines with no end punctuation -> likely a heading
    if re.match(r"^[A-Z][A-Za-z0-9 ,:-]{0,60}$", line) and not re.search(r"[.!?]$", line):
        return True
    return False

def fix_line_breaks(raw: str) -> str:
    lines = raw.split("\n")
    processed = []
    buf = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not buf:
            buf = s
            continue
        prev_no_punct = not re.search(r"[.!?:;]$", buf)
        next_not_heading = not is_heading_like(s)
        next_starts = re.match(r"^[a-zA-Z0-9\"'`]", s) is not None
        if prev_no_punct and next_not_heading and next_starts:
            buf += " " + s
        else:
            processed.append(buf)
            buf = s
    if buf:
        processed.append(buf)
    return "\n".join(processed)

def remove_bullets_anywhere(s: str) -> str:
    # remove bullets/labels at start and in middle: a. a) (a) i. (i) • - *
    s = re.sub(r"^\s*[\•\-\*]\s*", "", s)
    s = re.sub(r"^\s*[a-zA-Z]\s*[\.\)]\s*", "", s)
    s = re.sub(r"^\s*$begin:math:text$\?\[a\-zA\-Z\]$end:math:text$\s*", "", s)
    s = re.sub(r"^\s*$begin:math:text$\?\[ivxIVX\]\+$end:math:text$?\s*", "", s)
    s = re.sub(r"\s+[a-zA-Z]\s*[\.\)]\s+", " ", s)
    s = re.sub(r"\s+$begin:math:text$\?\[a\-zA\-Z\]$end:math:text$\s+", " ", s)
    s = re.sub(r"\s+[ivxIVX]+\s*[\.\)]\s+", " ", s)
    s = re.sub(r"\s+$begin:math:text$\?\[ivxIVX\]\+$end:math:text$\s+", " ", s)
    s = s.replace("•", " ").replace("·", " ")
    return s.strip()

def remove_leading_section_numbers(s: str) -> str:
    s = re.sub(r"^$begin:math:text$\?\[ivxIVX\]\+$end:math:text$?\s*", "", s)            # (i)
    s = re.sub(r"^\d+(\.\d+){0,3}\s*", "", s)            # 1. 1.2 2.3.4
    s = re.sub(r"^[$begin:math:text$\\\[\]\?\[a\-zA\-Z0\-9\]\{1\,3\}\[$end:math:text$\]]?\s+", "", s)  # a), (1)
    return s.strip()

# --------------------- 1) Extract selected pages and raw text ---------------------
pages_text = []
with pdfplumber.open(PDF_PATH) as pdf:
    for idx in SELECTED_PAGE_INDICES:
        if idx < len(pdf.pages):
            pages_text.append(pdf.pages[idx].extract_text() or "")
raw = "\n".join(pages_text)
raw = normalize_text(raw)
raw = fix_line_breaks(raw)
print("Raw text extracted from selected pages. Length (chars):", len(raw))

# --------------------- 2) Sentence tokenize & minimal cleaning ---------------------
from nltk import download as nltk_download
nltk_download('punkt')
nltk_download('averaged_perceptron_tagger')  # for optional verb checks if needed

sentences = []
for para in raw.split("\n"):
    para = para.strip()
    if not para:
        continue
    for s in sent_tokenize(para):
        s = s.strip()
        s = remove_leading_section_numbers(s)
        s = remove_bullets_anywhere(s)
        s = re.sub(r"\s+", " ", s).strip()
        # minimal filters: keep somewhat meaningful sentences; tune thresholds as needed
        if len(s) >= 12 and len(s.split()) >= 2:
            sentences.append(s)
print("Total sentences extracted (after minimal filter):", len(sentences))

# --------------------- 3) Compute embeddings (all-MiniLM-L6-v2) ---------------------
print("Loading embedding model:", EMB_MODEL)
embed_model = SentenceTransformer(EMB_MODEL)
embeddings = embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

# --------------------- 4) PCA reduction (no whitening) ---------------------
pca = PCA(n_components=min(PCA_DIM, embeddings.shape[1]), whiten=False, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)
print("PCA done. shape:", embeddings_pca.shape, "explained_variance_ratio_sum:", pca.explained_variance_ratio_.sum())

# --------------------- 5) Normalize after PCA ---------------------
embeddings_pca_norm = normalize(embeddings_pca, norm='l2', axis=1)

# --------------------- 6) Configure CountVectorizer + c-TF-IDF (legacy import safe) ---------------------
# Use legacy ClassTfidfTransformer when newer ClassTFIDF not available
try:
    # prefer ClassTFIDF if available (newer versions)
    from bertopic.vectorizers import ClassTFIDF as MyClassTFIDF
    print("Using ClassTFIDF (newer).")
except Exception:
    from bertopic.vectorizers import ClassTfidfTransformer as MyClassTFIDF
    print("Using ClassTfidfTransformer (legacy).")

vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)
ctfidf_model = MyClassTFIDF()  # instantiate

# --------------------- 7) HDBSCAN model for BERTopic ---------------------
hdb = hdbscan.HDBSCAN(
    min_cluster_size=HDB_MIN_CLUSTER_SIZE,
    min_samples=HDB_MIN_SAMPLES,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# --------------------- 8) Create BERTopic and fit with precomputed embeddings ---------------------
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    hdbscan_model=hdb,
    umap_model=None,               # we already did PCA
    calculate_probabilities=True,
    verbose=True
)

print("Fitting BERTopic (this will cluster + create representations)...")
topics, probs = topic_model.fit_transform(documents=sentences, embeddings=embeddings_pca_norm)
print("Initial topic counts:", Counter(topics))

# --------------------- 9) Inspect outliers (-1) and membership probabilities ---------------------
initial_counts = Counter(topics)
num_outliers_initial = initial_counts.get(-1, 0)
print("Initial number of outliers (label -1):", num_outliers_initial)

# HDBSCAN probabilities are available from BERTopic's hdbscan model
# retrieve underlying clusterer from topic_model
hdbscan_model_inside = topic_model.hdbscan_model
if hasattr(hdbscan_model_inside, "probabilities_"):
    hdbscan_probs = hdbscan_model_inside.probabilities_
    # Show distribution of probabilities
    print("HDBSCAN membership probability stats: min/mean/max:", np.min(hdbscan_probs), np.mean(hdbscan_probs), np.max(hdbscan_probs))
else:
    hdbscan_probs = None
    print("No probabilities available from HDBSCAN inside BERTopic.")

# --------------------- 10) Reduce outliers with two strategies and compare ---------------------
print("\nReducing outliers using BERTopic.reduce_outliers() (text c-TF-IDF strategy)...")
new_topics_ctfidf = topic_model.reduce_outliers(documents=sentences, topics=topics, strategy="c-TF-IDF")
print("Counts after c-TF-IDF reduction:", Counter(new_topics_ctfidf))

print("\nReducing outliers using BERTopic.reduce_outliers() (embeddings strategy)...")
# embeddings strategy requires passing embeddings used for clustering
new_topics_emb = topic_model.reduce_outliers(documents=sentences, topics=topics, strategy="embeddings", embeddings=embeddings_pca_norm)
print("Counts after embeddings-based reduction:", Counter(new_topics_emb))

# --------------------- 11) Compare some reassigned examples for both strategies ---------------------
def sample_reassignments(orig_topics, reduced_topics, n=10):
    samples = []
    for i in range(len(sentences)):
        if orig_topics[i] == -1 and reduced_topics[i] != -1:
            samples.append((i, sentences[i], orig_topics[i], reduced_topics[i]))
            if len(samples) >= n:
                break
    return samples

print("\nSample reassignments (c-TF-IDF strategy):")
for i, s, o, n_t in sample_reassignments(topics, new_topics_ctfidf, n=8):
    print(f"[idx {i}] -> from {o} to {n_t} : {s}")

print("\nSample reassignments (embeddings strategy):")
for i, s, o, n_t in sample_reassignments(topics, new_topics_emb, n=8):
    print(f"[idx {i}] -> from {o} to {n_t} : {s}")

# --------------------- 12) Choose strategy: pick embeddings-based if you prefer semantic mapping ----
# You can change this choice. For now we'll use the embeddings strategy result as final.
chosen_topics = new_topics_emb
print("\nChosen outlier reduction strategy: 'embeddings' (you can change this to 'ctfidf').")
print("Counts after chosen reduction:", Counter(chosen_topics))

# --------------------- 13) Apply update_topics to finalize assignment (do this as last step) ---------------------
# WARNING: After this, do not run reduce_topics()/merge_topics() -- update_topics should be last.
topic_model.update_topics(documents=sentences, topics=chosen_topics)
# Recompute final topics & probabilities using model internals
topics_final, probs_final = topic_model.transform(sentences, embeddings=embeddings_pca_norm)
print("Final topic counts (after update_topics):", Counter(topics_final))

# --------------------- 14) Build clusters dict and contiguous chunks (in-document) ---------------------
clusters = defaultdict(list)
for i, lab in enumerate(topics_final):
    clusters[lab].append((i, sentences[i]))

# Contiguous chunks by merging adjacent sentences that share same label
contiguous_chunks = []
if len(sentences) > 0:
    cur_lab = topics_final[0]
    cur_buf = [sentences[0]]
    for i in range(1, len(sentences)):
        if topics_final[i] == cur_lab:
            cur_buf.append(sentences[i])
        else:
            contiguous_chunks.append((cur_lab, " ".join(cur_buf)))
            cur_lab = topics_final[i]
            cur_buf = [sentences[i]]
    contiguous_chunks.append((cur_lab, " ".join(cur_buf)))

# --------------------- 15) Summaries & examples ---------------------
print("\nSummary:")
print(" Total sentences:", len(sentences))
print(" Total topics (unique labels):", len(set([t for t in topics_final if t != -1])))
print(" Outliers remaining (label -1):", Counter(topics_final).get(-1, 0))

print("\nTop topic examples (up to 10 topics):")
topic_info = topic_model.get_topic_info().head(20)
print(topic_info)

print("\nShow 5 clusters samples (skip -1):")
non_noise_topics = [t for t in set(topics_final) if t != -1]
for t in list(non_noise_topics)[:5]:
    print(f"\nTopic {t} — sample size {len(clusters[t])}")
    for idx, s in clusters[t][:3]:
        print("  -", s[:200])

print("\nShow 10 contiguous chunks (first 10):")
for lab, chunk in contiguous_chunks[:10]:
    print(f"[label {lab}] len_words={len(chunk.split())} -> {chunk[:200]}...\n")

# --------------------- 16) End: in-memory outputs ---------------------
# Available in memory:
# - sentences (list[str])
# - embeddings (np.ndarray) original
# - embeddings_pca_norm (np.ndarray) used for clustering
# - topic_model (BERTopic)
# - topics_final (list[int]) final topic per sentence
# - probs_final (np.ndarray) probability matrix per document (if enabled)
# - clusters (dict: topic_id -> list[(idx, sentence)])
# - contiguous_chunks (list[(topic_id, merged_text)]) contiguous in-doc chunks
