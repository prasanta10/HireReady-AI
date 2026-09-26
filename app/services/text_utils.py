"""Small, dependency-free text helpers shared by retrieval and ATS scoring."""
import re
from typing import List

STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "done", "down", "during",
    "each", "etc", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "like", "may", "me", "might", "more", "most", "must", "my", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "out", "over", "own", "per", "same",
    "shall", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "upon", "us", "use", "used", "using", "very", "via", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "within", "would", "you", "your", "yours",
    # Job-description boilerplate that carries no signal for matching.
    "ability", "able", "applicant", "applicants", "apply", "benefits", "candidate", "candidates",
    "company", "description", "employee", "employees", "employer", "equal", "experience", "good",
    "great", "help", "hiring", "job", "join", "knowledge", "looking", "must", "new", "offer",
    "opportunity", "plus", "position", "preferred", "req", "required", "requirement", "requirements",
    "responsibilities", "role", "salary", "skill", "skills", "strong", "team", "teams", "well",
    "work", "working", "years", "you'll", "we're", "nice", "familiar", "familiarity", "proficient",
    "proficiency", "expertise", "understanding", "including", "include", "includes", "hands",
    "min", "max", "etc.", "ideal", "ideally", "bonus", "want", "wants", "need", "needs",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./_-]*")
_WORD_RE = re.compile(r"\b[\w'-]+\b")


def tokenize(text: str, keep_stopwords: bool = False, min_length: int = 2) -> List[str]:
    """Lowercase word tokens, keeping tech punctuation (c++, node.js, ci/cd)."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    cleaned = []
    for token in tokens:
        token = token.strip("._-/")
        if len(token) < min_length or token.isdigit():
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markdown(markdown: str) -> str:
    """Flatten Markdown to plain prose so heuristics see the same text an ATS would."""
    if not markdown:
        return ""
    text = markdown
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)        # fenced code
    text = re.sub(r"`([^`]*)`", r"\1", text)                        # inline code
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)           # images
    text = re.sub(r"\[([^\]]*)\]\(([^)]*)\)", r"\1 \2", text)       # links -> text + url
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|.*\|\s*$", lambda m: m.group(0).replace("|", " "), text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)
    text = re.sub(r"^\s*[-=]{3,}\s*$", "", text, flags=re.MULTILINE)
    return normalize_whitespace(text)


def chunk_text(text: str, chunk_size: int, overlap: int = 0, max_chunks: int = 0) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    length = len(text)
    overlap = max(0, min(overlap, chunk_size // 2))

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            # Break on the last paragraph, sentence or space in the final third.
            for pattern in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(pattern)
                if cut > chunk_size * 0.5:
                    end = start + cut + len(pattern)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if max_chunks and len(chunks) >= max_chunks:
            break
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
