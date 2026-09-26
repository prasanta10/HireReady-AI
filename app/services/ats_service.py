"""Heuristic ATS (Applicant Tracking System) scorer.

Grades a resume against a job description without calling an LLM, so the score
is fast, deterministic and explainable. Six weighted checks add up to 100:

    keyword match .... 40   JD keywords (frequency weighted) present in resume
    skills coverage .. 20   hard/technical skills the JD asks for
    structure ........ 15   the sections a parser expects to find
    contact .......... 10   email, phone, professional links
    impact ........... 10   action verbs and quantified achievements
    formatting ........ 5   length, bullet hygiene, no first-person voice
"""
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.services.text_utils import STOPWORDS, strip_markdown, tokenize, word_count

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
SKILL_TOKENS: Set[str] = {
    # languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "c", "c++", "c#", "ruby",
    "php", "scala", "kotlin", "swift", "r", "matlab", "bash", "shell", "sql", "nosql", "html", "css",
    # frameworks / libraries
    "react", "angular", "vue", "next.js", "node.js", "express", "django", "flask", "fastapi",
    "spring", "rails", "streamlit", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "langchain", "huggingface", "opencv",
    "jquery", "bootstrap", "tailwind", "graphql", "grpc", "rest", "restful", "api", "apis",
    # data / infra
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis", "elasticsearch", "cassandra",
    "dynamodb", "snowflake", "bigquery", "docker", "kubernetes", "k8s", "terraform", "ansible",
    "jenkins", "github", "gitlab", "git", "linux", "unix", "aws", "azure", "gcp", "cloud",
    "serverless", "lambda", "microservices", "nginx", "prometheus", "grafana", "ci/cd", "devops",
    # ai / ml
    "ml", "ai", "llm", "llms", "nlp", "rag", "embeddings", "embedding", "transformers", "faiss",
    "chromadb", "pinecone", "vector", "inference", "fine-tuning", "ollama", "openai",
    # practice
    "agile", "scrum", "kanban", "testing", "pytest", "junit", "selenium", "debugging", "profiling",
    "optimization", "scalability", "concurrency", "async", "multithreading", "caching", "algorithms",
    "analytics", "visualization", "etl", "pipeline", "pipelines", "architecture", "security",
    "authentication", "oauth", "jwt", "websocket", "websockets",
}

SKILL_PHRASES: Sequence[str] = (
    "machine learning", "deep learning", "data science", "data engineering", "data structures",
    "natural language processing", "computer vision", "reinforcement learning", "large language model",
    "retrieval augmented generation", "vector database", "vector search", "semantic search",
    "prompt engineering", "distributed systems", "system design", "object oriented",
    "functional programming", "test driven development", "unit testing", "integration testing",
    "continuous integration", "continuous deployment", "version control", "code review",
    "rest api", "rest apis", "web services", "cloud computing", "infrastructure as code",
    "message queue", "event driven", "real time", "high availability", "load balancing",
    "problem solving", "cross functional", "time complexity", "operating systems", "computer networks",
)

# Aliases are matched in both directions: hitting either side counts as a match.
ALIASES: Dict[str, Set[str]] = {
    "javascript": {"js"},
    "typescript": {"ts"},
    "kubernetes": {"k8s"},
    "postgresql": {"postgres", "psql"},
    "machine learning": {"ml"},
    "artificial intelligence": {"ai"},
    "natural language processing": {"nlp"},
    "large language model": {"llm", "llms"},
    "retrieval augmented generation": {"rag"},
    "node.js": {"node", "nodejs"},
    "next.js": {"next", "nextjs"},
    "continuous integration": {"ci", "ci/cd"},
    "continuous deployment": {"cd", "ci/cd"},
    "amazon web services": {"aws"},
    "google cloud platform": {"gcp"},
    "scikit-learn": {"sklearn"},
    "rest api": {"rest", "restful", "api"},
    "object oriented": {"oop"},
    "test driven development": {"tdd"},
}

ACTION_VERBS: Set[str] = {
    "accelerated", "achieved", "architected", "authored", "automated", "boosted", "built",
    "collaborated", "configured", "constructed", "converted", "created", "cut", "debugged",
    "decreased", "delivered", "deployed", "designed", "developed", "devised", "diagnosed",
    "directed", "documented", "drove", "eliminated", "engineered", "enhanced", "established",
    "evaluated", "executed", "expanded", "extended", "formulated", "generated", "implemented",
    "improved", "increased", "initiated", "integrated", "introduced", "launched", "led",
    "leveraged", "maintained", "managed", "migrated", "modeled", "modernized", "optimized",
    "orchestrated", "overhauled", "parallelized", "pioneered", "prototyped", "published",
    "redesigned", "reduced", "refactored", "resolved", "restructured", "scaled", "shipped",
    "simplified", "spearheaded", "standardized", "streamlined", "strengthened", "supported",
    "tested", "trained", "transformed", "translated", "tuned", "upgraded", "validated", "wrote",
}

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
LINK_RE = re.compile(r"(linkedin\.com/\S+|github\.com/\S+|https?://\S+)", re.IGNORECASE)
PERCENT_MONEY_RE = re.compile(r"(\d+\s?%|[$€£₹]\s?\d)")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\d[\d,.]*")

SECTION_PATTERNS: Dict[str, str] = {
    "experience": r"(work\s+)?experience|employment|professional\s+background|internships?",
    "education": r"education|academics?|qualifications?",
    "skills": r"(technical\s+)?skills|technologies|tech\s+stack|competenc",
    "projects": r"projects?|portfolio|open\s+source",
}


# --------------------------------------------------------------------------- #
# Keyword handling
# --------------------------------------------------------------------------- #
def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _variants(term: str) -> Set[str]:
    """All spellings that should count as a hit for this term."""
    forms = {term, _singular(term)}
    forms |= ALIASES.get(term, set())
    for canonical, aliases in ALIASES.items():
        if term in aliases:
            forms.add(canonical)
            forms |= aliases
    return {form for form in forms if form}


def _phrase_present(phrase: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(phrase).replace(r"\ ", r"[\s-]+") + r"\b", text) is not None


def extract_jd_keywords(job_description: str, limit: int = 35) -> List[Tuple[str, float]]:
    """Weighted keywords from a JD: technical terms and phrases outrank prose."""
    text = (job_description or "").lower()
    if not text.strip():
        return []

    weights: Dict[str, float] = {}

    for phrase in SKILL_PHRASES:
        if _phrase_present(phrase, text):
            occurrences = len(re.findall(re.escape(phrase), text)) or 1
            weights[phrase] = 2.5 * (1 + math.log(occurrences))

    counts = Counter(tokenize(text))
    for token, count in counts.items():
        if len(token) < 2 or token in STOPWORDS:
            continue
        # "3+", "5+", "2025" and friends are requirements noise, not keywords.
        if token[0].isdigit() and token not in SKILL_TOKENS:
            continue
        weight = 1.0 + math.log(count)
        if token in SKILL_TOKENS:
            weight *= 2.0
        # Skip tokens already covered by a matched phrase.
        if any(token in phrase.split() for phrase in weights):
            continue
        weights[token] = weight

    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return [(term, round(weight, 3)) for term, weight in ranked[:limit]]


def _resume_surface(resume_text: str) -> Tuple[str, Set[str]]:
    lowered = resume_text.lower()
    tokens = set(tokenize(lowered))
    tokens |= {_singular(token) for token in tokens}
    return lowered, tokens


def _term_in_resume(term: str, lowered: str, tokens: Set[str]) -> bool:
    for form in _variants(term):
        if " " in form:
            if _phrase_present(form, lowered):
                return True
        elif form in tokens or _singular(form) in tokens:
            return True
    return False


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _score_keywords(keywords, lowered, tokens, max_points: float = 40.0):
    if not keywords:
        return max_points * 0.5, [], [], 0.0, "No job description keywords could be extracted."

    matched, missing, earned, total = [], [], 0.0, 0.0
    for term, weight in keywords:
        total += weight
        if _term_in_resume(term, lowered, tokens):
            matched.append(term)
            earned += weight
        else:
            missing.append(term)

    coverage = earned / total if total else 0.0
    return (
        round(max_points * coverage, 2),
        matched,
        missing,
        round(coverage, 3),
        f"{len(matched)}/{len(keywords)} job description keywords present "
        f"({coverage * 100:.0f}% weighted coverage).",
    )


def _score_skills(job_description, lowered, tokens, max_points: float = 20.0):
    jd_lower = (job_description or "").lower()
    jd_tokens = set(tokenize(jd_lower))
    required = {token for token in jd_tokens if token in SKILL_TOKENS}
    required |= {phrase for phrase in SKILL_PHRASES if _phrase_present(phrase, jd_lower)}

    if not required:
        return max_points * 0.5, [], [], "No explicit technical skills detected in the job description."

    present = sorted(skill for skill in required if _term_in_resume(skill, lowered, tokens))
    absent = sorted(required - set(present))
    ratio = len(present) / len(required)
    return (
        round(max_points * ratio, 2),
        present,
        absent,
        f"{len(present)}/{len(required)} required technical skills found.",
    )


def _score_structure(resume_text: str, max_points: float = 15.0):
    weights = {"skills": 5.0, "experience": 4.0, "education": 3.0, "projects": 3.0}
    lines = [line.strip(" #*-:\t") for line in resume_text.splitlines()]
    found, earned = [], 0.0

    for section, pattern in SECTION_PATTERNS.items():
        regex = re.compile(pattern, re.IGNORECASE)
        # A heading is a short line; fall back to a body match so plain-text
        # resumes without clear headings are not punished twice.
        heading_hit = any(regex.search(line) and len(line) <= 60 for line in lines if line)
        if heading_hit:
            found.append(section)
            earned += weights[section]
        elif regex.search(resume_text):
            found.append(section)
            earned += weights[section] * 0.5

    missing = [section for section in weights if section not in found]
    detail = f"Detected sections: {', '.join(found) if found else 'none'}."
    return round(min(max_points, earned), 2), found, missing, detail


def _score_contact(resume_text: str, max_points: float = 10.0):
    has_email = bool(EMAIL_RE.search(resume_text))
    has_phone = bool(PHONE_RE.search(resume_text))
    has_link = bool(LINK_RE.search(resume_text))
    earned = (4.0 if has_email else 0.0) + (3.0 if has_phone else 0.0) + (3.0 if has_link else 0.0)
    missing = [
        label
        for label, present in (("email", has_email), ("phone", has_phone), ("profile link", has_link))
        if not present
    ]
    detail = "All contact details present." if not missing else f"Missing: {', '.join(missing)}."
    return round(min(max_points, earned), 2), missing, detail


BULLET_RE = re.compile(r"^\s*([-*+\u2022\u2023\u25AA\u25CF\u00B7\u2043]|\d+[.)])\s*\S")


def _bullets(markdown: str) -> List[str]:
    """Bullet lines, covering Markdown markers and the glyphs PDFs leave behind."""
    bullets = [
        re.sub(r"^\s*([-*+\u2022\u2023\u25AA\u25CF\u00B7\u2043]|\d+[.)])\s*", "", line).strip()
        for line in markdown.splitlines()
        if BULLET_RE.match(line)
    ]
    return [bullet for bullet in bullets if bullet]


def _has_metric(text: str) -> bool:
    """True when a bullet carries a real number, ignoring calendar years."""
    if PERCENT_MONEY_RE.search(text):
        return True
    return bool(NUMBER_RE.search(YEAR_RE.sub(" ", text)))


def _first_word(bullet: str) -> str:
    """Leading word of a bullet, ignoring markdown emphasis and punctuation."""
    match = re.search(r"[A-Za-z][A-Za-z'-]*", bullet or "")
    return match.group(0).lower() if match else ""


def _score_impact(bullets: List[str], max_points: float = 10.0):
    if not bullets:
        return 0.0, {"bullets": 0}, "No bullet points found - ATS parsers read bullets best."

    verb_led = sum(1 for bullet in bullets if _first_word(bullet) in ACTION_VERBS)
    quantified = sum(1 for bullet in bullets if _has_metric(bullet))

    volume = min(3.0, len(bullets) / 8 * 3.0)
    verbs = 4.0 * (verb_led / len(bullets))
    numbers = 3.0 * min(1.0, (quantified / len(bullets)) / 0.4)  # 40% quantified = full marks

    stats = {
        "bullets": len(bullets),
        "action_verb_bullets": verb_led,
        "quantified_bullets": quantified,
    }
    detail = (
        f"{len(bullets)} bullets, {verb_led} start with an action verb, "
        f"{quantified} contain measurable results."
    )
    return round(min(max_points, volume + verbs + numbers), 2), stats, detail


def _score_formatting(resume_text: str, bullets: List[str], max_points: float = 5.0):
    words = word_count(resume_text)
    issues: List[str] = []
    earned = 0.0

    if 350 <= words <= 1000:
        earned += 2.0
    elif 250 <= words < 350 or 1000 < words <= 1300:
        earned += 1.0
        issues.append(f"Resume length is {words} words; 400-900 reads best for one or two pages.")
    else:
        issues.append(f"Resume length is {words} words, outside the 350-1000 word sweet spot.")

    if bullets:
        long_bullets = [bullet for bullet in bullets if word_count(bullet) > 32]
        if not long_bullets:
            earned += 1.0
        else:
            issues.append(f"{len(long_bullets)} bullet(s) exceed 32 words; tighten them to one line.")
    else:
        issues.append("Use bullet points instead of paragraphs for experience and projects.")

    first_person = FIRST_PERSON.findall(resume_text)
    if not first_person:
        earned += 1.0
    else:
        issues.append(f"Remove first-person pronouns ({len(first_person)} found).")

    risky = re.findall(r"^\s*\|.*\|\s*$", resume_text, flags=re.MULTILINE)
    emoji = re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", resume_text)
    if not risky and not emoji:
        earned += 1.0
    else:
        if risky:
            issues.append("Tables can break ATS parsers; use plain sections instead.")
        if emoji:
            issues.append("Remove emoji/graphical characters - many parsers drop them.")

    detail = "No formatting risks detected." if not issues else issues[0]
    return round(min(max_points, earned), 2), issues, detail


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _verdict(score: float) -> str:
    if score >= 85:
        return "Excellent match - ready to submit."
    if score >= 70:
        return "Good match - a few tweaks will push it higher."
    if score >= 55:
        return "Fair match - close some keyword gaps before applying."
    return "Weak match - this resume needs targeted rework for this role."


def _suggestions(missing_keywords, missing_skills, missing_sections, missing_contact,
                 impact_stats, formatting_issues) -> List[str]:
    tips: List[str] = []
    if missing_skills:
        tips.append(
            "Add the required technical skills the JD names but your resume omits: "
            + ", ".join(missing_skills[:8])
            + "."
        )
    if missing_keywords:
        tips.append(
            "Work these job description keywords into your bullets naturally: "
            + ", ".join(missing_keywords[:10])
            + "."
        )
    for section in missing_sections:
        tips.append(f"Add a clearly labelled '{section.title()}' section - parsers look for that heading.")
    if missing_contact:
        tips.append("Add your " + ", ".join(missing_contact) + " to the header.")

    bullets = impact_stats.get("bullets", 0)
    if bullets and impact_stats.get("action_verb_bullets", 0) < bullets * 0.7:
        tips.append("Start every bullet with a strong action verb (Built, Optimized, Led, Shipped).")
    if bullets and impact_stats.get("quantified_bullets", 0) < max(2, bullets * 0.4):
        tips.append("Quantify more bullets - add %, latency, throughput, users or cost saved.")
    if bullets < 8:
        tips.append("Aim for at least 8 bullets across experience and projects.")

    tips.extend(formatting_issues)
    return tips[:10]


def score_resume(resume_text: str, job_description: str, is_markdown: bool = True) -> Dict:
    """Grade a resume against a job description. Returns a JSON-ready report."""
    raw = resume_text or ""
    plain = strip_markdown(raw) if is_markdown else raw
    lowered, tokens = _resume_surface(plain)
    bullets = _bullets(raw) or _bullets(plain)

    keywords = extract_jd_keywords(job_description)
    kw_score, matched_kw, missing_kw, coverage, kw_detail = _score_keywords(keywords, lowered, tokens)
    skill_score, matched_skills, missing_skills, skill_detail = _score_skills(job_description, lowered, tokens)
    structure_score, found_sections, missing_sections, structure_detail = _score_structure(plain)
    contact_score, missing_contact, contact_detail = _score_contact(plain)
    impact_score, impact_stats, impact_detail = _score_impact(bullets)
    format_score, format_issues, format_detail = _score_formatting(plain, bullets)

    total = round(
        kw_score + skill_score + structure_score + contact_score + impact_score + format_score, 1
    )

    return {
        "score": total,
        "grade": _grade(total),
        "verdict": _verdict(total),
        "breakdown": [
            {"name": "Keyword Match", "score": kw_score, "max": 40, "detail": kw_detail},
            {"name": "Skills Coverage", "score": skill_score, "max": 20, "detail": skill_detail},
            {"name": "Section Structure", "score": structure_score, "max": 15, "detail": structure_detail},
            {"name": "Contact Info", "score": contact_score, "max": 10, "detail": contact_detail},
            {"name": "Impact & Metrics", "score": impact_score, "max": 10, "detail": impact_detail},
            {"name": "ATS Formatting", "score": format_score, "max": 5, "detail": format_detail},
        ],
        "keyword_coverage": coverage,
        "matched_keywords": matched_kw,
        "missing_keywords": missing_kw,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "word_count": word_count(plain),
        "bullet_stats": impact_stats,
        "suggestions": _suggestions(
            missing_kw, missing_skills, missing_sections, missing_contact, impact_stats, format_issues
        ),
    }


def compare_scores(before: Dict, after: Dict) -> Dict:
    """Delta between the original and tailored resume reports."""
    delta = round(after.get("score", 0) - before.get("score", 0), 1)
    gained = [
        keyword
        for keyword in after.get("matched_keywords", [])
        if keyword not in set(before.get("matched_keywords", []))
    ]
    return {
        "delta": delta,
        "improved": delta > 0,
        "keywords_gained": gained,
        "grade_before": before.get("grade"),
        "grade_after": after.get("grade"),
    }
