
"""
╔═══════════════════════════════════════════════════════════════╗
║            VeriFact — AI Fact-Checking Agent  v2              ║
║   Rebuilt for Accuracy: Evidence-First Pipeline               ║
╠═══════════════════════════════════════════════════════════════╣
║  Course  : Artificial Intelligence — Spring 2026              ║
║  Section : BCS-4K, FAST-NUCES Karachi                         ║
║  Team    : OM (24K-0711) · Vishal (24K-0625) · Zia (24K-0817) ║
╠═══════════════════════════════════════════════════════════════╣
║  Backend : Groq Cloud API  (OpenAI-compatible)                ║
║  Features:                                                    ║
║    • Unsupervised Learning: K-Means Clustering (scikit-learn) ║
║       - Now clusters by TOPIC diversity, keeps best per group ║
║    • Optimization: Knapsack Problem (Google OR-Tools)         ║
║       - Now values snippets by LLM relevance score, not words ║
║    • Adversarial Search: Minimax Debate Simulation            ║
║       - Now uses evidence consensus, not single-snippet picks ║
║                                                               ║
║  Accuracy Improvements over v1:                               ║
║    • Multi-pass evidence scoring (relevance × stance)         ║
║    • Source-credibility weighting                             ║
║    • Consensus detection across sources                       ║
║    • Chain-of-thought verdict reasoning                       ║
║    • Confidence calibrated to evidence volume + agreement     ║
║                                                               ║
║  Setup:                                                       ║
║    pip install openai duckduckgo-search scikit-learn ortools  ║
║    $env:GROQ_API_KEY = "gsk_..."   (Windows PowerShell)       ║
║    export GROQ_API_KEY="gsk_..."   (Mac/Linux)                ║
║                                                               ║
║  Run:    python main.py                                       ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import textwrap
import logging
from datetime import datetime
from collections import defaultdict
from urllib.parse import urlparse

from openai import OpenAI, APIError, APIStatusError
from duckduckgo_search import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from ortools.algorithms.python import knapsack_solver

MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS_QUERIES = 400
MAX_TOKENS_SCORE = 800
MAX_TOKENS_VERDICT = 2000
RESULTS_PER_QUERY = 8
NUM_QUERIES = 5
DDG_RETRIES = 3
DDG_RETRY_DELAY = 1.5
DEBUG = False

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(levelname)s — %(message)s")
log = logging.getLogger("verifact")

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
COLOUR = {
    "VERIFIED":       "\033[92m",
    "FALSE":          "\033[91m",
    "PARTIALLY TRUE": "\033[93m",
    "INCONCLUSIVE":   "\033[95m",
}
ICON = {
    "VERIFIED":       "✅",
    "FALSE":          "❌",
    "PARTIALLY TRUE": "⚠️ ",
    "INCONCLUSIVE":   "🔍",
}

CREDIBILITY_TIER = {
    "high":   ["reuters.com","apnews.com","bbc.com","bbc.co.uk","nature.com","science.org",
                "who.int","cdc.gov","nih.gov","nasa.gov","nytimes.com","theguardian.com",
                "washingtonpost.com","economist.com","ft.com","pubmed.ncbi.nlm.nih.gov",
                "snopes.com","factcheck.org","politifact.com","fullfact.org"],
    "medium": ["wikipedia.org","britannica.com","npr.org","abc.net.au","cbsnews.com",
                "nbcnews.com","time.com","forbes.com","bloomberg.com","cnbc.com"],
}

def _domain_credibility(url: str) -> float:
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        for domain in CREDIBILITY_TIER["high"]:
            if host == domain or host.endswith("." + domain):
                return 1.4
        for domain in CREDIBILITY_TIER["medium"]:
            if host == domain or host.endswith("." + domain):
                return 1.1
    except Exception:
        pass
    return 1.0

QUERY_PROMPT = f"""You are an expert research strategist for a fact-checking system.
Given a claim, generate exactly {NUM_QUERIES} diverse web-search queries to gather comprehensive evidence.

Rules:
  - Each query must target a DIFFERENT angle: direct verification, expert opinion, historical context, statistics, counter-evidence.
  - Queries must be 3–9 words each, no duplicates.
  - Do NOT include "fact-check", "debunk", or "verify" in any query text.
  - Cover both sides: queries that might confirm AND queries that might refute the claim.

Return ONLY a valid JSON object:
{{"queries": ["query one", "query two", "query three", "query four", "query five"]}}"""


RELEVANCE_SCORE_PROMPT = """You are a precise evidence evaluator for a fact-checking system.

Given a CLAIM and a list of web search SNIPPETS, evaluate each snippet on TWO dimensions:

1. RELEVANCE (0–10): How directly does this snippet address the specific claim?
   - 0 = completely unrelated
   - 5 = tangentially related
   - 10 = directly discusses the exact claim

2. STANCE (-10 to +10): What is the snippet's evidential stance on the claim being TRUE?
   - -10 = strong evidence the claim is FALSE
   - 0  = neutral / ambiguous / irrelevant
   - +10 = strong evidence the claim is TRUE

3. CREDIBILITY_NOTE: One sentence on any quality issues (satire, opinion, outdated, etc). Empty string if fine.

Return ONLY a valid JSON object. No markdown.
{
  "evaluations": [
    {"relevance": <int 0-10>, "stance": <int -10 to 10>, "credibility_note": "<string>"},
    ...
  ]
}

Evaluate ALL snippets in order. The array length must exactly match the number of snippets."""


VERDICT_PROMPT = """You are VeriFact — a rigorous, impartial AI fact-checker modeled on professional journalists.

You will receive:
  - A CLAIM to evaluate
  - SUPPORTING EVIDENCE: snippets that support the claim (with scores)
  - REFUTING EVIDENCE: snippets that refute the claim (with scores)
  - NEUTRAL EVIDENCE: snippets that are ambiguous
  - An EVIDENCE SUMMARY with weighted scores and source counts

Your task: issue the most accurate possible verdict using chain-of-thought reasoning.

VERDICT RULES (follow strictly):
  - VERIFIED:       Clear preponderance of credible, specific evidence supports the claim. Refuting evidence is weak or absent.
  - FALSE:          Clear preponderance of credible, specific evidence contradicts the claim. Supporting evidence is weak or absent.
  - PARTIALLY TRUE: Both supporting AND refuting evidence are substantive. The claim is true in some aspects, false in others.
  - INCONCLUSIVE:   Evidence is genuinely insufficient or too contradictory to form a reliable verdict. Use sparingly.

CONFIDENCE RULES:
  - HIGH:   Multiple independent, credible sources agree. Evidence is specific and unambiguous.
  - MEDIUM: Some credible sources agree, but evidence has gaps or minor contradictions.
  - LOW:    Evidence is thin, speculative, old, or from low-credibility sources.

CRITICAL: Do NOT default to INCONCLUSIVE because the topic is controversial. If the evidence clearly points one way, say so.

Return ONLY a valid JSON object with EXACTLY these keys:
{
  "reasoning": "<3-5 sentence chain-of-thought: what evidence exists, how strong it is, why you reached your verdict>",
  "verdict": "VERIFIED" | "FALSE" | "PARTIALLY TRUE" | "INCONCLUSIVE",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "summary": "<2-3 sentences for a general audience explaining the verdict clearly>",
  "key_findings": ["<specific fact 1 with source>", "<specific fact 2 with source>", "<specific fact 3 with source>"],
  "caveats": "<any important nuances, limitations of evidence, or things the user should know. Empty string if none.>"
}"""


DEBATER_PROMPT = """You are VeriFact — an AI Fact-Checking Debater.
You debate claims using real retrieved web evidence. Be concise, logical, and cite your sources.
Do NOT hallucinate facts. If the evidence contradicts your prior position, concede gracefully.
Return a JSON object: {"response": "<your argument>"}"""


def _init_client() -> OpenAI:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print(f"\n{BOLD}[ERROR]{RESET} GROQ_API_KEY is not set.")
        print("  Get a free key at: https://console.groq.com/keys")
        print("  Then run:")
        print("    Windows:    $env:GROQ_API_KEY = 'gsk_...'")
        print("    Mac/Linux:  export GROQ_API_KEY='gsk_...'\n")
        sys.exit(1)
    return OpenAI(api_key=key, base_url=GROQ_BASE_URL)

_client = _init_client()
LOOP_RETRIES = 3

def _groq_call(system_prompt: str = None, user_message: str = None,
               max_tokens: int = MAX_TOKENS_VERDICT, messages: list = None) -> str:
    temperature = 0.1
    if not messages:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
    for attempt in range(1, LOOP_RETRIES + 1):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            msg = str(exc)
            if "loop" in msg.lower() and attempt < LOOP_RETRIES:
                temperature = min(temperature + 0.2, 0.7)
            else:
                raise


def generate_queries(claim: str) -> list[str]:
    raw = _groq_call(QUERY_PROMPT, f'Claim: "{claim}"', MAX_TOKENS_QUERIES)
    data = json.loads(raw)
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError("LLM did not return a JSON array for queries.")
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:NUM_QUERIES]


def web_search(queries: list[str]) -> list[dict]:
    collected = []
    seen_urls = set()
    for query in queries:
        hits = []
        for attempt in range(1, DDG_RETRIES + 1):
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, max_results=RESULTS_PER_QUERY))
                break
            except Exception:
                if attempt < DDG_RETRIES:
                    time.sleep(DDG_RETRY_DELAY)
        for hit in hits:
            url = (hit.get("href") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            snippet = (hit.get("body") or "").strip()
            if len(snippet) < 30:
                continue
            collected.append({
                "query":   query,
                "title":   (hit.get("title") or "").strip(),
                "snippet": snippet,
                "url":     url,
                "credibility": _domain_credibility(url),
            })
    return collected


def cluster_results(results: list[dict]) -> list[dict]:
    if len(results) <= 5:
        return results

    texts = [r["snippet"] for r in results]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return results

    n_clusters = min(len(results), 12)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X)

    cluster_best: dict[int, dict] = {}
    for i, label in enumerate(kmeans.labels_):
        r = results[i]
        score = len(r["snippet"]) * r.get("credibility", 1.0)
        if label not in cluster_best or score > cluster_best[label]["_score"]:
            r["_score"] = score
            cluster_best[label] = r

    clustered = list(cluster_best.values())
    for r in clustered:
        r.pop("_score", None)

    if DEBUG:
        print(f"  {DIM}[Clustering] {len(results)} → {len(clustered)} diverse snippets.{RESET}")
    return clustered



def score_snippets_llm(claim: str, results: list[dict]) -> list[dict]:
    if not results:
        return []

    BATCH = 15
    all_evals = []

    for batch_start in range(0, len(results), BATCH):
        batch = results[batch_start: batch_start + BATCH]
        snippets_text = "\n".join(
            f"[{i}] (Source: {r['title']}) {r['snippet']}"
            for i, r in enumerate(batch)
        )
        user_msg = f'CLAIM: "{claim}"\n\nSNIPPETS:\n{snippets_text}'
        raw = _groq_call(RELEVANCE_SCORE_PROMPT, user_msg, MAX_TOKENS_SCORE)
        try:
            data = json.loads(raw)
            evals = data.get("evaluations", [])
        except json.JSONDecodeError:
            evals = []

        while len(evals) < len(batch):
            evals.append({"relevance": 0, "stance": 0, "credibility_note": ""})

        all_evals.extend(evals[:len(batch)])

    for i, r in enumerate(results):
        ev = all_evals[i] if i < len(all_evals) else {}
        relevance  = max(0, min(10, int(ev.get("relevance", 0))))
        stance     = max(-10, min(10, int(ev.get("stance", 0))))
        cred_note  = ev.get("credibility_note", "")

        cred_penalty = 0.6 if cred_note and len(cred_note) > 5 else 1.0

        relevance_weight = relevance / 10.0
        r["relevance"]       = relevance
        r["stance"]          = stance
        r["credibility_note"] = cred_note
        r["score"] = stance * relevance_weight * r.get("credibility", 1.0) * cred_penalty

    filtered = [r for r in results if r.get("relevance", 0) >= 3]
    if not filtered:
        filtered = sorted(results, key=lambda r: r.get("relevance", 0), reverse=True)[:5]

    if DEBUG:
        print(f"  {DIM}[Scoring] {len(filtered)}/{len(results)} snippets retained after relevance filter.{RESET}")
    return filtered


def optimize_snippets_knapsack(results: list[dict], max_chars: int = 6000) -> list[dict]:
    if not results:
        return []

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "SnippetKnapsack"
    )

    values  = [max(1, int((abs(r.get("score", 0)) + 0.1) * r.get("relevance", 1) * 10))
               for r in results]
    weights = [max(1, len(r["snippet"])) for r in results]

    solver.init(values, [weights], [max_chars])
    solver.solve()

    optimized = [r for i, r in enumerate(results) if solver.best_solution_contains(i)]

    if DEBUG:
        total_chars = sum(len(r["snippet"]) for r in optimized)
        print(f"  {DIM}[Knapsack] Selected {len(optimized)} snippets, {total_chars} chars.{RESET}")
    return optimized


def compute_evidence_summary(results: list[dict]) -> dict:
    if not results:
        return {"weighted_score": 0, "support_count": 0, "refute_count": 0,
                "neutral_count": 0, "consensus": "none", "top_sources": []}

    support = [r for r in results if r.get("score", 0) > 1.5]
    refute  = [r for r in results if r.get("score", 0) < -1.5]
    neutral = [r for r in results if -1.5 <= r.get("score", 0) <= 1.5]

    weighted_score = sum(r.get("score", 0) for r in results)

    support_domains = {urlparse(r["url"]).netloc for r in support}
    refute_domains  = {urlparse(r["url"]).netloc for r in refute}

    if len(support_domains) >= 3 and len(support_domains) > len(refute_domains) * 2:
        consensus = "supports_claim"
    elif len(refute_domains) >= 3 and len(refute_domains) > len(support_domains) * 2:
        consensus = "refutes_claim"
    elif support_domains and refute_domains:
        consensus = "mixed"
    else:
        consensus = "insufficient"

    top_sources = sorted(results, key=lambda r: abs(r.get("score", 0)), reverse=True)[:5]
    return {
        "weighted_score": round(weighted_score, 2),
        "support_count":  len(support),
        "refute_count":   len(refute),
        "neutral_count":  len(neutral),
        "consensus":      consensus,
        "top_sources":    [{"title": r["title"], "url": r["url"], "score": round(r.get("score",0),1)}
                           for r in top_sources],
    }


def minimax_debate(snippets: list[dict]) -> tuple[str, list[dict], dict]:
    _empty_summary = {"weighted_score": 0, "support_count": 0, "refute_count": 0,
                      "neutral_count": 0, "consensus": "insufficient", "top_sources": []}
    if not snippets:
        return "No evidence available.", [], _empty_summary

    summary = compute_evidence_summary(snippets)

    pool    = list(snippets)
    transcript = []
    cited  = []

    support_pool = sorted([r for r in pool if r.get("score", 0) > 1.5],
                          key=lambda r: r["score"], reverse=True)[:3]
    refute_pool  = sorted([r for r in pool if r.get("score", 0) < -1.5],
                          key=lambda r: r["score"])[:3]

    if support_pool:
        for r in support_pool:
            transcript.append(
                f"PROPONENT (argues TRUE):\n"
                f"  Evidence [score={r['score']:.1f}, relevance={r['relevance']}/10]: "
                f"{r['snippet']}\n  [Source: {r['title']} — {r['url']}]"
            )
            cited.append(r)
    else:
        transcript.append("PROPONENT: No substantive supporting evidence found.")

    if refute_pool:
        for r in refute_pool:
            transcript.append(
                f"OPPONENT (argues FALSE):\n"
                f"  Evidence [score={r['score']:.1f}, relevance={r['relevance']}/10]: "
                f"{r['snippet']}\n  [Source: {r['title']} — {r['url']}]"
            )
            cited.append(r)
    else:
        transcript.append("OPPONENT: No substantive refuting evidence found.")

    neutral_pool = sorted([r for r in pool if -1.5 <= r.get("score", 0) <= 1.5
                           and r.get("relevance", 0) >= 6],
                          key=lambda r: r["relevance"], reverse=True)[:2]
    if neutral_pool:
        for r in neutral_pool:
            transcript.append(
                f"NEUTRAL CONTEXT [relevance={r['relevance']}/10]: "
                f"{r['snippet']}\n  [Source: {r['title']}]"
            )

    debate_text = "\n\n".join(transcript)

    if DEBUG:
        print(f"\n  {DIM}--- MINIMAX DEBATE ---")
        print(f"  Support: {summary['support_count']} snippets | Refute: {summary['refute_count']} snippets")
        print(f"  Weighted score: {summary['weighted_score']} | Consensus: {summary['consensus']}")
        print(f"  {debate_text[:600]}...")
        print(f"  ----------------------{RESET}\n")

    return debate_text, cited, summary


def synthesize_verdict(claim: str, debate_text: str, cited: list[dict], summary: dict) -> dict:
    _safe = {"weighted_score": 0, "support_count": 0, "refute_count": 0,
             "neutral_count": 0, "consensus": "insufficient"}
    summary = {**_safe, **(summary or {})}
    verdict_msg = (
        f'CLAIM: "{claim}"\n\n'
        f'EVIDENCE SUMMARY:\n'
        f'  - Weighted score: {summary["weighted_score"]} '
        f'(positive = supports claim, negative = refutes)\n'
        f'  - Sources supporting: {summary["support_count"]} | '
        f'Sources refuting: {summary["refute_count"]} | '
        f'Neutral: {summary["neutral_count"]}\n'
        f'  - Multi-source consensus: {summary["consensus"]}\n\n'
        f'DEBATE TRANSCRIPT:\n{debate_text}'
    )

    raw = _groq_call(VERDICT_PROMPT, verdict_msg, MAX_TOKENS_VERDICT)
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines()
                        if not l.strip().startswith("```")).strip()

    verdict = json.loads(raw)
    verdict["cited_sources"] = [{"title": r["title"], "url": r["url"]} for r in cited]
    return verdict


def fact_check(claim: str) -> dict:
    claim = claim.strip()
    if not claim:
        raise ValueError("Claim cannot be empty.")

    start = time.time()

    queries = generate_queries(claim)

    raw_results = web_search(queries)
    if not raw_results:
        return {
            "claim": claim, "verdict": "INCONCLUSIVE", "confidence": "LOW",
            "summary": "No web results could be retrieved. Check your internet connection.",
            "key_findings": [], "caveats": "", "cited_sources": [],
            "search_queries": queries, "sources_searched": 0,
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "elapsed_sec": round(time.time() - start, 1),
        }

    clustered = cluster_results(raw_results)
    scored = score_snippets_llm(claim, clustered)
    optimized = optimize_snippets_knapsack(scored)
    debate_text, cited, summary = minimax_debate(optimized)
    verdict = synthesize_verdict(claim, debate_text, cited, summary)

    verdict.update({
        "claim":           claim,
        "search_queries":  queries,
        "sources_searched": len(raw_results),
        "checked_at":      datetime.utcnow().isoformat() + "Z",
        "elapsed_sec":     round(time.time() - start, 1),
    })
    return verdict


def print_result(r: dict) -> None:
    v   = r.get("verdict", "INCONCLUSIVE")
    col = COLOUR.get(v, "")
    ico = ICON.get(v, "?")
    sep = "─" * 64

    print(f"\n{sep}")
    print(f"  {ico}  {col}{BOLD}{v}{RESET}  {DIM}(confidence: {r.get('confidence', 'LOW')}){RESET}")

    reasoning = r.get("reasoning", "")
    if reasoning:
        print(f"\n  {BOLD}Reasoning:{RESET}")
        print(textwrap.indent(textwrap.fill(reasoning, 58), "    "))

    print(f"\n  {BOLD}Summary:{RESET}")
    print(textwrap.indent(textwrap.fill(r.get("summary", ""), 58), "    "))

    findings = r.get("key_findings", [])
    if findings:
        print(f"\n  {BOLD}Key Findings:{RESET}")
        for f in findings:
            print("    › " + textwrap.fill(f, 56, subsequent_indent="      "))

    caveats = r.get("caveats", "")
    if caveats:
        print(f"\n  {DIM}⚠  Caveats: {caveats}{RESET}")

    sources = r.get("cited_sources", [])
    print(f"\n  {BOLD}Sources Cited:{RESET}")
    if not sources:
        print(f"    {DIM}No sources cited.{RESET}")
    for i, s in enumerate(sources, 1):
        print(f"    [{i}] {s.get('title', 'Unknown')}")
        print(f"        {DIM}{s.get('url', '')}{RESET}")

    print(f"\n  {BOLD}Queries used:{RESET}")
    for q in r.get("search_queries", []):
        print(f"    • {q}")

    elapsed = r.get("elapsed_sec", "?")
    srcs    = r.get("sources_searched", "?")
    print(f"\n  {DIM}{elapsed}s elapsed  ·  {srcs} total sources searched{RESET}")
    print(sep + "\n")


def interactive_debate(topic: str) -> None:
    print(f"\n{BOLD}⚔️  DEBATE MODE INITIATED{RESET}")
    print(f"  {DIM}Topic: {topic}{RESET}")
    print(f"  {DIM}Type 'yield', 'exit', or 'quit' to end the debate.{RESET}\n")

    chat_history = [{"role": "system", "content": DEBATER_PROMPT}]
    user_input = topic

    while True:
        chat_history.append({"role": "user", "content": user_input})
        print(f"\n  {DIM}⏳ AI is researching your argument...{RESET}", flush=True)
        try:
            queries     = generate_queries(user_input)
            raw         = web_search(queries)
            clustered   = cluster_results(raw)
            scored      = score_snippets_llm(user_input, clustered)
            optimized   = optimize_snippets_knapsack(scored, max_chars=3000)

            evidence_text = "\n".join(
                f"[{r['title']}] (score={r.get('score',0):.1f}) {r['snippet']}"
                for r in optimized
            )

            current_messages = list(chat_history)
            evidence_prompt = (
                f"USER'S LATEST ARGUMENT:\n{user_input}\n\n"
                f"EVIDENCE FOUND:\n{evidence_text or 'No evidence found.'}\n\n"
                f"Respond to the user's argument using this evidence."
            )
            current_messages[-1] = {"role": "user", "content": evidence_prompt}

            raw_resp = _groq_call(messages=current_messages, max_tokens=800)
            if raw_resp.startswith("```"):
                raw_resp = "\n".join(l for l in raw_resp.splitlines()
                                     if not l.strip().startswith("```")).strip()

            try:
                ai_response = json.loads(raw_resp).get("response", "I have no response.")
            except json.JSONDecodeError:
                ai_response = "I encountered an error formulating my response."

            chat_history.append({"role": "assistant", "content": ai_response})
            print(f"\n{BOLD}🤖 AI Debater:{RESET}")
            print(textwrap.indent(textwrap.fill(ai_response, width=80), "  "))

        except Exception as exc:
            print(f"\n  {COLOUR['FALSE']}[Error]{RESET} {exc}")

        print("\n" + "─" * 60)
        try:
            user_input = input(f"\n{BOLD}You › {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDebate ended.")
            break

        if user_input.lower() in ("exit", "quit", "q", "yield"):
            print("Debate ended. Good match!")
            break
        if not user_input:
            user_input = "Please continue."


def main() -> None:
    global DEBUG
    print(f"\n{BOLD}⚡ VeriFact v2 — Accurate AI Fact-Checker (Groq + K-Means + Knapsack + Minimax){RESET}")
    print(f"  Model : {MODEL}")
    print(f"  Type a claim and press Enter.")
    print(f"  Commands: {BOLD}exit{RESET} · {BOLD}debug{RESET} · {BOLD}debate <topic>{RESET}\n")

    while True:
        try:
            claim = input(f"{BOLD}Claim › {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not claim:
            continue
        if claim.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if claim.lower() == "debug":
            DEBUG = not DEBUG
            state = f"{COLOUR['VERIFIED']}ON{RESET}" if DEBUG else f"{COLOUR['FALSE']}OFF{RESET}"
            print(f"  Debug mode: {state}\n")
            continue
        if claim.lower().startswith("debate "):
            topic = claim[7:].strip()
            if topic:
                interactive_debate(topic)
            else:
                print("Provide a topic. Example: debate The Earth is flat")
            continue

        print(f"\n  {DIM}⏳ Running AI Pipeline...{RESET}", flush=True)
        try:
            result = fact_check(claim)
            print_result(result)
        except json.JSONDecodeError:
            print(f"  {COLOUR['FALSE']}[Error]{RESET} AI returned unexpected format. Try again.\n")
        except APIStatusError as exc:
            print(f"  {COLOUR['FALSE']}[API Error {exc.status_code}]{RESET} {exc.message}\n")
        except APIError as exc:
            print(f"  {COLOUR['FALSE']}[API Error]{RESET} {exc}\n")
        except Exception as exc:
            print(f"  {COLOUR['FALSE']}[Error]{RESET} {exc}\n")


if __name__ == "__main__":
    main()