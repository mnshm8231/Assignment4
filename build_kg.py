"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import json
import os
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

# Map regulation name keywords → rule type
_REG_TYPE_MAP = [
    (["exam", "examination"], "exam_rule"),
    (["id card", "replacement", "easycard", "mifare"], "admin_rule"),
    (["grading", "grade"], "grade_rule"),
    (["credit transfer"], "credit_rule"),
    (["course selection", "course"], "course_rule"),
    (["general"], "general_rule"),
]


def _infer_rule_type(reg_name: str, content: str) -> str:
    """Infer rule type from regulation name and article content."""
    combined = (reg_name + " " + content).lower()
    for keywords, rule_type in _REG_TYPE_MAP:
        if any(k in combined for k in keywords):
            return rule_type
    return "general_rule"


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """
    Use local LLM to extract structured rules from an article and return
    {"rules": [{"type": ..., "action": ..., "result": ...}, ...]}.

    Falls back to an empty list on any failure; build_graph() will then call
    build_fallback_rules() as a safety net.
    """
    tok = get_tokenizer()
    pipe = get_raw_pipeline()
    if tok is None or pipe is None:
        return {"rules": []}

    rule_type_hint = _infer_rule_type(reg_name, content)
    # Trim content to avoid exceeding context window
    content_snippet = content[:700].strip()
    if not content_snippet:
        return {"rules": []}

    prompt = (
        f"Extract ALL rules from the regulation article below and output valid JSON only.\n"
        f"Regulation: {reg_name} | Article: {article_number}\n"
        f"Content: {content_snippet}\n\n"
        f"Required JSON format:\n"
        f'{{"rules": [{{"type": "{rule_type_hint}", "action": "<condition or subject>", "result": "<outcome or requirement>"}}]}}\n'
        f"Output JSON now:"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You extract university regulation rules as structured JSON. "
                "Output valid JSON only — no markdown, no extra text."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        raw_output: str = pipe(formatted, max_new_tokens=350)[0]["generated_text"].strip()

        # ── Try several JSON extraction strategies ──────────────────────────
        candidates = []

        # 1) Everything between the outermost { }
        brace_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group())

        # 2) Full output as-is
        candidates.append(raw_output)

        for text in candidates:
            try:
                data = json.loads(text)
                rules = data.get("rules", [])
                valid = []
                for r in rules:
                    if not isinstance(r, dict):
                        continue
                    action = str(r.get("action", "")).strip()
                    result = str(r.get("result", "")).strip()
                    if action and result:
                        valid.append(
                            {
                                "type": str(r.get("type", rule_type_hint)),
                                "action": action[:300],
                                "result": result[:300],
                            }
                        )
                if valid:
                    return {"rules": valid}
            except (json.JSONDecodeError, ValueError):
                continue

    except Exception as e:
        print(f"    [LLM] extraction error for {article_number}: {e}")

    return {"rules": []}


def build_fallback_rules(article_number: str, content: str, reg_name: str = "") -> list[dict[str, str]]:
    """
    Deterministic fallback: split article text into sentences and create a
    Rule for every sentence that contains a numeric fact or key regulation term.
    Always returns at least one rule (the full article text) so no article is
    left uncovered.
    """
    if not content.strip():
        return []

    rule_type = _infer_rule_type(reg_name, content)
    rules: list[dict[str, str]] = []
    seen: set[str] = set()

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.;:])\s+", content)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        # Prioritise sentences that contain factual information
        has_fact = re.search(
            r"\b\d+\b|NTD|points?|credits?|semesters?|years?|minutes?|working\s+days?|percent|%",
            sent,
            re.IGNORECASE,
        )
        if has_fact:
            key = sent[:60]
            if key not in seen:
                seen.add(key)
                rules.append(
                    {"type": rule_type, "action": sent[:300], "result": sent[:300]}
                )

    # Always guarantee at least one rule with the full content
    full_key = content[:60]
    if not rules or full_key not in seen:
        rules.insert(
            0,
            {"type": rule_type, "action": content[:300], "result": content[:300]},
        )

    return rules[:8]


# SQLite tables used:
# - regulations(reg_id, name, category)
# - articles(reg_id, article_number, content)


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # Warm up local LLM once before the extraction loop
    print("[*] Loading local LLM …")
    load_local_llm()
    print("[*] LLM ready.\n")

    with driver.session() as session:
        # Fixed strategy: clear existing graph data before rebuilding.
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Read regulations and create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Read articles and create Article + HAS_ARTICLE.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create full-text index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0
        seen_rules: set[tuple[str, str, str]] = set()  # (reg_name, art_num, action_prefix)

        print(f"[*] Extracting rules from {len(articles)} articles …\n")

        for reg_id, article_number, content in articles:
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))
            print(f"  [{reg_name}] {article_number} …", end=" ", flush=True)

            # ── Primary: LLM extraction ────────────────────────────────────
            extraction = extract_entities(article_number, reg_name, content)
            rules = extraction.get("rules", [])

            # ── Fallback: deterministic sentence splitting ─────────────────
            if not rules:
                rules = build_fallback_rules(article_number, content, reg_name)
                print(f"(fallback, {len(rules)} rules)", flush=True)
            else:
                print(f"(LLM, {len(rules)} rules)", flush=True)

            # ── Write Rule nodes and CONTAINS_RULE edges ───────────────────
            for rule in rules:
                action = str(rule.get("action", "")).strip()
                result = str(rule.get("result", "")).strip()

                if not action or not result:
                    continue

                # Deduplicate by (reg_name, article_number, first-60-chars-of-action)
                dedup_key = (reg_name, article_number, action[:60])
                if dedup_key in seen_rules:
                    continue
                seen_rules.add(dedup_key)

                rule_id = f"{reg_name}__{article_number}__{rule_counter}"
                rule_counter += 1

                session.run(
                    """
                    MATCH (a:Article {number: $num, reg_name: $reg_name})
                    CREATE (r:Rule {
                        rule_id:  $rule_id,
                        type:     $type,
                        action:   $action,
                        result:   $result,
                        art_ref:  $art_ref,
                        reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    num=article_number,
                    reg_name=reg_name,
                    rule_id=rule_id,
                    type=rule.get("type", "general_rule"),
                    action=action,
                    result=result,
                    art_ref=article_number,
                )

        print(f"\n[*] Total Rule nodes created: {rule_counter}")

        # 4) Create full-text index on Rule fields.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        # 5) Coverage audit (provided scaffold).
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
