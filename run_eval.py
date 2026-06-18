"""
GDEval-LLM evaluation engine.

Implements and compares three assessment methods on the SAME images and rubric:
  1. direct  - Direct LMM scoring (holistic, no rubric)
  2. rubric  - Rubric-based prompting (the common "paste rubric into chatbot" approach)
  3. gdeval  - The proposed evidence-grounded staged prompting workflow

Important implementation note for the feedback-quality experiment:
  Score generation and final student-facing feedback generation are separated.
  Each method first produces method-specific assessment information, and then
  ALL THREE methods pass that information through the SAME standardized Korean
  feedback-generation prompt.

This makes the feedback comparison depend on the assessment information available
before feedback generation, not on different final feedback prompts.

Usage for the reported experiments:
  python run_eval.py --provider claude --method all --images input --out output --repeats 1
  python run_eval.py --provider openai --method all --images input --out output --repeats 1

Optional assignment brief:
  If the study design provides the same assignment brief to the LMM prompts,
  pass it with --brief-file path/to/brief.txt. The same brief text is then
  inserted into all method-specific scoring prompts and the standardized final
  feedback prompt.

Two providers are supported so results can be reported on both models:
  --provider claude  -> reads ANTHROPIC_API_KEY, uses the Anthropic SDK
  --provider openai  -> reads OPENAI_API_KEY, uses the OpenAI SDK
Outputs are written per provider (e.g. output/claude/results_gdeval.csv) so the
runs never overwrite each other.

No API key is passed in code; each SDK reads its key from the environment.
Provider, model, and configuration fields are logged to every output row.
"""

import os, sys, json, csv, base64, argparse, time, statistics
from dataclasses import dataclass, asdict, field

try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import openai
except ImportError:
    openai = None

# ---- Reproducibility config (logged to every output) ----
# Per-provider model strings. Override with --model if you use a different version.
PROVIDER_MODELS = {
    "claude": "claude-opus-4-8",          # exact Claude label recorded in output files
    "openai": "gpt-4o",                  # exact GPT-4o label recorded in output files
}
CONFIG = {
    "provider": "claude",
    "model": PROVIDER_MODELS["claude"],
    "temperature": 0.0,                  # logged value; Claude Opus call may omit this parameter
    "max_tokens": 2000,
    "image_max_edge_px": 1568,           # downscale long edge for consistent input
    "assignment_brief_provided": False,
}

# ============================================================
# Data structures
# ============================================================

@dataclass
class CriterionResult:
    code: str
    name: str
    visual_evidence: list          # observable features selected for this criterion; GDEval only
    score: float                   # 1-7
    justification: str             # evidence -> score reasoning; GDEval only

@dataclass
class WorkResult:
    work_id: str
    method: str
    criteria: list                 # list[CriterionResult]
    overall_score: float           # direct: model holistic score; rubric/GDEval: mean of criterion scores
    feedback: str                  # final standardized Korean student-facing feedback
    raw_model_output: str = ""     # JSON string containing scoring and standardized-feedback raw outputs
    initial_feedback: str = ""     # direct/rubric initial feedback text before standardized generation
    feedback_generation_input: str = ""  # text supplied to the standardized final feedback prompt
    config: dict = field(default_factory=lambda: dict(CONFIG))

# ============================================================
# Image handling and provider calls
# ============================================================

def load_image_b64(path):
    from PIL import Image
    import io
    img = Image.open(path).convert("RGB")
    w, h = img.size
    edge = max(w, h)
    if edge > CONFIG["image_max_edge_px"]:
        s = CONFIG["image_max_edge_px"] / edge
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def make_client(provider):
    if provider == "claude":
        if anthropic is None:
            sys.exit("anthropic SDK not installed: pip install anthropic")
        return anthropic.Anthropic()       # reads ANTHROPIC_API_KEY
    if provider == "openai":
        if openai is None:
            sys.exit("openai SDK not installed: pip install openai")
        return openai.OpenAI()             # reads OPENAI_API_KEY
    sys.exit(f"unknown provider: {provider}")


def call_model(client, image_b64, system, user_text):
    """Single LMM call, provider-agnostic. Returns raw text.
    Auto-retries on 429/500/529 with exponential backoff (max 5 attempts)."""
    provider = CONFIG["provider"]

    for attempt in range(5):
        try:
            if provider == "claude":
                kwargs = {}
                if CONFIG["temperature"] is not None and not CONFIG["model"].startswith("claude-opus-4-8"):
                    kwargs["temperature"] = CONFIG["temperature"]
                msg = client.messages.create(
                    model=CONFIG["model"],
                    max_tokens=CONFIG["max_tokens"],
                    system=system,
                    **kwargs,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64",
                             "media_type": "image/jpeg", "data": image_b64}},
                            {"type": "text", "text": user_text},
                        ],
                    }],
                )
                return "".join(b.text for b in msg.content if b.type == "text")

            # openai
            resp = client.chat.completions.create(
                model=CONFIG["model"],
                max_tokens=CONFIG["max_tokens"],
                temperature=CONFIG["temperature"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url":
                         {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ]},
                ],
            )
            return resp.choices[0].message.content

        except Exception as e:
            err_str = str(e)
            # Retry on overload (529), internal server error (500), rate limit (429)
            if any(code in err_str for code in ["529", "500", "429", "overloaded", "rate_limit", "InternalServer"]):
                wait = 20 * (2 ** attempt)  # 20s, 40s, 80s, 160s, 320s
                print(f"  [retry {attempt+1}/5] API error ({err_str[:60]}...) — waiting {wait}s")
                time.sleep(wait)
                if attempt == 4:
                    raise
            else:
                raise


def parse_json(text):
    """Strip code fences and parse. Raises on failure so scores are never fabricated."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    return json.loads(t.strip())


def brief_block(assignment_brief):
    if not assignment_brief:
        return ""
    return f"\n\nAssignment brief:\n{assignment_brief.strip()}\n"

# ============================================================
# Scoring prompts: each method first creates method-specific assessment information
# ============================================================

def run_direct_scoring(client, work_id, image_b64, rubric, assignment_brief=""):
    """Direct baseline scoring call: holistic score + initial brief feedback text."""
    system = "You are an experienced graphic design instructor."
    user = (
        "Evaluate this undergraduate graphic design work overall on a 1-7 scale "
        "(1=very poor, 7=excellent) and give brief feedback to the student."
        f"{brief_block(assignment_brief)}\n"
        "Return ONLY JSON: {\"overall_score\": <number>, \"feedback\": \"<text>\"}"
    )
    raw = call_model(client, image_b64, system, user)
    d = parse_json(raw)
    initial_feedback = d.get("feedback", "")
    result = WorkResult(
        work_id=work_id,
        method="direct",
        criteria=[],
        overall_score=float(d["overall_score"]),
        feedback="",
        initial_feedback=initial_feedback,
        raw_model_output=json.dumps({"direct_scoring": raw}, ensure_ascii=False),
    )
    payload = {
        "overall_score": result.overall_score,
        "initial_feedback": initial_feedback,
    }
    return result, payload


def run_rubric_scoring(client, work_id, image_b64, rubric, assignment_brief=""):
    """Rubric baseline scoring call: full rubric anchors + five criterion scores + initial rubric-based feedback text."""
    crit_text = "\n".join(
        f"- {c['code']} {c['name']}:\n"
        f"  Definition: {c['definition']}\n"
        f"  Anchor 1: {c['anchors']['1']}\n"
        f"  Anchor 4: {c['anchors']['4']}\n"
        f"  Anchor 7: {c['anchors']['7']}"
        for c in rubric["criteria"])
    system = "You are an experienced graphic design instructor."
    user = (
        "Evaluate this graphic design work using the full rubric below. "
        "Score each criterion 1-7 and give overall feedback."
        f"{brief_block(assignment_brief)}\n\n"
        f"Rubric:\n{crit_text}\n\n"
        "Return ONLY JSON: {\"scores\": {\"C1\": n, \"C2\": n, \"C3\": n, "
        "\"C4\": n, \"C5\": n}, \"feedback\": \"<text>\"}"
    )
    raw = call_model(client, image_b64, system, user)
    d = parse_json(raw)
    crits = [CriterionResult(c["code"], c["name"], [], float(d["scores"][c["code"]]), "")
             for c in rubric["criteria"]]
    overall = statistics.mean(cr.score for cr in crits)
    initial_feedback = d.get("feedback", "")
    result = WorkResult(
        work_id=work_id,
        method="rubric",
        criteria=crits,
        overall_score=overall,
        feedback="",
        initial_feedback=initial_feedback,
        raw_model_output=json.dumps({"rubric_scoring": raw}, ensure_ascii=False),
    )
    payload = {
        "criterion_scores": {cr.code: cr.score for cr in crits},
        "initial_feedback": initial_feedback,
    }
    return result, payload


def _anchor_block(c):
    a = c["anchors"]
    return (f"{c['code']} — {c['name']}\n  Definition: {c['definition']}\n"
            f"  Anchor 1: {a['1']}\n  Anchor 4: {a['4']}\n  Anchor 7: {a['7']}")


def run_gdeval_scoring(client, work_id, image_b64, rubric, assignment_brief=""):
    """GDEval scoring calls: evidence extraction + five independent criterion-level calls."""
    system = (
        "You are an expert graphic design assessor. You evaluate strictly from "
        "observable visual evidence. You never infer intent you cannot see. "
        "You output only valid JSON."
    )

    # ---- Visual evidence extraction call ----
    s1_user = (
        "STAGE 1 — VISUAL EVIDENCE EXTRACTION.\n"
        "List concrete, observable visual features of this design. Describe only "
        "what is visible (layout, spacing, alignment, type choices, sizes, color "
        "relationships, contrast, finishing), not quality judgements yet."
        f"{brief_block(assignment_brief)}\n"
        "Return ONLY JSON: {\"observations\": [\"<feature>\", ...]}"
    )
    s1_raw = call_model(client, image_b64, system, s1_user)
    observations = parse_json(s1_raw).get("observations", [])

    # ---- Per-criterion independent calls; each call embeds evidence selection, anchor mapping, scoring, justification ----
    crits = []
    stage_raw = {"visual_evidence_extraction": s1_raw}
    for c in rubric["criteria"]:
        s24_user = (
            "STAGES 2-4 — evaluate ONE criterion in isolation. Ignore all other "
            "criteria; do not let them influence this score.\n\n"
            f"Criterion:\n{_anchor_block(c)}\n"
            f"{brief_block(assignment_brief)}\n"
            "Visual observations already extracted from the work:\n"
            f"{json.dumps(observations, ensure_ascii=False)}\n\n"
            "Steps:\n"
            "1. Select ONLY the observations relevant to THIS criterion as evidence.\n"
            "2. Map that evidence to the rubric anchors and assign a 1-7 score "
            "(interpolate between anchors).\n"
            "3. Justify the score in <=2 sentences that reference the evidence.\n"
            "Return ONLY JSON: {\"visual_evidence\": [\"<obs>\", ...], "
            "\"score\": <number 1-7>, \"justification\": \"<text>\"}"
        )
        raw = call_model(client, image_b64, system, s24_user)
        d = parse_json(raw)
        crits.append(CriterionResult(
            c["code"], c["name"],
            d.get("visual_evidence", []),
            float(d["score"]),
            d.get("justification", "")))
        stage_raw[f"criterion_call_{c['code']}"] = raw

    overall = round(statistics.mean(cr.score for cr in crits), 3)
    result = WorkResult(
        work_id=work_id,
        method="gdeval",
        criteria=crits,
        overall_score=overall,
        feedback="",
        raw_model_output=json.dumps(stage_raw, ensure_ascii=False),
    )
    payload = {
        "observations": observations,
        "criterion_scores": {cr.code: cr.score for cr in crits},
        "criterion_visual_evidence": {cr.code: cr.visual_evidence for cr in crits},
        "criterion_justifications": {cr.code: cr.justification for cr in crits},
    }
    return result, payload

# ============================================================
# Standardized final feedback generation: SAME prompt for all methods
# ============================================================

KO_SECTION = {
    "C1": "시각적 위계·레이아웃",
    "C2": "타이포그래피",
    "C3": "색채",
    "C4": "컨셉·전달력",
    "C5": "완성도",
}


def build_feedback_information(method, scoring_payload, rubric):
    """Build method-specific information block supplied to the same final feedback prompt."""
    if method == "direct":
        return (
            "평가 정보:\n"
            f"- 전체 점수: {scoring_payload['overall_score']}\n"
            f"- 초기 직접 평가 피드백: {scoring_payload.get('initial_feedback', '')}\n"
        )

    if method == "rubric":
        score_lines = []
        scores = scoring_payload.get("criterion_scores", {})
        for c in rubric["criteria"]:
            label = KO_SECTION.get(c["code"], c["name"])
            score_lines.append(f"- {label}: {scores.get(c['code'], '')}")
        return (
            "평가 정보:\n"
            "기준별 점수:\n" + "\n".join(score_lines) + "\n"
            f"초기 루브릭 기반 피드백: {scoring_payload.get('initial_feedback', '')}\n"
        )

    if method == "gdeval":
        lines = ["평가 정보:", "관찰된 시각 요소:"]
        for obs in scoring_payload.get("observations", []):
            lines.append(f"- {obs}")
        lines.append("기준별 평가 정보:")
        scores = scoring_payload.get("criterion_scores", {})
        evidence = scoring_payload.get("criterion_visual_evidence", {})
        justs = scoring_payload.get("criterion_justifications", {})
        for c in rubric["criteria"]:
            code = c["code"]
            label = KO_SECTION.get(code, c["name"])
            ev = evidence.get(code, [])
            ev_text = "; ".join(ev) if isinstance(ev, list) else str(ev)
            lines.append(
                f"- {label}: 점수 {scores.get(code, '')}; "
                f"관련 시각 증거: {ev_text}; "
                f"근거 기반 설명: {justs.get(code, '')}"
            )
        return "\n".join(lines) + "\n"

    raise ValueError(f"unknown method for feedback information: {method}")


def standardized_feedback_prompt(method_specific_information, assignment_brief=""):
    """The same final Korean student-facing feedback prompt used across all methods."""
    brief = ""
    if assignment_brief:
        brief = f"과제 브리프:\n{assignment_brief.strip()}\n\n"
    return (
        "아래 평가 정보를 바탕으로 학생에게 전하는 한국어 피드백을 작성하세요.\n"
        "점수 숫자, C1~C5 코드명, method 이름은 본문에 절대 노출하지 마세요.\n"
        "따뜻하고 격려하는 디자인 교육자의 어조로 작성하세요.\n"
        "구조:\n"
        "1) 첫 문단 — 강점 칭찬:\n"
        "작품의 강점을 구체적으로 칭찬해 학생의 자신감을 높입니다.\n"
        "2) 항목별 평가:\n"
        "다음 항목명을 그대로 사용해 각 항목을 간결하게 평가합니다.\n"
        "- 시각적 위계·레이아웃\n"
        "- 타이포그래피\n"
        "- 색채\n"
        "- 컨셉·전달력\n"
        "- 완성도\n"
        "3) 마지막 문단 — 종합·개선 방향:\n"
        "작품을 종합적으로 평가하되, 보완점은 구체적인 명령이 아니라 방향성 중심으로 "
        "부드럽게 제시하고 희망적으로 마무리합니다.\n"
        "작성 원칙:\n"
        "- 추상적 칭찬만 하지 말고, 가능한 경우 관찰 가능한 디자인 요소를 언급하세요.\n"
        "- 학생이 다음 수정에서 무엇을 개선할 수 있는지 알 수 있도록 작성하세요.\n"
        "- 비판은 단정적으로 하지 말고 교육적이고 지원적인 어조로 제시하세요.\n"
        "- 350~600자 내외를 목표로 하세요.\n\n"
        f"{brief}"
        f"{method_specific_information}\n"
        "Return ONLY JSON:\n"
        "{\"feedback\": \"<한국어 피드백 전문>\"}"
    )


def generate_standardized_feedback(client, image_b64, method, scoring_payload, rubric, assignment_brief=""):
    """Run the same standardized final feedback prompt for direct, rubric, and GDEval."""
    system = "You are an experienced graphic design instructor. You output only valid JSON."
    info = build_feedback_information(method, scoring_payload, rubric)
    user = standardized_feedback_prompt(info, assignment_brief=assignment_brief)
    raw = call_model(client, image_b64, system, user)
    feedback = parse_json(raw).get("feedback", "")
    return feedback, raw, info

# ============================================================
# Public method wrappers used by the driver
# ============================================================

def _attach_standardized_feedback(scored_result, scoring_payload, client, image_b64, rubric, assignment_brief=""):
    feedback, feedback_raw, feedback_input = generate_standardized_feedback(
        client, image_b64, scored_result.method, scoring_payload, rubric, assignment_brief=assignment_brief
    )
    scoring_raw = scored_result.raw_model_output
    scored_result.feedback = feedback
    scored_result.feedback_generation_input = feedback_input
    scored_result.raw_model_output = json.dumps({
        "scoring_outputs": scoring_raw,
        "standardized_feedback_generation": feedback_raw,
    }, ensure_ascii=False)
    return scored_result


def run_direct(client, work_id, image_b64, rubric, assignment_brief=""):
    direct_score, payload = run_direct_scoring(client, work_id, image_b64, rubric, assignment_brief)
    return _attach_standardized_feedback(direct_score, payload, client, image_b64, rubric, assignment_brief)


def run_rubric(client, work_id, image_b64, rubric, assignment_brief=""):
    rubric_score, payload = run_rubric_scoring(client, work_id, image_b64, rubric, assignment_brief)
    return _attach_standardized_feedback(rubric_score, payload, client, image_b64, rubric, assignment_brief)


def run_gdeval(client, work_id, image_b64, rubric, assignment_brief=""):
    gdeval_score, payload = run_gdeval_scoring(client, work_id, image_b64, rubric, assignment_brief)
    return _attach_standardized_feedback(gdeval_score, payload, client, image_b64, rubric, assignment_brief)


METHODS = {"direct": run_direct, "rubric": run_rubric, "gdeval": run_gdeval}

# ============================================================
# Driver
# ============================================================

def evaluate(method, images_dir, rubric, repeats, dry_run, assignment_brief=""):
    client = None
    if not dry_run:
        client = make_client(CONFIG["provider"])

    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    results = []
    for fn in files:
        work_id = os.path.splitext(fn)[0]
        b64 = None if dry_run else load_image_b64(os.path.join(images_dir, fn))
        for rep in range(repeats):
            if dry_run:
                print(f"[dry-run] {method} | {work_id} | repeat {rep+1}")
                continue
            for attempt in range(3):  # retry on transient parse/API errors
                try:
                    r = METHODS[method](client, work_id, b64, rubric, assignment_brief)
                    r_dict = asdict(r)
                    r_dict["repeat"] = rep + 1
                    results.append(r_dict)
                    break
                except Exception as e:
                    print(
                        f"  ! attempt {attempt+1}/3 failed | "
                        f"method={method} | work_id={work_id} | rep={rep+1} | "
                        f"error={type(e).__name__}: {e}"
                    )
                    if attempt == 2:
                        print(f"  !! final failure | method={method} | work_id={work_id} | rep={rep+1}")
                    time.sleep(5)
    return results


def flatten_for_csv(results, rubric):
    codes = [c["code"] for c in rubric["criteria"]]
    rows = []
    for r in results:
        row = {
            "work_id": r["work_id"],
            "method": r["method"],
            "repeat": r.get("repeat", 1),
            "overall_score": r["overall_score"],
            "feedback": r["feedback"],
            "initial_feedback": r.get("initial_feedback", ""),
            "feedback_generation_input": r.get("feedback_generation_input", ""),
            "provider": r["config"].get("provider", ""),
            "model": r["config"]["model"],
            "temperature": r["config"].get("temperature", ""),
            "assignment_brief_provided": r["config"].get("assignment_brief_provided", False),
        }
        by = {cr["code"]: cr for cr in r["criteria"]}
        for code in codes:
            cr = by.get(code)
            row[f"{code}_score"] = cr["score"] if cr else ""
            row[f"{code}_evidence"] = " | ".join(cr["visual_evidence"]) if cr else ""
            row[f"{code}_justification"] = cr["justification"] if cr else ""
        rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        print("no rows to write"); return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def load_text_file(path):
    if not path:
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="claude", choices=["claude", "openai"])
    ap.add_argument("--model", default=None, help="override the default model string")
    ap.add_argument("--method", default="all", choices=["direct", "rubric", "gdeval", "all"])
    ap.add_argument("--images", default="input")
    ap.add_argument("--rubric", default="rubric.json")
    ap.add_argument("--brief-file", default=None, help="optional assignment brief text inserted into all prompts")
    ap.add_argument("--out", default="output")
    ap.add_argument("--repeats", type=int, default=1, help="number of repeated runs; set to 1 for the reported experiments")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    CONFIG["provider"] = args.provider
    CONFIG["model"] = args.model or PROVIDER_MODELS[args.provider]
    assignment_brief = load_text_file(args.brief_file)
    CONFIG["assignment_brief_provided"] = bool(assignment_brief.strip())

    rubric = json.load(open(args.rubric, encoding="utf-8"))
    # write to output/<provider>/ so Claude and OpenAI runs never collide
    out_dir = args.out if args.out.endswith(".csv") else os.path.join(args.out, args.provider)
    methods = ["direct", "rubric", "gdeval"] if args.method == "all" else [args.method]
    print(f"provider={CONFIG['provider']} model={CONFIG['model']} "
          f"temp={CONFIG['temperature']} repeats={args.repeats} "
          f"assignment_brief_provided={CONFIG['assignment_brief_provided']}")
    for m in methods:
        res = evaluate(m, args.images, rubric, args.repeats, args.dry_run, assignment_brief=assignment_brief)
        if not args.dry_run:
            rows = flatten_for_csv(res, rubric)
            out = out_dir if out_dir.endswith(".csv") else os.path.join(out_dir, f"results_{m}.csv")
            write_csv(rows, out)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            json.dump(res, open(out.replace(".csv", ".json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
