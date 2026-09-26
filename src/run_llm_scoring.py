#!/usr/bin/env python3
"""Run the four prompt conditions and save the model scores."""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from openai import (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from paths import ELLIPSE_RESULTS, ELLIPSE_SAMPLE, INTERIM_DATA_DIR

# paths
DEFAULT_INPUT_PATH = ELLIPSE_RESULTS if ELLIPSE_RESULTS.exists() else ELLIPSE_SAMPLE
INPUT_PATH = Path(os.getenv("ELLIPSE_INPUT_PATH", str(DEFAULT_INPUT_PATH)))
OUTPUT_PATH = Path(os.getenv("ELLIPSE_OUTPUT_PATH", str(ELLIPSE_RESULTS)))

# experiment settings
NUM_ESSAYS_TO_RUN = int(os.getenv("ELLIPSE_NUM_ESSAYS", "1200"))
REPEAT_PER_ESSAY = 5
TEMPERATURE = None
SLEEP_SECONDS = 1.0

CLIENT_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 3

OVERWRITE_EXISTING_RESULTS = False

# batch mode is available for OpenAI jobs
RUN_MODE = os.getenv("ELLIPSE_RUN_MODE", "sync")
# other modes prepare, submit, check, download, or merge a batch

BATCH_DIR = INTERIM_DATA_DIR / "batch_openai"
BATCH_INPUT_PATH = BATCH_DIR / "batch_input.jsonl"
BATCH_ID_PATH = BATCH_DIR / "batch_id.txt"
BATCH_OUTPUT_PATH = BATCH_DIR / "batch_output.jsonl"
BATCH_ERROR_PATH = BATCH_DIR / "batch_errors.jsonl"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_ENDPOINT = "/v1/responses"
VALID_SCORES = {
    "1.0", "1.5", "2.0", "2.5", "3.0",
    "3.5", "4.0", "4.5", "5.0"
}

GEMINI_REASONING_EFFORT = "minimal"

def csv_env(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


EXPERIMENT_CONDITIONS = csv_env("ELLIPSE_CONDITIONS", "essay_only")


DEEPSEEK_MODELS = csv_env("ELLIPSE_DEEPSEEK_MODELS")
OPENAI_MODELS = csv_env("ELLIPSE_OPENAI_MODELS")
GEMINI_MODELS = csv_env("ELLIPSE_GEMINI_MODELS")


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "not-set",
    base_url=DEEPSEEK_BASE_URL,
    timeout=CLIENT_TIMEOUT_SECONDS,
    max_retries=0,
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY or "not-set",
    timeout=CLIENT_TIMEOUT_SECONDS,
    max_retries=0,
)

gemini_client = OpenAI(
    api_key=GEMINI_API_KEY or "not-set",
    base_url=GEMINI_BASE_URL,
    timeout=CLIENT_TIMEOUT_SECONDS,
    max_retries=0,
)


# rubric text from ELLIPSE
ELLIPSE_HOLISTIC_RUBRIC = """
ELLIPSE holistic scoring rubric (Overall, 1.0–5.0):

5.0: Native-like facility in the use of language with syntactic variety, appropriate word choice and phrases, well-controlled text organization, and precise use of grammar and conventions. Language inaccuracies are rare and do not impede communication.

4.0: Facility in the use of language with syntactic variety and range of words and phrases, controlled organization, and accuracy in grammar and conventions. Occasional language inaccuracies rarely impede communication.

3.0: Facility limited to the use of common structures and generic vocabulary; organization generally controlled although connection is sometimes absent or unsuccessful; errors in grammar, syntax, and usage are present. Communication is impeded in some cases.

2.0: Inconsistent facility in sentence formation, word choice, and mechanics; organization partially developed but may be missing or unsuccessful. Communication is impeded in many instances by language inaccuracies.

1.0: A limited range of familiar words or phrases loosely strung together; frequent errors in grammar, syntax, and usage. Communication is impeded in most cases by language inaccuracies.
""".strip()

ELLIPSE_ANALYTIC_RUBRIC_TEXT = """
[Overall]
5.0: Native-like facility in the use of language with syntactic variety, appropriate word choice and phrases, well-controlled text organization, precise use of grammar and conventions, and only rare language inaccuracies that do not impede communication.
4.0: Facility in the use of language with syntactic variety and range of words and phrases, controlled organization, and accuracy in grammar and conventions; occasional language inaccuracies rarely impede communication.
3.0: Facility limited to the use of common structures and generic vocabulary; organization generally controlled although connection is sometimes absent or unsuccessful; errors in grammar, syntax, and usage are present; communication is impeded in some cases.
2.0: Inconsistent facility in sentence formation, word choice, and mechanics; organization partially developed but may be missing or unsuccessful; communication is impeded in many instances by language inaccuracies.
1.0: A limited range of familiar words or phrases loosely strung together; frequent errors in grammar, syntax, and usage; communication is impeded in most cases by language inaccuracies.

[Cohesion]
5.0: Text organization consistently well controlled using a variety of effective linguistic features such as reference and transitional words and phrases to connect ideas across sentences and paragraphs; appropriate overlap of ideas.
4.0: Organization generally well controlled; a range of cohesive devices used appropriately such as reference and transitional words and phrases to connect ideas; generally appropriate overlap of ideas.
3.0: Organization generally controlled; cohesive devices used but limited in type; some repetitive, mechanical, or faulty use of cohesion within and/or between sentences and paragraphs.
2.0: Organization only partially developed with a lack of logical sequencing of ideas; some basic cohesive devices used but with inaccuracy or repetition.
1.0: No clear control of organization; cohesive devices not present or unsuccessfully used; presentation of ideas unclear.

[Syntax]
5.0: Flexible and effective use of a full range of syntactic structures including simple, compound, and complex sentences; rare minor and negligible errors in sentence formation.
4.0: Appropriate use of a variety of syntactic structures, such as simple, compound, and complex sentences; occasional errors or inappropriateness in sentence formation.
3.0: Simple, compound, and complex syntactic structures are present although the range may be limited; some apparent errors in sentence formation, especially in more complex sentences.
2.0: Some sentence variation is used; many sentence structure problems.
1.0: Pervasive and basic errors in sentence structure and word order that cause confusion; basic sentence errors are common.

[Vocabulary]
5.0: Wide range of vocabulary flexibly and effectively used to convey precise meanings; skillful use of topic-related terms and less common words; rare negligible inaccuracies in word use.
4.0: Sufficient range of vocabulary to allow flexibility and precision; appropriate use of topic-related terms and less common lexical items.
3.0: Minimally adequate range of vocabulary for the topic; no precise use of subtle word meanings; topic-related terms only used occasionally; attempts to use less common vocabulary but with some inaccuracy.
2.0: Narrow range of vocabulary to convey basic and elementary meaning; topic-related terms used inappropriately; errors in word formation and word choice that may distort meanings.
1.0: Limited vocabulary often inappropriately used; limited control of word choice and word forms; little attempt to use topic-related terms.

[Phraseology]
5.0: Flexible and effective use of a variety of phrases, such as idioms, collocations, and lexical bundles, to convey precise and subtle meanings; rare minor inaccuracies that are negligible.
4.0: Appropriate use of a variety of phrases, such as idioms, collocations, and lexical bundles; occasional inaccuracies and colloquialisms.
3.0: Evident use of phrases such as idioms, collocations, and lexical bundles but without much variety; some noticeable repetitions and misuses.
2.0: Narrow range of phrases, such as collocations and lexical bundles, used to convey basic and elementary meaning; many repetitions and/or misuses of phrases.
1.0: Memorized chunks of language, or simple phrasal patterns, predominate; many repetitions and misuses of phrases.

[Grammar]
5.0: Command of grammar and usage with few or no errors.
4.0: Minimal errors in grammar and usage.
3.0: Some errors in grammar and usage.
2.0: Many errors in grammar and usage.
1.0: Errors in grammar and usage throughout.

[Conventions]
5.0: Consistent use of appropriate conventions to convey meaning; spelling, capitalization, and punctuation errors are nonexistent or negligible.
4.0: Generally consistent use of appropriate conventions to convey meaning; spelling, capitalization, and punctuation errors are few and not distracting.
3.0: Developing use of conventions to convey meaning; errors in spelling, capitalization, and punctuation are sometimes distracting.
2.0: Variable use of conventions; spelling, capitalization, and punctuation errors are frequent and distracting.
1.0: Minimal use of conventions; spelling, capitalization, and punctuation errors occur throughout.
""".strip()

ANALYTIC_REQUIRED_KEYS = {
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
    "Overall",
}


# format feature values for the prompt
def format_nested_metrics(metrics_dict: dict) -> str:
    if not isinstance(metrics_dict, dict) or not metrics_dict:
        return "(No metrics available)"

    lines = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_k, sub_v in value.items():
                lines.append(f"- {sub_k}: {sub_v}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines).strip()


def get_all_metrics_block(metrics: dict) -> str:

    if not isinstance(metrics, dict):
        return "(No metrics available)"

    sections = []

    # readability
    if metrics.get("Readability") is not None:

        sections.append("[Readability]")

        sections.append(
            format_nested_metrics(
                metrics.get("Readability")
            )
        )

        sections.append("")

    # lexical complexity
    if metrics.get("Lexical Complexity") is not None:

        sections.append("[Lexical Complexity]")

        sections.append(
            format_nested_metrics(
                metrics.get("Lexical Complexity")
            )
        )

        sections.append("")

    # syntactic complexity
    if metrics.get("Syntactic Complexity") is not None:

        sections.append("[Syntactic Complexity]")

        sections.append(
            format_nested_metrics(
                metrics.get("Syntactic Complexity")
            )
        )

        sections.append("")

    # cohesion
    if metrics.get("Cohesion") is not None:

        sections.append("[Cohesion]")

        sections.append(
            format_nested_metrics(
                metrics.get("Cohesion")
            )
        )

        sections.append("")

    if not sections:
        return "(No metrics available)"

    return "\n".join(sections).strip()


# four prompt versions
def build_prompt_essay_only(task_prompt: str, essay_text: str, rubric_text: str) -> str:
    return f"""
You are an expert English writing assessor.

Your task is to assign an overall ELLIPSE writing score to the learner essay.

Use the following ELLIPSE holistic rubric as the scoring standard:

{rubric_text}

Scoring instructions:
- Read the rubric, writing prompt, and learner response carefully.
- Assign ONE overall score only, according to the rubric.
- Allowed scores are exactly:
  1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
- If the essay falls between two whole-score descriptors, use the corresponding half-band score.
- Do NOT provide any explanation.
- Do NOT provide any extra words, punctuation, or formatting.
- Output exactly one score from the allowed list above.

Writing prompt:
{task_prompt}

Learner essay:
{essay_text}
""".strip()


def build_prompt_essay_no_rubric(task_prompt: str, essay_text: str) -> str:
    return f"""
You are an expert English writing assessor.

Your task is to assign an overall ELLIPSE writing score to the learner essay.

Scoring instructions:
- Read the writing prompt and learner response carefully.
- Assign ONE overall score only based on your judgment of the essay quality.
- Allowed scores are exactly:
  1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
- Use the full score scale as appropriately as possible.
- Do NOT provide any explanation.
- Do NOT provide any extra words, punctuation, or formatting.
- Output exactly one score from the allowed list above.

Writing prompt:
{task_prompt}

Learner essay:
{essay_text}
""".strip()


def build_prompt_essay_plus_all_metrics(task_prompt: str, essay_text: str, rubric_text: str, metrics: dict) -> str:
    all_metrics_block = get_all_metrics_block(metrics)

    return f"""
You are an expert English writing assessor.

Your task is to assign an overall ELLIPSE writing score to the learner essay.

Use the following ELLIPSE holistic rubric as the scoring standard:

{rubric_text}

Scoring instructions:
- Read the rubric, writing prompt, and learner response carefully.
- In addition to the essay itself, consider the provided linguistic metrics as relevant evidence for evaluating the writing.
- Use both the essay and the provided metrics as evidence.
- Assign ONE overall score only, according to the rubric.
- Allowed scores are exactly:
  1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
- If the essay falls between two whole-score descriptors, use the corresponding half-band score.
- Do NOT provide any explanation.
- Do NOT provide any extra words, punctuation, or formatting.
- Output exactly one score from the allowed list above.

Writing prompt:
{task_prompt}

Learner essay:
{essay_text}

Linguistic Metrics:
{all_metrics_block}
""".strip()


def build_prompt_essay_plus_analytic_rubric(task_prompt: str, essay_text: str, analytic_rubric_text: str) -> str:
    return f"""
You are an expert English writing assessor.

Your task is to evaluate the learner essay using the ELLIPSE rubric.

Use the following ELLIPSE analytic and holistic rubric:

{analytic_rubric_text}

Scoring instructions:
- Read the writing prompt and learner essay carefully.
- First assign one score for each analytic dimension:
  Cohesion, Syntax, Vocabulary, Phraseology, Grammar, and Conventions.
- Then assign one Overall score.
- The Overall score should reflect the overall writing quality of the essay as a whole.
- The Overall score should NOT be treated as a simple arithmetic average of the analytic dimension scores.
- Allowed scores are exactly:
  1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
- Use the full score scale as appropriately as possible.
- Do NOT provide any explanation.
- Do NOT provide any extra words outside the JSON object.
- Output only valid JSON in the following format:

{{
  "Cohesion": "3.0",
  "Syntax": "3.0",
  "Vocabulary": "3.0",
  "Phraseology": "3.0",
  "Grammar": "3.0",
  "Conventions": "3.0",
  "Overall": "3.0"
}}

Writing prompt:
{task_prompt}

Learner essay:
{essay_text}
""".strip()


def normalize_dimension_score(x) -> str | None:
    if x is None:
        return None

    if isinstance(x, (int, float)):
        x = f"{float(x):.1f}"
    elif isinstance(x, str):
        x = x.strip()
        try:
            x = f"{float(x):.1f}"
        except ValueError:
            return None
    else:
        return None

    return x if x in VALID_SCORES else None


def parse_dimension_json_output(raw_text: str) -> dict | None:
    if not raw_text:
        return None

    raw_text = raw_text.strip()

    try:
        obj = json.loads(raw_text)
    except Exception:
        try:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            obj = json.loads(raw_text[start:end + 1])
        except Exception:
            return None

    parsed = {}

    for key in ANALYTIC_REQUIRED_KEYS:
        value = obj.get(key)
        normalized = normalize_dimension_score(value)
        if normalized is None:
            return None
        parsed[key] = normalized

    return parsed


def build_prompt_by_condition(
    condition_name: str,
    task_prompt: str,
    essay_text: str,
    metrics: dict,
    rubric_text: str
) -> str:

    if condition_name == "essay_only":

        return build_prompt_essay_only(
            task_prompt,
            essay_text,
            rubric_text
        )

    elif condition_name == "essay_no_rubric":

        return build_prompt_essay_no_rubric(
            task_prompt,
            essay_text
        )

    elif condition_name == "essay_plus_all_metrics":

        return build_prompt_essay_plus_all_metrics(
            task_prompt,
            essay_text,
            rubric_text,
            metrics
        )

    elif condition_name == "essay_plus_analytic_rubric":

        return build_prompt_essay_plus_analytic_rubric(
            task_prompt,
            essay_text,
            ELLIPSE_ANALYTIC_RUBRIC_TEXT
        )

    else:

        raise ValueError(
            f"Unknown condition: {condition_name}"
        )


# clean model output
def normalize_ellipse_score(text: str) -> str:
    if not text:
        return "INVALID"

    cleaned = text.strip()

    if cleaned in VALID_SCORES:
        return cleaned

    matches = re.findall(r"\b(?:1\.0|1\.5|2\.0|2\.5|3\.0|3\.5|4\.0|4\.5|5\.0)\b", cleaned)
    for m in matches:
        if m in VALID_SCORES:
            return m

    return "INVALID"


def is_analytic_condition(condition_name: str) -> bool:
    return condition_name == "essay_plus_analytic_rubric"


# provider calls
def _chat_completion(
    client: OpenAI,
    provider: str,
    model_name: str,
    prompt_text: str
) -> str:
    if provider == "deepseek":
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            stream=False
        )

        return resp.choices[0].message.content.strip()

    # Gemini goes through Google's OpenAI-compatible endpoint
    if provider == "gemini":

        kwargs = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "stream": False,
        }

        if TEMPERATURE is not None:
            kwargs["temperature"] = TEMPERATURE

        if GEMINI_REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = GEMINI_REASONING_EFFORT

        resp = client.chat.completions.create(**kwargs)

        return resp.choices[0].message.content.strip()

    kwargs = {
        "model": model_name,
        "input": prompt_text,
    }

    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    if provider == "openai":
        kwargs["reasoning"] = {
            "effort": "none"
        }

    resp = client.responses.create(**kwargs)

    return resp.output_text.strip()


def call_with_retry(provider: str, model_name: str, prompt_text: str) -> str:
    if provider == "deepseek":
        client = deepseek_client
    elif provider == "openai":
        client = openai_client
    elif provider == "gemini":
        client = gemini_client
    else:
        raise ValueError(f"Unknown provider: {provider}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _chat_completion(client, provider, model_name, prompt_text)

        except AuthenticationError as e:
            raise RuntimeError(
                f"AuthenticationError for {provider}::{model_name}. "
                f"Check API key / base_url."
            ) from e

        except BadRequestError as e:
            raise RuntimeError(
                f"BadRequestError for {provider}::{model_name}. "
                f"Possibly invalid model, too-long prompt, or bad request."
            ) from e

        except (APITimeoutError, APIConnectionError) as e:
            wait = 2 ** attempt
            print(f"{type(e).__name__} ({attempt}/{MAX_RETRIES}), retry in {wait}s", flush=True)
            time.sleep(wait)

        except RateLimitError:
            wait = 5 * attempt
            print(f"RateLimitError ({attempt}/{MAX_RETRIES}), retry in {wait}s", flush=True)
            time.sleep(wait)

    raise RuntimeError("API failed after retries")


# OpenAI batch handling
def make_batch_custom_id(
    text_id: str,
    model_key: str,
    condition_name: str,
    run_k: int
) -> str:

    return (
        f"{text_id}"
        f"||{model_key}"
        f"||{condition_name}"
        f"||{run_k}"
    )


def parse_batch_custom_id(custom_id: str):

    parts = custom_id.split("||")

    if len(parts) != 4:
        raise ValueError(f"Invalid batch custom_id: {custom_id}")

    text_id, model_key, condition_name, run_k = parts

    return text_id, model_key, condition_name, int(run_k)


def build_openai_batch_body(
    model_name: str,
    prompt_text: str,
    condition_name: str
) -> dict:

    body = {
        "model": model_name,
        "input": prompt_text,
    }

    if TEMPERATURE is not None:
        body["temperature"] = TEMPERATURE

    # analytic output needs more room for the six dimensions
    body["max_output_tokens"] = (
        256
        if is_analytic_condition(condition_name)
        else 16
    )

    body["reasoning"] = {
        "effort": "none"
    }

    return body


def extract_text_from_responses_body(body: dict) -> str:

    if not isinstance(body, dict):
        return ""

    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()

    parts = []

    for item in body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts).strip()


def generate_openai_batch_input(records: list, jobs: list):

    openai_jobs = [
        job
        for job in jobs
        if job[0] == "openai"
    ]

    if not openai_jobs:
        raise RuntimeError("No OpenAI models selected for batch_prepare.")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    total_requests = 0
    skipped_complete = 0

    with BATCH_INPUT_PATH.open("w", encoding="utf-8") as f_out:

        for condition_name in EXPERIMENT_CONDITIONS:

            for i, obj in enumerate(records, start=1):

                text_id = str(obj.get("text_id_kaggle"))
                task_prompt = str(obj.get("prompt", "")).strip()
                essay_text = str(obj.get("full_text", "")).strip()
                metrics = obj.get("metrics", {})

                if not has_required_metrics(condition_name, metrics):
                    raise ValueError(
                        f"\n Missing required metrics!\n"
                        f"Essay: {text_id}\n"
                        f"Condition: {condition_name}\n"
                        f"Metrics keys found: {list(metrics.keys())}"
                    )

                prompt_text = build_prompt_by_condition(
                    condition_name=condition_name,
                    task_prompt=task_prompt,
                    essay_text=essay_text,
                    metrics=metrics,
                    rubric_text=ELLIPSE_HOLISTIC_RUBRIC
                )

                for provider, model_name, model_key in openai_jobs:

                    ensure_result_slots(
                        obj,
                        model_key,
                        condition_name
                    )

                    if OVERWRITE_EXISTING_RESULTS:
                        obj["ellipse_result"][model_key][condition_name] = []
                    else:
                        reset_if_error(
                            obj,
                            model_key,
                            condition_name
                        )

                    if (
                        not OVERWRITE_EXISTING_RESULTS
                        and is_complete(
                            obj,
                            model_key,
                            condition_name
                        )
                    ):
                        skipped_complete += 1
                        continue

                    score_list = obj["ellipse_result"][model_key][condition_name]

                    for run_k in range(len(score_list) + 1, REPEAT_PER_ESSAY + 1):

                        custom_id = make_batch_custom_id(
                            text_id=text_id,
                            model_key=model_key,
                            condition_name=condition_name,
                            run_k=run_k
                        )

                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": BATCH_ENDPOINT,
                            "body": build_openai_batch_body(
                                model_name=model_name,
                                prompt_text=prompt_text,
                                condition_name=condition_name
                            )
                        }

                        f_out.write(
                            json.dumps(req, ensure_ascii=False) + "\n"
                        )

                        total_requests += 1

    print(
        f"Prepared {total_requests} requests in {BATCH_INPUT_PATH}; "
        f"skipped {skipped_complete} completed result sets.",
        flush=True,
    )


def submit_openai_batch():

    if not BATCH_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Batch input not found: {BATCH_INPUT_PATH}. "
            f"Run RUN_MODE='batch_prepare' first."
        )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batch_file = openai_client.files.create(
        file=BATCH_INPUT_PATH.open("rb"),
        purpose="batch"
    )

    batch = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW
    )

    BATCH_ID_PATH.write_text(
        batch.id,
        encoding="utf-8"
    )

    print(f"Submitted batch {batch.id}; id saved to {BATCH_ID_PATH}.", flush=True)


def get_saved_batch_id() -> str:

    if not BATCH_ID_PATH.exists():
        raise FileNotFoundError(
            f"Batch id file not found: {BATCH_ID_PATH}"
        )

    return BATCH_ID_PATH.read_text(encoding="utf-8").strip()


def show_openai_batch_status():

    batch_id = get_saved_batch_id()
    batch = openai_client.batches.retrieve(batch_id)

    print(f"Batch id: {batch.id}", flush=True)
    print(f"Status  : {batch.status}", flush=True)
    print(f"Output  : {batch.output_file_id}", flush=True)
    print(f"Errors  : {batch.error_file_id}", flush=True)


def save_file_content(file_id: str, path: Path):

    if not file_id:
        return

    content = openai_client.files.content(file_id)

    if hasattr(content, "write_to_file"):
        content.write_to_file(str(path))
    else:
        data = content.read()
        path.write_bytes(data)


def download_openai_batch_output():

    batch_id = get_saved_batch_id()
    batch = openai_client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch is not completed yet. Current status: {batch.status}"
        )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    save_file_content(
        batch.output_file_id,
        BATCH_OUTPUT_PATH
    )

    if batch.error_file_id:
        save_file_content(
            batch.error_file_id,
            BATCH_ERROR_PATH
        )

    message = f"Downloaded batch output to {BATCH_OUTPUT_PATH}"
    if batch.error_file_id:
        message += f" and errors to {BATCH_ERROR_PATH}"
    print(message + ".", flush=True)


def merge_openai_batch_output(records: list):

    if not BATCH_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Batch output not found: {BATCH_OUTPUT_PATH}. "
            f"Run RUN_MODE='batch_download' first."
        )

    idx = index_by_text_id(records)

    merged = 0
    skipped_existing = 0
    skipped_gap = 0
    invalid = 0
    errors = 0
    batch_items = []

    def is_existing_valid_result(existing, condition_name: str) -> bool:

        if is_analytic_condition(condition_name):
            if not isinstance(existing, dict):
                return False
            if not ANALYTIC_REQUIRED_KEYS.issubset(set(existing.keys())):
                return False
            return all(
                existing.get(key) in VALID_SCORES
                for key in ANALYTIC_REQUIRED_KEYS
            )

        return existing in VALID_SCORES

    with BATCH_OUTPUT_PATH.open("r", encoding="utf-8") as f_in:

        for line in f_in:

            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            custom_id = item.get("custom_id")

            try:
                text_id, model_key, condition_name, run_k = parse_batch_custom_id(custom_id)
            except Exception:
                errors += 1
                continue

            batch_items.append(
                (
                    text_id,
                    model_key,
                    condition_name,
                    run_k,
                    item
                )
            )

    # batch responses may arrive out of order
    batch_items.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
            x[3]
        )
    )

    for text_id, model_key, condition_name, run_k, item in batch_items:

        obj = idx.get(text_id)
        if obj is None:
            errors += 1
            continue

        ensure_result_slots(
            obj,
            model_key,
            condition_name
        )

        result_list = obj["ellipse_result"][model_key][condition_name]

        # keep any valid result already saved
        if run_k <= len(result_list):
            existing = result_list[run_k - 1]
            if is_existing_valid_result(existing, condition_name):
                skipped_existing += 1
                continue

        elif run_k != len(result_list) + 1:
            skipped_gap += 1
            continue

        response = item.get("response")
        error = item.get("error")

        if error is not None or not isinstance(response, dict):
            label = "ERROR"
        else:
            body = response.get("body", {})
            raw = extract_text_from_responses_body(body)

            if is_analytic_condition(condition_name):
                parsed = parse_dimension_json_output(raw)
                label = parsed if parsed is not None else "INVALID"
            else:
                label = normalize_ellipse_score(raw)

        if label == "INVALID":
            invalid += 1
        if label == "ERROR":
            errors += 1

        if run_k <= len(result_list):
            result_list[run_k - 1] = label
        else:
            result_list.append(label)

        merged += 1

    if merged > 0:
        save_checkpoint(records, OUTPUT_PATH)

    print(
        f"Merged {merged} results into {OUTPUT_PATH}; skipped {skipped_existing} "
        f"existing results and {skipped_gap} gaps; {invalid} invalid and "
        f"{errors} error results.",
        flush=True,
    )


def run_batch_mode(records: list, jobs: list):

    if RUN_MODE == "batch_prepare":
        generate_openai_batch_input(records, jobs)

    elif RUN_MODE == "batch_submit":
        submit_openai_batch()

    elif RUN_MODE == "batch_status":
        show_openai_batch_status()

    elif RUN_MODE == "batch_download":
        download_openai_batch_output()

    elif RUN_MODE == "batch_merge":
        merge_openai_batch_output(records)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


# JSONL loading and checkpoints
def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_text_id(records):
    idx = {}
    for r in records:
        text_id = r.get("text_id_kaggle")
        if text_id is not None:
            idx[str(text_id)] = r
    return idx


def ensure_result_slots(obj: dict, model_key: str, condition_name: str):
    if "ellipse_result" not in obj or not isinstance(obj["ellipse_result"], dict):
        obj["ellipse_result"] = {}

    if model_key not in obj["ellipse_result"]:
        obj["ellipse_result"][model_key] = {}

    obj["ellipse_result"][model_key].setdefault(condition_name, [])


def has_bad_values(lst):
    """Check scalar scores and analytic-score dictionaries."""
    for x in lst:
        if isinstance(x, str) and x in {"ERROR", "INVALID"}:
            return True

        if isinstance(x, dict):
            if not ANALYTIC_REQUIRED_KEYS.issubset(set(x.keys())):
                return True
            for key in ANALYTIC_REQUIRED_KEYS:
                if x.get(key) not in VALID_SCORES:
                    return True

    return False


def is_complete(obj: dict, model_key: str, condition_name: str) -> bool:
    ensure_result_slots(obj, model_key, condition_name)
    result_list = obj["ellipse_result"][model_key][condition_name]

    if len(result_list) != REPEAT_PER_ESSAY:
        return False
    if has_bad_values(result_list):
        return False
    return True


def reset_if_error(obj: dict, model_key: str, condition_name: str):
    ensure_result_slots(obj, model_key, condition_name)
    result_list = obj["ellipse_result"][model_key][condition_name]

    if has_bad_values(result_list):
        obj["ellipse_result"][model_key][condition_name] = []

def has_required_metrics(condition_name: str, metrics: dict) -> bool:
    if condition_name != "essay_plus_all_metrics":
        return True
    required = {
        "Readability",
        "Lexical Complexity",
        "Syntactic Complexity",
        "Cohesion",
    }
    return isinstance(metrics, dict) and required.issubset(metrics)

def save_checkpoint(records: list, out_path: Path):
    tmp = out_path.with_suffix(".tmp.jsonl")
    with tmp.open("w", encoding="utf-8") as f_out:
        for r in records:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def run_experiment_condition(records: list, jobs: list, condition_name: str):
    total_n = len(records)

    for i, obj in enumerate(records, start=1):
        text_id = str(obj.get("text_id_kaggle"))
        if i == 1 or i % 25 == 0 or i == total_n:
            print(f"{condition_name}: essay {i}/{total_n}", flush=True)

        task_prompt = str(obj.get("prompt", "")).strip()
        essay_text = str(obj.get("full_text", "")).strip()
        metrics = obj.get("metrics", {})

        if not has_required_metrics(condition_name, metrics):
            raise ValueError(
                f"Essay {text_id} lacks the features required for {condition_name}."
            )

        prompt_text = build_prompt_by_condition(
            condition_name=condition_name,
            task_prompt=task_prompt,
            essay_text=essay_text,
            metrics=metrics,
            rubric_text=ELLIPSE_HOLISTIC_RUBRIC
        )

        for provider, model_name, model_key in jobs:
            ensure_result_slots(obj, model_key, condition_name)
            reset_if_error(obj, model_key, condition_name)

            if OVERWRITE_EXISTING_RESULTS:
                obj["ellipse_result"][model_key][condition_name] = []

            if (
                not OVERWRITE_EXISTING_RESULTS
                and is_complete(
                    obj,
                    model_key,
                    condition_name
                )
            ):

                continue

            score_list = obj["ellipse_result"][model_key][condition_name]

            while len(score_list) < REPEAT_PER_ESSAY:
                try:
                    raw = call_with_retry(
                        provider,
                        model_name,
                        prompt_text
                    )

                    if is_analytic_condition(condition_name):
                        parsed = parse_dimension_json_output(raw)
                        label = parsed if parsed is not None else "INVALID"

                    else:
                        label = normalize_ellipse_score(raw)

                except Exception as e:
                    label = "ERROR"
                    print(
                        f"API error for essay {text_id}, {model_key}: "
                        f"{type(e).__name__}",
                        flush=True,
                    )

                score_list.append(label)

                save_checkpoint(records, OUTPUT_PATH)

                time.sleep(SLEEP_SECONDS)

                if (
                    isinstance(label, str)
                    and label in {"ERROR", "INVALID"}
                ):

                    reset_if_error(obj, model_key, condition_name)
                    score_list = obj["ellipse_result"][model_key][condition_name]
                    save_checkpoint(records, OUTPUT_PATH)
                    break


def main():
    all_records = load_jsonl(INPUT_PATH)
    if RUN_MODE == "batch_merge":
        records = all_records
    else:
        records = all_records[:NUM_ESSAYS_TO_RUN]

    jobs = [
        (provider, model, f"{provider}::{model}")
        for provider, models in (
            ("openai", OPENAI_MODELS),
            ("deepseek", DEEPSEEK_MODELS),
            ("gemini", GEMINI_MODELS),
        )
        for model in models
    ]
    print(f"Loaded {len(records)} essays for {len(jobs)} models.", flush=True)

    if RUN_MODE != "sync":
        run_batch_mode(records=records, jobs=jobs)
        return

    for condition_name in EXPERIMENT_CONDITIONS:
        run_experiment_condition(
            records=records,
            jobs=jobs,
            condition_name=condition_name
        )

    save_checkpoint(records, OUTPUT_PATH)
    print("Scoring finished.", flush=True)


if __name__ == "__main__":
    main()
