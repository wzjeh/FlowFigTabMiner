import json
import os
import re
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv


def sanitize_json_text(text: str) -> str:
    s = text or ""
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*", "", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

    decoder = json.JSONDecoder()
    array_candidate = None
    best_candidate = None
    best_priority = -1
    best_len = -1
    for idx, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(s[idx:])
        except json.JSONDecodeError:
            continue
        candidate = s[idx:idx + end]
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        priority = 2 if isinstance(parsed, list) else 1 if isinstance(parsed, dict) else 0
        if isinstance(parsed, list):
            if array_candidate is None or len(candidate) > len(array_candidate):
                array_candidate = candidate
        if priority > best_priority or (priority == best_priority and len(candidate) > best_len):
            best_candidate = candidate
            best_priority = priority
            best_len = len(candidate)
    if array_candidate:
        return array_candidate.strip()
    if best_candidate:
        return best_candidate.strip()

    candidates = []
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_ch:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidates.append(s[start:i + 1])
                        start = -1
    if candidates:
        s = max(candidates, key=len)

    s = re.sub(r"//.*?(?=\n|$)", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    s = re.sub(r":\s*(\.\.\.)\s*([,\}\]])", r": null\2", s)
    return s.strip()


class LLMEngine:
    def __init__(self, api_key=None, base_url=None, model=None, provider=None):
        load_dotenv(override=True)

        from src.utils.config import load_config

        self.cfg = load_config()
        llm_cfg = self.cfg.get("llm", {})
        adj_cfg = llm_cfg.get("adjudication", {})
        ext_cfg = llm_cfg.get("extraction", {})

        self.max_tokens = int(
            os.getenv("LLM_MAX_TOKENS")
            or adj_cfg.get("max_tokens")
            or ext_cfg.get("max_tokens")
            or 65536
        )
        self.provider = (
            provider
            or adj_cfg.get("provider")
            or os.getenv("LLM_PROVIDER")
            or llm_cfg.get("default_provider", "dashscope")
        )
        primary_model = model or adj_cfg.get("model_name") or os.getenv("LLM_MODEL_NAME")
        fallback_models = adj_cfg.get("model_fallbacks") or ext_cfg.get("model_fallbacks") or []
        self.model_candidates = self._build_model_candidates(primary_model, fallback_models)
        self.model_index = 0 if self.model_candidates else -1
        self.model = self.model_candidates[self.model_index] if self.model_candidates else primary_model
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        if self.provider == "dashscope":
            if not self.api_key:
                self.api_key = os.getenv("QWEN_API_KEY")
            if not self.api_key:
                print("[LLMEngine] WARNING: No API Key found for DashScope.")
            else:
                dashscope.api_key = self.api_key
                dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
            self.model = self.model or "qwen-plus"
            print(f"[LLMEngine] Initialized DashScope Native SDK (Model: {self.model})")
        elif self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key or "dummy", base_url=self.base_url)
            self.model = self.model or "model"
            print(f"[LLMEngine] Initialized Generic OpenAI Client (URL: {self.base_url}, Model: {self.model})")

    def chat(self, system_prompt, user_prompt, max_retries: int = 1) -> str:
        import time

        print(f"[LLMEngine] Chat Request via {self.provider}...")
        for attempt in range(max_retries + 1):
            result = self._chat_once(system_prompt, user_prompt)
            if result is not None:
                return result
            if attempt < max_retries:
                print(f"[LLMEngine] Retrying in 5s (attempt {attempt + 2}/{max_retries + 1})...")
                time.sleep(5)
        print("[LLMEngine] All attempts failed - returning empty string.")
        return ""

    def _chat_once(self, system_prompt, user_prompt):
        try:
            if self.provider == "dashscope":
                while True:
                    if self._use_dashscope_compatible_mode():
                        text = self._chat_dashscope_compatible(system_prompt, user_prompt)
                        if text is not None:
                            return text
                        if self._advance_model():
                            continue
                        return None

                    responses = dashscope.Generation.call(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        result_format="message",
                        max_tokens=self._resolve_max_tokens(),
                        enable_thinking=False,
                        stream=True,
                        incremental_output=True,
                    )
                    chunks = []
                    last_chunk = None
                    for chunk in responses:
                        last_chunk = chunk
                        if chunk.status_code == HTTPStatus.OK:
                            text = chunk.output.choices[0].message.content
                            if text:
                                chunks.append(text)
                    if last_chunk is not None and last_chunk.status_code == HTTPStatus.OK:
                        return "".join(chunks)

                    code = getattr(last_chunk, "code", "unknown") if last_chunk else "no_response"
                    msg = getattr(last_chunk, "message", "") if last_chunk else ""
                    print(f"[LLMEngine] API Error: {code} - {msg}")
                    if self._should_advance_model(code, msg) and self._advance_model():
                        continue
                    return None

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                return response.choices[0].message.content

            print(f"[LLMEngine] Unknown provider: {self.provider}")
            return None
        except Exception as e:
            print(f"[LLMEngine] Exception during chat: {e}")
            if self._should_advance_model("", str(e)) and self._advance_model():
                return self._chat_once(system_prompt, user_prompt)
            return None

    def _resolve_max_tokens(self) -> int:
        if self.provider == "dashscope":
            return max(1, min(self.max_tokens, 32768))
        return max(1, self.max_tokens)

    def _use_dashscope_compatible_mode(self) -> bool:
        return bool(self.model) and self.model.startswith("qwen3.5")

    def _dashscope_compatible_base(self) -> str:
        return os.getenv(
            "DASHSCOPE_COMPATIBLE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        ).rstrip("/")

    def _extract_compatible_message(self, payload):
        if not isinstance(payload, dict):
            return ""
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()

        texts = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for block in item.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text", "")
                    if text:
                        texts.append(text)
        return "".join(texts).strip()

    def _chat_dashscope_compatible(self, system_prompt, user_prompt):
        try:
            import requests
        except Exception as exc:
            print(f"[LLMEngine] requests is unavailable for compatible-mode: {exc}")
            return None

        url = f"{self._dashscope_compatible_base()}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": self._resolve_max_tokens(),
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        text = self._extract_compatible_message(data)
        if text:
            return text
        print(f"[LLMEngine] Compatible API returned no message. Keys: {sorted(data.keys())}")
        return None

    def _build_model_candidates(self, explicit_model, fallback_models):
        candidates = []
        if explicit_model:
            candidates.append(explicit_model)
        for model_name in fallback_models:
            if model_name and model_name not in candidates:
                candidates.append(model_name)
        return candidates

    def _should_advance_model(self, code, message) -> bool:
        text = f"{code} {message}".lower()
        patterns = [
            "quota",
            "exhaust",
            "insufficient",
            "token limit",
            "limit exceeded",
            "free allocation",
            "bill",
            "balance",
            "arrears",
            "resource has been exhausted",
            "model not found",
            "unsupported model",
            "invalid model",
            "no such model",
            "url error",
            "winerror 10013",
            "newconnectionerror",
            "failed to establish a new connection",
            "connection pool",
            "connection refused",
            "connection reset",
            "connection aborted",
            "name or service not known",
            "temporary failure in name resolution",
            "max retries exceeded",
            "timed out",
            "read timeout",
            "timeout",
        ]
        return any(pattern in text for pattern in patterns)

    def _advance_model(self) -> bool:
        if self.model_index < 0:
            return False
        next_index = self.model_index + 1
        if next_index >= len(self.model_candidates):
            print("[LLMEngine] No fallback model left.")
            return False
        previous = self.model
        self.model_index = next_index
        self.model = self.model_candidates[self.model_index]
        print(f"[LLMEngine] Switching model: {previous} -> {self.model}")
        return True

    def adjucate(self, pdf_text, unique_terms, pdf_name):
        cache_path = f"data/output/{pdf_name}_knowledge.json"
        if os.path.exists(cache_path):
            print(f"[LLMEngine] Loading cached knowledge from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"[LLMEngine] Sending request via {self.provider}...")
        prompt = self._construct_prompt(pdf_text, unique_terms)

        try:
            if self.provider == "dashscope":
                if self._use_dashscope_compatible_mode():
                    return self._call_dashscope_compatible(prompt, cache_path)
                return self._call_dashscope(prompt, cache_path)
            if self.provider == "openai":
                return self._call_openai_generic(prompt, cache_path)
            raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            print(f"[LLMEngine] Exception during API call: {e}")
            import traceback

            traceback.print_exc()
            return {"term_mapping": {}, "global_conditions": {}}

    def _call_dashscope(self, prompt, cache_path):
        response = dashscope.Generation.call(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized Chemical Literature Assistant. Your task is to extract experimental conditions and resolve abbreviations from scientific text.",
                },
                {"role": "user", "content": prompt},
            ],
            result_format="message",
        )

        if response.status_code == HTTPStatus.OK:
            raw_content = response.output.choices[0].message.content
            return self._process_response(raw_content, cache_path)
        print(f"[LLMEngine] API Error: {response.code} - {response.message}")
        return {"term_mapping": {}, "global_conditions": {}}

    def _call_dashscope_compatible(self, prompt, cache_path):
        try:
            import requests
        except Exception as exc:
            print(f"[LLMEngine] requests is unavailable for compatible-mode: {exc}")
            return {"term_mapping": {}, "global_conditions": {}}

        url = f"{self._dashscope_compatible_base()}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "instructions": "You are a specialized Chemical Literature Assistant. Your task is to extract experimental conditions and resolve abbreviations from scientific text.",
            "input": prompt,
            "max_output_tokens": self._resolve_max_tokens(),
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        raw_content = self._extract_compatible_message(data)
        if raw_content:
            return self._process_response(raw_content, cache_path)
        print(f"[LLMEngine] Compatible API returned no message. Keys: {sorted(data.keys())}")
        return {"term_mapping": {}, "global_conditions": {}}

    def _call_openai_generic(self, prompt, cache_path):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized Chemical Literature Assistant. Your task is to extract experimental conditions and resolve abbreviations from scientific text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content
        return self._process_response(raw_content, cache_path)

    def _process_response(self, raw_content, cache_path):
        print(f"[LLMEngine] Received response ({len(raw_content)} chars)")
        parsed_json = self._clean_json(raw_content)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=2, ensure_ascii=False)
        return parsed_json

    def _construct_prompt(self, text, terms):
        return f"""
Analyze the following scientific text (from a PDF) to resolve specific abbreviations and identifying global experimental conditions.

## Input Terms/Abbreviations to Resolve:
{", ".join(terms)}
(If a term is a common chemical symbol like 'H2' or 'O2', you can ignore it. Focus on ambiguous abbreviations like 'nb', 'PhCl', 'r.t.', 'cat.')

## Task:
1. **Term Mapping**: Search the text for the definition of the terms above. Map the abbreviation to its full chemical name or entity.
2. **Global Conditions**: Identify experimental conditions that appear to be constant or fixed for the visualized experiments.

## Output Format:
Return ONLY a valid JSON object.
{{
    "term_mapping": {{
        "nb": "nitrobenzene"
    }},
    "global_conditions": {{
        "Pressure": "0.1 MPa"
    }}
}}

## Text Content:
{text[:100000]}
(Truncated if too long)
"""

    def _clean_json(self, content):
        cleaned = sanitize_json_text(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("[LLMEngine] JSON Decode Error. Cleaned content:")
            print(cleaned)
            return {"term_mapping": {}, "global_conditions": {}}
