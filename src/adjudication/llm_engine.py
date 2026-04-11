import os
import re
import json
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv


def sanitize_json_text(text: str) -> str:
    """
    Multi-pass JSON sanitization adapted from FlowChemAgents.
    Handles both JSON objects ({}) and arrays ([]).

    Steps:
    1. Strip markdown code fences
    2. Remove control characters
    3. Extract the largest complete { } or [ ] block
    4. Remove // and /* */ comments
    5. Remove trailing commas before } or ]
    6. Normalize ellipsis (... / …) to null
    """
    s = text or ""

    # 1. Strip markdown fences  ```json ... ``` or ``` ... ```
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*", "", s)

    # 2. Remove control characters (keep tab \x09, newline \x0a, CR \x0d)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

    # 3. Find the largest complete JSON structure (object or array)
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

    # 4. Remove // line comments and /* block comments */
    s = re.sub(r"//.*?(?=\n|$)", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # 5. Remove trailing commas before } or ]
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # 6. Normalize ellipsis to null
    s = re.sub(r":\s*(\.\.\.|…)\s*([,\}\]])", r": null\2", s)

    return s.strip()

class LLMEngine:
    def __init__(self, api_key=None, base_url=None, model=None, provider=None):
        load_dotenv(override=True)
        
        # Load Global Config
        from src.utils.config import load_config
        self.cfg = load_config()
        llm_cfg = self.cfg.get("llm", {})
        
        # Configuration Priorities: 
        # 1. Constructor Params
        # 2. Config.yaml Defaults (Specific Context)
        # 3. Environment Variables
        
        adj_cfg = llm_cfg.get("adjudication", {})
        ext_cfg = llm_cfg.get("extraction", {})
        
        # Determine specific context: default to adjudication if unspecified
        # Prefer YAML values if present, then environment variables
        self.provider = provider or adj_cfg.get("provider") or os.getenv("LLM_PROVIDER") or llm_cfg.get("default_provider", "dashscope")
        self.model = model or adj_cfg.get("model_name") or os.getenv("LLM_MODEL_NAME")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        
        # Defaults based on provider
        if self.provider == "dashscope":
            import dashscope
            if not self.api_key:
                 # Try legacy env var
                 self.api_key = os.getenv("QWEN_API_KEY")
            
            if not self.api_key:
                 print("[LLMEngine] WARNING: No API Key found for DashScope.")
            else:
                 dashscope.api_key = self.api_key
                 # Restore intl endpoint (required for sk-061... key)
                 dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
            
            self.model = self.model or "qwen-plus"
            print(f"[LLMEngine] Initialized DashScope Native SDK (Model: {self.model})")
            
        elif self.provider == "openai":
            from openai import OpenAI
            if not self.base_url:
                # If no base_url provided, check if it's a real OpenAI key or error?
                # For this project, user seems to use local LLM via Ngrok/Colab mostly
                pass 
                # raise ValueError("LLM_BASE_URL is required for 'openai' provider (e.g. Colab/Ngrok URL)")
            
            self.client = OpenAI(
                api_key=self.api_key or "dummy", # Local models often ignore key
                base_url=self.base_url
            )
            self.model = self.model or "model" # Default for many local servers
            print(f"[LLMEngine] Initialized Generic OpenAI Client (URL: {self.base_url}, Model: {self.model})")

    def chat(self, system_prompt, user_prompt, max_retries: int = 1) -> str:
        """
        Generic chat completion with retry on failure.
        Returns empty string on unrecoverable error (never returns an error
        message string, which would break downstream JSON parsing).
        """
        import time
        print(f"[LLMEngine] Chat Request via {self.provider}...")

        for attempt in range(max_retries + 1):
            result = self._chat_once(system_prompt, user_prompt)
            if result is not None:
                return result
            if attempt < max_retries:
                print(f"[LLMEngine] Retrying in 5s (attempt {attempt + 2}/{max_retries + 1})...")
                time.sleep(5)

        print("[LLMEngine] All attempts failed — returning empty string.")
        return ""

    def _chat_once(self, system_prompt, user_prompt):
        """Single chat attempt. Returns None on failure (triggers retry)."""
        try:
            if self.provider == "dashscope":
                import dashscope
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                responses = dashscope.Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format='message',
                    max_tokens=32768,
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
                else:
                    code = getattr(last_chunk, 'code', 'unknown') if last_chunk else 'no_response'
                    msg  = getattr(last_chunk, 'message', '') if last_chunk else ''
                    print(f"[LLMEngine] API Error: {code} — {msg}")
                    return None  # triggers retry

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content

            else:
                print(f"[LLMEngine] Unknown provider: {self.provider}")
                return None

        except Exception as e:
            print(f"[LLMEngine] Exception during chat: {e}")
            return None

    def adjucate(self, pdf_text, unique_terms, pdf_name):
        """
        Main adjudication function.
        """
        # Check cache
        cache_path = f"data/output/{pdf_name}_knowledge.json"
        if os.path.exists(cache_path):
            print(f"[LLMEngine] Loading cached knowledge from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)

        print(f"[LLMEngine] Sending request via {self.provider}...")
        prompt = self._construct_prompt(pdf_text, unique_terms)
        
        try:
            if self.provider == "dashscope":
                return self._call_dashscope(prompt, cache_path)
            elif self.provider == "openai":
                return self._call_openai_generic(prompt, cache_path)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

        except Exception as e:
            print(f"[LLMEngine] Exception during API call: {e}")
            import traceback
            traceback.print_exc()
            return {"term_mapping": {}, "global_conditions": {}}

    def _call_dashscope(self, prompt, cache_path):
        import dashscope
        messages = [
            {"role": "system", "content": "You are a specialized Chemical Literature Assistant. Your task is to extract experimental conditions and resolve abbreviations from scientific text."},
            {"role": "user", "content": prompt}
        ]
        
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message'
        )
        
        if response.status_code == HTTPStatus.OK:
            raw_content = response.output.choices[0].message.content
            return self._process_response(raw_content, cache_path)
        else:
            print(f"[LLMEngine] API Error: {response.code} - {response.message}")
            return {"term_mapping": {}, "global_conditions": {}}

    def _call_openai_generic(self, prompt, cache_path):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a specialized Chemical Literature Assistant. Your task is to extract experimental conditions and resolve abbreviations from scientific text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        raw_content = response.choices[0].message.content
        return self._process_response(raw_content, cache_path)

    def _process_response(self, raw_content, cache_path):
        print(f"[LLMEngine] Received response ({len(raw_content)} chars)")
        parsed_json = self._clean_json(raw_content)
        
        # Cache result
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(parsed_json, f, indent=2)
            
        return parsed_json

    def _construct_prompt(self, text, terms):
        return f"""
Analyze the following scientific text (from a PDF) to resolve specific abbreviations and identifying global experimental conditions.

## Input Terms/Abbreviations to Resolve:
{", ".join(terms)}
(If a term is a common chemical symbol like 'H2' or 'O2', you can ignore it. Focus on ambiguous abbreviations like 'nb', 'PhCl', 'r.t.', 'cat.')

## Task:
1. **Term Mapping**: Search the text for the definition of the terms above. Map the abbreviation to its full chemical name or entity.
   - Format: "Abbreviation": "Full Name"
2. **Global Conditions**: Identify experimental conditions that appear to be constant/fixed for the visualized experiments (e.g. "All reactions were carried out at 0.1 MPa", "Temperature was fixed at 25C").
   - Extract: Pressure, Temperature, Solvent, Catalyst (if constant), Reactor Type, Reactor Inner Diameter (ID).
   - Format: "Condition Name": "Value"

## Output Format:
Return ONLY a valid JSON object. Do not include markdown formatting or explanation.
{{
    "term_mapping": {{
        "nb": "nitrobenzene",
        "key2": "value2"
    }},
    "global_conditions": {{
        "Pressure": "0.1 MPa",
        "Solvent": "Toluene",
        "Reactor Type": "Microreactor",
        "Reactor ID": "0.8 mm"
    }}
}}

## Text Content:
{text[:100000]} 
(Truncated if too long)
"""

    def _clean_json(self, content):
        """Parse LLM response to dict using sanitize_json_text."""
        cleaned = sanitize_json_text(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("[LLMEngine] JSON Decode Error. Cleaned content:")
            print(cleaned)
            return {"term_mapping": {}, "global_conditions": {}}
