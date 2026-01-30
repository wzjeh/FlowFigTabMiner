import os
import json
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

class LLMEngine:
    def __init__(self, api_key=None, base_url=None, model=None, provider=None):
        load_dotenv()
        
        # Configuration
        self.provider = provider or os.getenv("LLM_PROVIDER", "dashscope") # 'dashscope' or 'openai'
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("QWEN_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL_NAME")
        
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
                 dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
            
            self.model = self.model or "qwen-plus"
            print(f"[LLMEngine] Initialized DashScope Native SDK (Model: {self.model})")
            
        elif self.provider == "openai":
            from openai import OpenAI
            if not self.base_url:
                raise ValueError("LLM_BASE_URL is required for 'openai' provider (e.g. Colab/Ngrok URL)")
            
            self.client = OpenAI(
                api_key=self.api_key or "dummy", # Local models often ignore key
                base_url=self.base_url
            )
            self.model = self.model or "model" # Default for many local servers
            print(f"[LLMEngine] Initialized Generic OpenAI Client (URL: {self.base_url}, Model: {self.model})")

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
   - Extract: Pressure, Temperature, Solvent, Catalyst (if constant).
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
        "Solvent": "Toluene"
    }}
}}

## Text Content:
{text[:100000]} 
(Truncated if too long)
"""

    def _clean_json(self, content):
        """Helper to strip markdown ```json ... ```"""
        content = content.strip()
        if content.startswith("```"):
            # Remove first line
            content = content.split("\n", 1)[1]
            # Remove last line if ```
            if content.rstrip().endswith("```"):
                content = content.rstrip().rsplit("\n", 1)[0]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("[LLMEngine] JSON Decode Error. Raw content:")
            print(content)
            return {"term_mapping": {}, "global_conditions": {}}
