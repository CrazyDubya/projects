# harmonized_api_wrappers.py

import os
import json
import anthropic
from openai import OpenAI
from google.generativeai import GenerativeModel, types, configure
from monsterapi import client

class APIWrapper:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open("apis/config.json") as f:
            return json.load(f)

    def get_model_info(self, model_type, model_name):
        model_config = self.config.get(model_type, {}).get(model_name, {})
        return model_config

    def process_model(self, model_type, model_name, **kwargs):
        model_info = self.get_model_info(model_type, model_name)
        if not model_info:
            raise ValueError(f"Invalid model type or name: {model_type}/{model_name}")

        if model_type == "CLAUDE_MODELS":
            return self.process_claude_model(model_info, **kwargs)
        elif model_type == "MONSTER_MODELS":
            return self.process_monster_model(model_info, **kwargs)
        elif model_type == "OPENAI_MODELS":
            return self.process_openai_model(model_info, **kwargs)
        elif model_type == "PERPLEXITY_MODELS":
            return self.process_perplexity_model(model_info, **kwargs)
        elif model_type == "LOCAL_MODELS":
            return self.process_local_model(model_info, **kwargs)
        elif model_type == "GEMINI_MODELS":
            return self.process_gemini_model(model_info, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def process_claude_model(self, model_info, temperature, system_prompt, refined_input, max_tokens):
        api_key = os.environ.get("CLAUDE_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_info["name"],
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": refined_input}]
        )
        return response.content[0].text

    def process_monster_model(self, model_info, prompt, parameters):
        api_key = os.environ.get("MONSTER_API_KEY")
        client_instance = MonsterAPI(api_key)
        response = client_instance.get_response(model=model_info["name"], data={
            "prompt": prompt,
            **parameters
        })
        process_id = response["process_id"]
        result = client_instance.wait_and_get_result(process_id)
        return result

    def process_openai_model(self, model_info, system_prompt, refined_input, temperature, max_tokens):
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refined_input}
        ]
        response = client.chat.completions.create(
            model=model_info["name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def process_perplexity_model(self, model_info, system_prompt, refined_input, temperature, max_tokens):
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refined_input}
        ]
        response = client.chat.completions.create(
            model=model_info["name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def process_local_model(self, model_info, system_prompt, refined_input, temperature, max_tokens):
        base_url = f"http://localhost:{model_info['port']}/v1"
        client = OpenAI(api_key="NONE", base_url=base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refined_input}
        ]
        response = client.chat.completions.create(
            model=model_info["name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def process_gemini_model(self, model_info, system_prompt, refined_input, temperature, max_tokens):
        api_key = os.environ.get("GOOGLE_API_KEY")
        configure(api_key=api_key)
        model = GenerativeModel(model_info["name"])
        prompt = f"{system_prompt}\n{refined_input}"
        generation_config = types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text