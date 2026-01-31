"""
RPAC Benchmark - LLM Client Abstraction
Supports OpenAI, Anthropic, Google AI, AWS Bedrock, OpenRouter, and Groq
"""

import os
import time
import functools
from abc import ABC, abstractmethod
from typing import Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip


def retry_with_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator that retries a function with exponential backoff on rate limit errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limit errors
                    is_rate_limit = any(x in error_str for x in [
                        'rate limit', 'rate_limit', '429', 'too many requests',
                        'resource exhausted', 'resourceexhausted', 'quota exceeded',
                        'throttl', 'retry after'
                    ])
                    
                    if is_rate_limit and attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"      ⏳ Rate limited. Waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        last_exception = e
                    else:
                        raise e
            raise last_exception
        return wrapper
    return decorator


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response for the given prompt."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model_id: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, api_key: str, model_id: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class GoogleClient(LLMClient):
    """Google AI (Gemini) API client."""
    
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        
        # Safely extract text from response
        # response.text can raise ValueError for some responses
        try:
            return response.text
        except ValueError:
            # Fallback: try to get text from parts
            if response.parts:
                return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            # Last resort: try candidates
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            raise ValueError(f"Could not extract text from Gemini response: {response}")


class BedrockClient(LLMClient):
    """AWS Bedrock API client (supports Claude, Llama, Mistral, etc.)."""
    
    def __init__(self, api_key: str = None, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", region: str = "us-east-1"):
        """
        Initialize Bedrock client.
        
        Args:
            api_key: Not used (Bedrock uses AWS credentials from environment/config)
            model_id: Bedrock model ID (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
            region: AWS region (default: us-east-1)
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 package not installed. Run: pip install boto3")
        
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        import json
        
        # Determine model family and format request accordingly
        if "anthropic" in self.model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        elif "meta" in self.model_id or "llama" in self.model_id.lower():
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        elif "mistral" in self.model_id:
            body = {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif "nova" in self.model_id.lower() or "amazon.nova" in self.model_id:
            # Amazon Nova uses the Converse API format
            body = {
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature
                }
            }
        elif "amazon" in self.model_id and "titan" in self.model_id.lower():
            # Amazon Titan uses textGenerationConfig
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            }
        else:
            # Default Claude-style format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        result = json.loads(response["body"].read())
        
        # Parse response based on model family
        if "anthropic" in self.model_id:
            return result["content"][0]["text"]
        elif "meta" in self.model_id or "llama" in self.model_id.lower():
            return result.get("generation", result.get("outputs", [{}])[0].get("text", ""))
        elif "mistral" in self.model_id:
            return result.get("outputs", [{}])[0].get("text", "")
        elif "nova" in self.model_id.lower() or "amazon.nova" in self.model_id:
            # Nova returns: {"output": {"message": {"content": [{"text": "..."}]}}}
            return result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        elif "amazon" in self.model_id and "titan" in self.model_id.lower():
            return result["results"][0]["outputText"]
        else:
            return result.get("content", [{}])[0].get("text", str(result))

class OpenRouterClient(LLMClient):
    """OpenRouter API client - access 100+ models through one API."""
    
    def __init__(self, api_key: str, model_id: str = "openai/gpt-4o"):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model_id: Model ID in format "provider/model" (e.g., openai/gpt-4o, anthropic/claude-3.5-sonnet)
        
        Popular models:
            - openai/gpt-4o, openai/gpt-4o-mini
            - anthropic/claude-3.5-sonnet, anthropic/claude-3-opus
            - meta-llama/llama-3.1-70b-instruct
            - mistralai/mistral-large
            - google/gemini-pro-1.5
            - qwen/qwen-2.5-72b-instruct
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


class GroqClient(LLMClient):
    """Groq API client - blazing fast inference for open models."""
    
    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model_id: Model ID
        
        Free tier models:
            - llama-3.3-70b-versatile (best quality)
            - llama-3.1-8b-instant (fastest)
            - mixtral-8x7b-32768 (good balance)
            - gemma2-9b-it (Google's Gemma)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_id = model_id
    
    @retry_with_backoff()
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


def create_client(provider: str, api_key: str, model_id: str) -> LLMClient:
    """Factory function to create LLM client based on provider."""
    providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "bedrock": BedrockClient,
        "openrouter": OpenRouterClient,
        "groq": GroqClient
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(providers.keys())}")
    
    return providers[provider](api_key=api_key, model_id=model_id)


def get_api_key(env_var: str) -> str:
    """Get API key from environment variable."""
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(f"Environment variable {env_var} not set")
    return key
