#!/usr/bin/env python3
"""
Proper Nomic Game - Enhanced with Model Performance Tracking
- Comprehensive behavioral analysis and coherence tracking
- Cross-game model performance metrics and weighted assignment
- Anti-tie-breaking obsession system with semantic analysis
- Strategic engagement detection for quality control
- Multi-dimensional model scoring and classification
"""

import argparse
import json
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Union

import requests
from flask import Flask, jsonify, render_template_string, request

# Import Game Master system
from game_master import GameMaster, DynamicRuleEngine, LinguisticAnalyzer, ContextRestorationManager
from game_master_api import GameMasterEngine, create_game_master_engine


# LLM Provider Architecture for unified Ollama and OpenRouter support
class LLMProvider(Protocol):
    """Protocol for LLM providers (Ollama, OpenRouter, etc.)"""

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Generate text using the specified model"""
        ...

    def assign_ports(self, players: List["Player"]) -> None:
        """Assign ports/endpoints to players"""
        ...

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        ...


class OpenRouterClient:
    """OpenRouter API client for cloud-based LLM access"""

    def __init__(self, api_key: str, use_big_models: bool = False, use_hybrid: bool = False):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/anthropics/claude-code",
            "X-Title": "Nomic Game AI Evaluation",
            "Content-Type": "application/json",
        }

        # Small models (cost-efficient)
        self.small_models = [
            "google/gemini-2.5-flash",        # Gemini 2.5 Flash - cost-efficient
            "google/gemini-2.0-flash-001",    # Gemini 2.0 Flash - 1M context (fallback)
            "openai/gpt-4o-mini",             # GPT-4o Mini - 128k context
            "anthropic/claude-3.5-haiku",     # Claude 3.5 Haiku - 200k context
            "x-ai/grok-3-mini"                # Grok-3 Mini - cost-efficient
        ]

        # Standard OpenRouter models (verified IDs)
        self.standard_models = [
            "google/gemini-2.0-flash-001",  # Gemini 2.0 Flash - 1M context
            "openai/gpt-4o-mini",  # GPT-4o Mini - 128k context
            "anthropic/claude-3.5-haiku",  # Claude 3.5 Haiku - 200k context
            "anthropic/claude-3.5-sonnet",  # Claude 3.5 Sonnet - 200k context (larger than haiku)
        ]

        # Big models (premium SOTA models)
        self.big_models = [
            "anthropic/claude-3.5-sonnet",      # Claude 3.5 Sonnet - 200k context
            "openai/gpt-4o",                    # GPT-4o - 128k context  
            "google/gemini-2.5-pro-preview",    # Gemini 2.5 Pro - 1.05M context
            "x-ai/grok-3-beta"                  # Grok-3 Beta - TBD context
        ]

        # Model family pairings for hybrid mode
        self.model_families = {
            "claude": {
                "big": "anthropic/claude-3.5-sonnet",
                "small": "anthropic/claude-3.5-haiku"
            },
            "openai": {
                "big": "openai/gpt-4o", 
                "small": "openai/gpt-4o-mini"
            },
            "google": {
                "big": "google/gemini-2.5-pro-preview",
                "small": "google/gemini-2.5-flash"  # Prefer 2.5 Flash, fallback to 2.0
            },
            "xai": {
                "big": "x-ai/grok-3-beta",
                "small": "x-ai/grok-3-mini"
            }
        }

        # Select model set based on configuration
        self.use_big_models = use_big_models
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            # In hybrid mode, make both big and small models available
            self.available_models = list(set(self.big_models + self.small_models))
        else:
            self.available_models = self.big_models if use_big_models else self.standard_models

        # Cost tracking per model ($/1M tokens - verified from research)
        self.model_costs = {
            # Small models (cost-efficient)
            "google/gemini-2.5-flash": {"input": 0.30, "output": 1.20},  # Estimated similar to 2.0 Flash
            "google/gemini-2.0-flash-001": {"input": 0.35, "output": 1.05},
            "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "anthropic/claude-3.5-haiku": {"input": 0.25, "output": 1.25},
            "x-ai/grok-3-mini": {"input": 0.20, "output": 0.80},  # Estimated cost-efficient
            # Standard models
            "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
            # Big models (premium pricing)
            "openai/gpt-4o": {"input": 5.00, "output": 15.00},
            "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
            "x-ai/grok-3-beta": {"input": 3.00, "output": 15.00},  # TBD - will verify
        }
        
        # Maximum output tokens per model (based on API documentation)
        self.max_output_tokens = {
            # Small models
            "google/gemini-2.5-flash": 8192,  # Estimated similar to 2.0 Flash
            "google/gemini-2.0-flash-001": 8192,  # Confirmed 8192 max output tokens
            "openai/gpt-4o-mini": 16000,  # Documented 16k max output
            "anthropic/claude-3.5-haiku": 4096,  # Documented 4k max output
            "x-ai/grok-3-mini": 8192,  # Estimated standard output limit
            # Standard/Big models
            "anthropic/claude-3.5-sonnet": 8192,  # 8k with beta header, 4k without
            "openai/gpt-4o": 16000,  # Same as mini for output
            "google/gemini-2.5-pro-preview": 8192,  # Standard Gemini output limit
            "x-ai/grok-3-beta": 8192,  # Estimated standard output limit
        }
        
        # Context window sizes (for calculating safe output space)
        self.context_windows = {
            # Standard models
            "google/gemini-2.0-flash-001": 1000000,  # 1M context window
            "openai/gpt-4o-mini": 128000,  # 128k context window
            "anthropic/claude-3.5-haiku": 200000,  # 200k context window
            "anthropic/claude-3.5-sonnet": 200000,  # 200k context window
            # Big models
            "openai/gpt-4o": 128000,  # 128k context window
            "google/gemini-2.5-pro-preview": 1050000,  # 1.05M context window
            "x-ai/grok-3-beta": 256000,  # Estimated 256k context window
        }

        self.session_costs = {}  # Track costs per session
        self.session_budget = 5.0  # Default $5 budget limit
        self.budget_warnings_shown = set()  # Track which warnings we've already shown

    def calculate_safe_max_tokens(self, model: str, prompt: str) -> int:
        """Calculate safe max tokens based on prompt size and model limits"""
        # Rough token estimation: ~1 token per 4 characters
        prompt_tokens = len(prompt) // 4
        
        # Get model's context window and max output
        context_window = self.context_windows.get(model, 128000)
        max_output = self.max_output_tokens.get(model, 4096)
        
        # Calculate available space (use 90% of remaining context to be safe)
        available_space = int((context_window - prompt_tokens) * 0.90)
        
        # Return the smaller of: available space or model's max output
        return min(available_space, max_output)

    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Generate text using OpenRouter API"""
        # Calculate safe max tokens if not explicitly provided
        if "max_tokens" not in kwargs:
            safe_max = self.calculate_safe_max_tokens(model, prompt)
            kwargs["max_tokens"] = safe_max
        
        # Add special header for Claude 3.5 Sonnet to enable 8k output
        headers = self.headers.copy()
        if model == "anthropic/claude-3.5-sonnet":
            headers["anthropic-beta"] = "max-tokens-3-5-sonnet-2024-07-15"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Track usage and costs
                if "usage" in data:
                    self._track_usage(model, data["usage"])
                    # Check budget after tracking costs
                    self._check_budget_limits()

                return content
            else:
                print(f"OpenRouter API error {response.status_code}: {response.text}")
                return f"API Error: {response.status_code}"

        except requests.exceptions.Timeout:
            return "API Timeout: Request took too long"
        except Exception as e:
            print(f"OpenRouter error: {str(e)}")
            return f"API Error: {str(e)}"

    def assign_ports(self, players: List["Player"]) -> None:
        """Assign models to players for OpenRouter with smart rotation based on player count"""
        import random
        
        # Smart model assignment based on player count
        if len(players) == 4:
            # 4 players: use all 4 models (one each)
            models = self.available_models.copy()
        elif len(players) in [2, 3]:
            # 2-3 players: random selection from all models
            models = random.sample(self.available_models, len(players))
        else:
            # 5+ players: fill with each model, then random order
            models = self.available_models * ((len(players) // len(self.available_models)) + 1)
            models = models[:len(players)]
            random.shuffle(models)
        
        for i, player in enumerate(players):
            player.model = models[i]
            player.port = None  # No port needed for API calls
            
        print(f" Model assignments for this game: {dict(zip([f'Player {p.id}' for p in players], [p.model for p in players]))}")

    def get_available_models(self) -> List[str]:
        """Get list of available OpenRouter models"""
        return self.available_models.copy()

    def _track_usage(self, model: str, usage: dict):
        """Track API usage and costs"""
        if model not in self.session_costs:
            self.session_costs[model] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0}

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        if model in self.model_costs:
            input_cost = (input_tokens / 1000000) * self.model_costs[model]["input"]
            output_cost = (output_tokens / 1000000) * self.model_costs[model]["output"]
            total_cost = input_cost + output_cost

            self.session_costs[model]["input_tokens"] += input_tokens
            self.session_costs[model]["output_tokens"] += output_tokens
            self.session_costs[model]["total_cost"] += total_cost

    def get_session_costs(self) -> dict:
        """Get current session costs"""
        return self.session_costs.copy()
    
    def get_total_session_cost(self) -> float:
        """Get total cost for all models in current session"""
        total = 0.0
        for model_costs in self.session_costs.values():
            total += model_costs.get("total_cost", 0.0)
        return total
    
    def verify_api_models(self) -> dict:
        """Verify which models are available on OpenRouter API"""
        print("🔍 Verifying API model availability...")
        verification_results = {}
        
        # Simple test prompt for verification
        test_prompt = "Respond with exactly the word 'TEST' and nothing else."
        
        for model in self.available_models:
            try:
                print(f"  Testing {model}...")
                
                # Make a minimal request to test availability
                headers = self.headers.copy()
                if model == "anthropic/claude-3.5-sonnet":
                    headers["anthropic-beta"] = "max-tokens-3-5-sonnet-2024-07-15"
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10,
                    "stream": False,
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions", 
                    headers=headers, 
                    json=payload, 
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    
                    # Track tiny usage for verification
                    if "usage" in data:
                        self._track_usage(model, data["usage"])
                    
                    verification_results[model] = {
                        "available": True,
                        "response": content,
                        "context_window": self.context_windows.get(model, "Unknown"),
                        "max_output": self.max_output_tokens.get(model, "Unknown")
                    }
                    print(f"    ✅ {model} - Available (response: '{content}')")
                    
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    verification_results[model] = {
                        "available": False,
                        "error": error_msg,
                        "context_window": self.context_windows.get(model, "Unknown"),
                        "max_output": self.max_output_tokens.get(model, "Unknown")
                    }
                    print(f"    ❌ {model} - Error: {error_msg}")
                    
            except Exception as e:
                error_msg = str(e)
                verification_results[model] = {
                    "available": False,
                    "error": error_msg,
                    "context_window": self.context_windows.get(model, "Unknown"),
                    "max_output": self.max_output_tokens.get(model, "Unknown")
                }
                print(f"    ❌ {model} - Exception: {error_msg}")
        
        # Summary
        available_count = sum(1 for result in verification_results.values() if result["available"])
        total_count = len(verification_results)
        
        print(f"\n📊 Verification Summary: {available_count}/{total_count} models available")
        if available_count == 0:
            print("⚠️  WARNING: No models are available! Check your OpenRouter API key and credits.")
        elif available_count < total_count:
            print("⚠️  Some models unavailable - game will use available models only")
        else:
            print("✅ All models verified and ready!")
            
        return verification_results
    
    def _check_budget_limits(self):
        """Check if we're approaching or exceeding budget limits"""
        total_cost = self.get_total_session_cost()
        
        # Budget warning thresholds
        warning_50 = self.session_budget * 0.5  # 50% warning
        warning_80 = self.session_budget * 0.8  # 80% warning
        
        # Check for budget warnings
        if total_cost >= self.session_budget:
            if "over_budget" not in self.budget_warnings_shown:
                print(f"\n🚨 BUDGET EXCEEDED: ${total_cost:.4f} / ${self.session_budget:.2f}")
                print("💰 Session budget limit reached!")
                response = input("   Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("🛑 Session stopped due to budget limit")
                    raise Exception("Budget limit exceeded - user chose to stop")
                else:
                    print("⚠️  Continuing with budget exceeded...")
                self.budget_warnings_shown.add("over_budget")
                
        elif total_cost >= warning_80:
            if "warning_80" not in self.budget_warnings_shown:
                print(f"\n⚠️  Budget Warning: ${total_cost:.4f} / ${self.session_budget:.2f} (80%)")
                self.budget_warnings_shown.add("warning_80")
                
        elif total_cost >= warning_50:
            if "warning_50" not in self.budget_warnings_shown:
                print(f"\n💡 Budget Notice: ${total_cost:.4f} / ${self.session_budget:.2f} (50%)")
                self.budget_warnings_shown.add("warning_50")

    def validate_rule(self, rule_text: str, context: str):
        """Validate rule using OpenRouter API (compatible with OllamaManager interface)"""
        prompt = f"""RULE VALIDATION TASK

PROPOSED RULE: "{rule_text}"

CURRENT GAME CONTEXT:
{context}

Please validate this rule for:
1. CLARITY: Is the rule clearly written and unambiguous?
2. COMPLETENESS: Does it specify all necessary details?
3. CONSISTENCY: Does it conflict with existing rules?
4. ENFORCEABILITY: Can this rule be properly implemented?

Respond with:
VALID: true/false
ISSUES: [list any problems]
SUGGESTION: [how to improve if invalid]"""

        try:
            # Use a small, fast model for validation
            response = self.generate("openai/gpt-4o-mini", prompt, temperature=0.3, max_tokens=500)

            # Parse response
            valid = "VALID: true" in response or "true" in response.lower()

            return {"valid": valid, "issues": response, "raw_response": response}
        except Exception as e:
            # If validation fails, default to valid to keep game running
            return {"valid": True, "issues": f"Validation error: {str(e)}", "raw_response": "Error during validation"}


class OllamaClient:
    """Existing Ollama client wrapped in new interface"""

    def __init__(self, base_port: int = 11435):
        self.base_port = base_port
        self.validator_port = base_port  # First port for validator
        self.base_ports = list(range(base_port + 1, base_port + 8))  # Next 7 ports for players
        self.assigned_ports = {}

    def generate(self, model: str, prompt: str, port: int = 11434, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"http://localhost:{port}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 200),
                    },
                },
                timeout=10,
            )
            return response.json().get("response", "Error generating response")
        except Exception as e:
            return f"Mock response for {model} on port {port} (Error: {str(e)})"

    def assign_ports(self, players: List["Player"]) -> None:
        """Assign ports to players for Ollama"""
        models = ["llama3.2:3b", "gemma2:2b", "qwen2.5:1.5b", "smollm2:1.7b", "llama3.2:1b", "qwen2.5:3b"]

        for i, player in enumerate(players):
            player.model = models[i % len(models)]
            player.port = self.base_ports[i + 1] if i + 1 < len(self.base_ports) else 11434
            self.assigned_ports[player.id] = player.port

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return ["llama3.2:3b", "gemma2:2b", "qwen2.5:1.5b", "smollm2:1.7b", "llama3.2:1b", "qwen2.5:3b"]

    def validate_rule(self, rule_text: str, context: str):
        """Validate rule using Ollama (compatible interface)"""
        prompt = f"""RULE VALIDATION TASK

PROPOSED RULE: "{rule_text}"

CURRENT GAME CONTEXT:
{context}

Please validate this rule for clarity, completeness, and consistency.

Respond with:
VALID: true/false
ISSUES: [list any problems]"""

        try:
            # Use validator port for rule validation
            response = self.generate("llama3.2:3b", prompt, self.validator_port, temperature=0.3, max_tokens=200)

            # Parse response
            valid = "VALID: true" in response or "true" in response.lower()

            return {"valid": valid, "issues": response, "raw_response": response}
        except Exception as e:
            # If validation fails, default to valid to keep game running
            return {"valid": True, "issues": f"Validation error: {str(e)}", "raw_response": "Error during validation"}


class InputSanitizer:
    """Comprehensive input sanitization for security and context window optimization"""

    def __init__(self):
        # ASCII characters + basic punctuation only
        self.allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:()'\"/-\n\r\t")

        # Character limits optimized for context windows
        self.max_rule_length = 500
        self.max_explanation_length = 300
        self.max_total_input = 800

        # Dangerous patterns that could indicate attacks
        self.dangerous_patterns = [
            # SQL injection patterns
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s",
            r'[\'";]',
            r"--",
            r"/\*.*?\*/",
            # XSS patterns
            r"<script.*?>",
            r"javascript:",
            r"onclick=",
            r"onerror=",
            # LLM prompt injection patterns
            r"(?i)(ignore|forget|disregard)\s+(previous|above|all)\s+(instructions?|prompts?)",
            r"(?i)system\s*:",
            r"(?i)assistant\s*:",
            r"(?i)human\s*:",
            r"(?i)pretend\s+(you|to)\s+(are|be)",
            r"(?i)act\s+as\s+(if|though)",
            r"(?i)role\s*play",
            r"(?i)new\s+(instructions?|rules?|system)",
            r"\[INST\]|\[/INST\]",
            r"<\|.*?\|>",
        ]

    def sanitize_text(self, text: str, max_length: int = None) -> str:
        """Sanitize text input with ASCII filtering and length limits"""
        if not text:
            return ""

        # Convert to string if not already
        text = str(text)

        # Filter to allowed characters only
        sanitized = "".join(char for char in text if char in self.allowed_chars)

        # Apply length limit
        if max_length:
            sanitized = sanitized[:max_length]

        # Strip whitespace
        sanitized = sanitized.strip()

        return sanitized

    def sanitize_rule_text(self, rule_text: str) -> dict:
        """Sanitize rule text with comprehensive security checks"""
        if not rule_text:
            return {"success": False, "error": "Rule text cannot be empty"}

        # Basic sanitization
        sanitized = self.sanitize_text(rule_text, self.max_rule_length)

        if not sanitized:
            return {"success": False, "error": "Rule text contains only invalid characters"}

        if len(sanitized) < 10:
            return {"success": False, "error": "Rule text too short (minimum 10 characters)"}

        # Check for dangerous patterns
        security_check = self.check_security_patterns(sanitized)
        if not security_check["safe"]:
            return {"success": False, "error": f"Security violation: {security_check['reason']}"}

        return {
            "success": True,
            "sanitized": sanitized,
            "original_length": len(rule_text),
            "final_length": len(sanitized),
        }

    def sanitize_explanation(self, explanation: str) -> dict:
        """Sanitize explanation text"""
        if not explanation:
            return {"success": False, "error": "Explanation cannot be empty"}

        sanitized = self.sanitize_text(explanation, self.max_explanation_length)

        if not sanitized:
            return {"success": False, "error": "Explanation contains only invalid characters"}

        if len(sanitized) < 5:
            return {"success": False, "error": "Explanation too short (minimum 5 characters)"}

        # Check for dangerous patterns
        security_check = self.check_security_patterns(sanitized)
        if not security_check["safe"]:
            return {"success": False, "error": f"Security violation: {security_check['reason']}"}

        return {
            "success": True,
            "sanitized": sanitized,
            "original_length": len(explanation),
            "final_length": len(sanitized),
        }

    def check_security_patterns(self, text: str) -> dict:
        """Check text for dangerous security patterns"""
        import re

        for pattern in self.dangerous_patterns:
            if re.search(pattern, text):
                return {"safe": False, "reason": "Potentially malicious content detected"}

        # Check for excessive special characters (could indicate obfuscation)
        special_char_count = sum(1 for char in text if not char.isalnum() and char != " ")
        if special_char_count > len(text) * 0.3:  # More than 30% special characters
            return {"safe": False, "reason": "Excessive special characters"}

        return {"safe": True, "reason": "Content appears safe"}

    def validate_total_input_size(self, rule_text: str, explanation: str) -> dict:
        """Validate total input size doesn't exceed limits"""
        total_length = len(rule_text) + len(explanation)

        if total_length > self.max_total_input:
            return {"valid": False, "error": f"Total input too long ({total_length}/{self.max_total_input} characters)"}

        return {"valid": True, "total_length": total_length}


class ContextWindowManager:
    """Context window optimization and token tracking for different LLM models"""

    def __init__(self):
        # Model-specific context limits (tokens) - synchronized with OpenRouterManager
        self.model_limits = {
            # Standard models
            "google/gemini-2.0-flash-001": 1000000,  # Gemini 2.0 Flash - 1M context (matches OpenRouter)
            "openai/gpt-4o-mini": 128000,  # GPT-4o Mini - 128k context
            "anthropic/claude-3.5-haiku": 200000,  # Claude 3.5 Haiku - 200k context
            "anthropic/claude-3.5-sonnet": 200000,  # Claude 3.5 Sonnet - 200k context
            "anthropic/claude-3-haiku": 200000,  # Claude 3 Haiku - 200k context
            # Big models
            "openai/gpt-4o": 128000,  # GPT-4o - 128k context
            "google/gemini-2.5-pro-preview": 1050000,  # Gemini 2.5 Pro - 1.05M context
            "x-ai/grok-3-beta": 256000,  # Grok 3 Beta - 256k context (added missing model)
            # Legacy support
            "xai/grok-3-mini": 1000000,  # Grok 3 Mini - 1M context (legacy naming)
            "default": 8000,  # Conservative default for local models
        }

        # Approximate tokens per character (rough estimate)
        self.chars_per_token = 4  # English text averages ~4 chars per token

        # Context usage tracking
        self.current_usage = {}  # model -> current token count
        self.usage_history = []  # Historical usage for optimization

        # Content priorities for trimming (higher = keep longer)
        self.content_priorities = {
            "current_rules": 10,  # Always keep current rules
            "recent_proposals": 8,  # Recent proposals are important
            "current_turn": 9,  # Current turn context is critical
            "player_states": 6,  # Player internal states
            "game_history": 4,  # Game history can be trimmed
            "idle_thoughts": 2,  # Idle thoughts can be removed first
        }
        
        # History reservation settings - always reserve a portion for historical context
        self.history_reservation_percentage = 0.15  # Reserve 15% of context for history
        self.minimum_history_tokens = 2000  # Minimum tokens to reserve for history
        self.maximum_history_tokens = 50000  # Maximum tokens to use for history
        
        # Context degradation detection settings
        self.degradation_threshold = 0.95  # Consider degraded if consistently above 95%
        self.degradation_window = 5  # Number of consecutive high-usage events to trigger
        self.player_degradation_status = {}  # player_id -> degradation tracking
        self.emergency_resets_per_game = {}  # Track emergency resets per player

    def get_model_limit(self, model: str) -> int:
        """Get context window limit for a specific model"""
        return self.model_limits.get(model, self.model_limits["default"])

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length"""
        if not text:
            return 0
        return max(1, len(text) // self.chars_per_token)

    def get_context_usage(self, model: str) -> dict:
        """Get current context usage for a model"""
        usage = self.current_usage.get(model, 0)
        limit = self.get_model_limit(model)
        percentage = (usage / limit * 100) if limit > 0 else 0

        return {
            "model": model,
            "current_tokens": usage,
            "limit_tokens": limit,
            "usage_percentage": round(percentage, 1),
            "remaining_tokens": limit - usage,
            "status": "critical" if percentage > 90 else "warning" if percentage > 75 else "normal",
        }

    def track_usage(self, model: str, prompt: str, response: str = ""):
        """Track token usage for a model"""
        prompt_tokens = self.estimate_tokens(prompt)
        response_tokens = self.estimate_tokens(response)
        total_tokens = prompt_tokens + response_tokens

        if model not in self.current_usage:
            self.current_usage[model] = 0

        self.current_usage[model] += total_tokens

        # Log usage for optimization
        self.usage_history.append(
            {
                "timestamp": datetime.now(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens,
                "cumulative_usage": self.current_usage[model],
            }
        )

        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "cumulative_usage": self.current_usage[model],
        }

    def reset_usage(self, models: List[str] = None):
        """Reset context usage for specified models or all models"""
        if models is None:
            # Reset all models
            self.current_usage.clear()
            print(" Context usage reset for all models")
        else:
            # Reset specific models
            reset_count = 0
            for model in models:
                if model in self.current_usage:
                    del self.current_usage[model]
                    reset_count += 1
            print(f" Context usage reset for {reset_count} models")
        
        # Add reset event to history
        self.usage_history.append({
            "timestamp": datetime.now(),
            "event": "context_reset",
            "models_reset": models or "all",
            "action": "Manual context reset - starting fresh"
        })

    def reset_for_new_game(self):
        """Reset context usage when starting a new game"""
        previous_usage = self.current_usage.copy()
        self.reset_usage()
        
        # Log previous game usage for analytics
        if previous_usage:
            total_previous = sum(previous_usage.values())
            print(f" Previous game context usage: {total_previous:,} tokens across {len(previous_usage)} models")
        
        print(" ✅ Context reset for new game - starting fresh")

    def track_degradation(self, model: str, player_id: int, usage_percentage: float) -> dict:
        """Track context degradation for a specific player and detect intervention needs"""
        if player_id not in self.player_degradation_status:
            self.player_degradation_status[player_id] = {
                "consecutive_high_usage": 0,
                "total_degradation_events": 0,
                "last_intervention": None,
                "degradation_history": [],
                "model": model
            }
        
        status = self.player_degradation_status[player_id]
        
        # Track this usage event
        is_degraded = usage_percentage >= (self.degradation_threshold * 100)
        status["degradation_history"].append({
            "timestamp": datetime.now(),
            "usage_percentage": usage_percentage,
            "is_degraded": is_degraded,
            "model": model
        })
        
        # Keep only recent history (last 10 events)
        status["degradation_history"] = status["degradation_history"][-10:]
        
        if is_degraded:
            status["consecutive_high_usage"] += 1
            status["total_degradation_events"] += 1
        else:
            status["consecutive_high_usage"] = 0  # Reset counter on good usage
        
        # Determine if intervention is needed
        needs_intervention = (
            status["consecutive_high_usage"] >= self.degradation_window and
            (status["last_intervention"] is None or 
             (datetime.now() - status["last_intervention"]).total_seconds() > 1800)  # 30 min cooldown
        )
        
        intervention_urgency = "none"
        if needs_intervention:
            if status["consecutive_high_usage"] >= self.degradation_window * 2:
                intervention_urgency = "critical"  # Severe degradation
            else:
                intervention_urgency = "moderate"  # Standard intervention
        
        return {
            "player_id": player_id,
            "model": model,
            "current_usage": usage_percentage,
            "consecutive_high_usage": status["consecutive_high_usage"],
            "total_degradation_events": status["total_degradation_events"],
            "needs_intervention": needs_intervention,
            "intervention_urgency": intervention_urgency,
            "time_since_last_intervention": (
                (datetime.now() - status["last_intervention"]).total_seconds()
                if status["last_intervention"] else None
            )
        }

    def mark_intervention_applied(self, player_id: int):
        """Mark that an intervention has been applied to a player"""
        if player_id in self.player_degradation_status:
            self.player_degradation_status[player_id]["last_intervention"] = datetime.now()
            self.player_degradation_status[player_id]["consecutive_high_usage"] = 0
            
            # Track emergency resets
            if player_id not in self.emergency_resets_per_game:
                self.emergency_resets_per_game[player_id] = 0
            self.emergency_resets_per_game[player_id] += 1
            
            print(f" 🚨 Context intervention applied to player {player_id} (#{self.emergency_resets_per_game[player_id]} this game)")

    def get_degradation_summary(self) -> dict:
        """Get summary of all player degradation status"""
        summary = {
            "players_needing_intervention": [],
            "players_at_risk": [],
            "total_interventions_this_game": sum(self.emergency_resets_per_game.values()),
            "degradation_overview": {}
        }
        
        for player_id, status in self.player_degradation_status.items():
            recent_avg = sum(h["usage_percentage"] for h in status["degradation_history"][-3:]) / max(1, len(status["degradation_history"][-3:]))
            
            player_summary = {
                "player_id": player_id,
                "model": status["model"],
                "consecutive_high_usage": status["consecutive_high_usage"],
                "recent_average_usage": round(recent_avg, 1),
                "total_degradation_events": status["total_degradation_events"],
                "interventions_this_game": self.emergency_resets_per_game.get(player_id, 0)
            }
            
            summary["degradation_overview"][player_id] = player_summary
            
            if status["consecutive_high_usage"] >= self.degradation_window:
                summary["players_needing_intervention"].append(player_id)
            elif status["consecutive_high_usage"] >= self.degradation_window // 2:
                summary["players_at_risk"].append(player_id)
        
        return summary

    def optimize_prompt(self, model: str, prompt: str, game_context: dict) -> str:
        """Optimize prompt length for model's context window"""
        usage = self.get_context_usage(model)

        # If usage is normal, return as-is
        if usage["usage_percentage"] < 75:
            return prompt

        # If critical, use emergency reduction
        if usage["usage_percentage"] > 95:
            return self._emergency_context_reduction(prompt, game_context)

        # If very high, use intelligent reduction with higher compression
        if usage["usage_percentage"] > 85:
            return self.intelligent_context_reduction(prompt, game_context, target_reduction=0.5, model=model)

        # If warning level, use moderate intelligent reduction
        return self.intelligent_context_reduction(prompt, game_context, target_reduction=0.3, model=model)

    def _moderate_trim(self, prompt: str, game_context: dict, remaining_tokens: int) -> str:
        """Moderate context trimming"""
        target_tokens = int(remaining_tokens * 0.8)  # Use 80% of remaining space

        # Trim lower priority content first
        sections = self._split_prompt_sections(prompt)
        optimized_sections = []
        current_tokens = 0

        # Sort sections by priority
        sorted_sections = sorted(sections.items(), key=lambda x: self.content_priorities.get(x[0], 5), reverse=True)

        for section_name, section_content in sorted_sections:
            section_tokens = self.estimate_tokens(section_content)
            if current_tokens + section_tokens <= target_tokens:
                optimized_sections.append(section_content)
                current_tokens += section_tokens
            else:
                # Partially include high-priority sections
                if self.content_priorities.get(section_name, 5) >= 7:
                    remaining_space = target_tokens - current_tokens
                    if remaining_space > 100:  # Only if meaningful space left
                        truncated = section_content[: remaining_space * self.chars_per_token]
                        optimized_sections.append(truncated + "... [truncated]")
                        break

        return "\n\n".join(optimized_sections)

    def _aggressive_trim(self, prompt: str, game_context: dict, remaining_tokens: int) -> str:
        """Aggressive context trimming for critical usage"""
        target_tokens = int(remaining_tokens * 0.5)  # Use 50% of remaining space (more aggressive)

        # Emergency compression for critical context situations
        if remaining_tokens < 1000:
            return self._emergency_context_reduction(prompt, game_context)

        # Keep only highest priority content
        sections = self._split_prompt_sections(prompt)
        essential_sections = []
        current_tokens = 0

        # Only keep priority 9+ content (raised threshold)
        for section_name, section_content in sections.items():
            if self.content_priorities.get(section_name, 5) >= 9:
                section_tokens = self.estimate_tokens(section_content)
                if current_tokens + section_tokens <= target_tokens:
                    essential_sections.append(section_content)
                    current_tokens += section_tokens

        trimmed_prompt = "\n\n".join(essential_sections)

        # Add context warning
        warning = "[CONTEXT CRITICALLY OPTIMIZED: Major game history removed due to token limits]"
        return f"{warning}\n\n{trimmed_prompt}"
    
    def _emergency_context_reduction(self, prompt: str, game_context: dict) -> str:
        """Emergency context reduction for critical failures"""
        # Extract absolute essentials only
        game = game_context.get("game")
        if not game:
            return "EMERGENCY MODE: Minimal context due to critical token shortage."
        
        # Minimal essential context
        essential_parts = []
        
        # Current turn and player info
        current_turn = getattr(game, 'turn_number', 1)
        essential_parts.append(f"TURN {current_turn} - EMERGENCY CONTEXT MODE")
        
        # Only current rules that are actively affecting the game
        if hasattr(game, 'rules'):
            mutable_count = len(game.rules.get("mutable", []))
            immutable_count = len(game.rules.get("immutable", []))
            essential_parts.append(f"RULES: {mutable_count} mutable, {immutable_count} immutable")
        
        # Current player status only
        if hasattr(game, 'players'):
            player_info = []
            for p in game.players[:4]:  # Only show first 4 players
                player_info.append(f"{p.name}: {p.points}pts")
            essential_parts.append(f"PLAYERS: {', '.join(player_info)}")
        
        # Critical game state
        essential_parts.append("TASK: Generate valid proposal in format INTERNAL_THOUGHT: / RULE_TEXT: / EXPLANATION:")
        
        return "\n\n".join(essential_parts)

    def _split_prompt_sections(self, prompt: str) -> dict:
        """Split prompt into sections for selective trimming"""
        # Enhanced prompt parsing to better identify sections
        sections = {}
        
        # Look for common Nomic prompt markers
        if "NOMIC STRATEGIC REALITY" in prompt:
            # This is a main game context - treat as high priority
            sections["current_rules"] = prompt
        elif "DELIBERATION" in prompt or "PROPOSAL" in prompt:
            # Deliberation/proposal prompt - high priority
            sections["current_turn"] = prompt
        elif "VOTING" in prompt:
            # Voting prompt - high priority  
            sections["current_turn"] = prompt
        elif "REFLECTION" in prompt:
            # End game reflection - medium priority
            sections["player_states"] = prompt
        else:
            # Unknown prompt type - treat conservatively
            if len(prompt) <= 2000:
                # Short prompt - keep it all as high priority
                sections["current_turn"] = prompt
            else:
                # Long prompt - split more intelligently
                sections["current_turn"] = prompt[:2000]  # Increased from 1000
                sections["game_history"] = prompt[2000:]
        
        return sections

    def get_usage_analytics(self) -> dict:
        """Get usage analytics for all models"""
        analytics = {"models": {}, "total_requests": len(self.usage_history), "optimization_events": 0}

        for model in self.current_usage:
            usage = self.get_context_usage(model)
            model_history = [h for h in self.usage_history if h["model"] == model]

            analytics["models"][model] = {
                **usage,
                "total_requests": len(model_history),
                "avg_tokens_per_request": (
                    sum(h["total_tokens"] for h in model_history) / len(model_history) if model_history else 0
                ),
                "peak_usage": max((h["cumulative_usage"] for h in model_history), default=0),
            }

        return analytics

    def reset_usage(self, model: str = None):
        """Reset usage tracking for a model or all models"""
        if model:
            self.current_usage[model] = 0
        else:
            self.current_usage.clear()

    def compress_turn_history(self, game_context: dict, num_turns: int = 5) -> str:
        """Compress recent turn history into a concise summary"""
        if not game_context.get("game"):
            return "No game context available for compression."
        
        game = game_context["game"]
        
        # Extract recent turn information
        recent_events = []
        if hasattr(game, 'events'):
            recent_events = game.events[-num_turns * 5:]  # Approximate events per turn
        
        # Get recent proposals
        recent_proposals = []
        if hasattr(game, 'proposals'):
            recent_proposals = game.proposals[-num_turns:]
        
        # Compress into summary
        summary_parts = []
        
        # Current game status
        if hasattr(game, 'turn_number'):
            summary_parts.append(f"TURN SUMMARY (Last {num_turns} turns from turn {game.turn_number}):")
        
        # Successful proposals summary
        successful_rules = []
        failed_proposals = []
        
        for proposal in recent_proposals:
            if hasattr(proposal, 'passed') and proposal.passed:
                successful_rules.append(f"Rule {proposal.id}: {proposal.rule_text[:60]}...")
            else:
                failed_proposals.append(f"Failed by {proposal.player_id}: {proposal.rule_text[:40]}...")
        
        if successful_rules:
            summary_parts.append(f"✅ NEW RULES ({len(successful_rules)}): " + "; ".join(successful_rules))
        
        if failed_proposals:
            summary_parts.append(f"❌ FAILED PROPOSALS ({len(failed_proposals)}): " + "; ".join(failed_proposals[:3]))
        
        # Player scoring trends
        if hasattr(game, 'players'):
            score_summary = []
            for player in game.players[:4]:  # Limit to 4 players
                score_summary.append(f"{player.name}: {player.points}pts")
            summary_parts.append(f"📊 CURRENT SCORES: {', '.join(score_summary)}")
        
        return "\n".join(summary_parts)

    def create_hierarchical_summary(self, game_context: dict) -> dict:
        """Create a hierarchical summary with different levels of detail"""
        if not game_context.get("game"):
            return {"error": "No game context available"}
        
        game = game_context["game"]
        
        # Level 1: Essential game state (highest priority)
        level1 = {
            "current_turn": getattr(game, 'turn_number', 1),
            "total_rules": len(getattr(game, 'rules', {}).get('mutable', [])) + len(getattr(game, 'rules', {}).get('immutable', [])),
            "leader_points": max([p.points for p in getattr(game, 'players', [])]) if hasattr(game, 'players') and game.players else 0
        }
        
        # Level 2: Recent activity (medium priority)
        level2 = {
            "recent_turns_summary": self.compress_turn_history(game_context, 3),
            "active_players": len(getattr(game, 'players', [])),
            "last_proposal": getattr(game.proposals[-1], 'rule_text', 'None')[:100] if hasattr(game, 'proposals') and game.proposals else "None"
        }
        
        # Level 3: Historical context (lowest priority)
        level3 = {
            "full_turn_summary": self.compress_turn_history(game_context, 10),
            "total_events": len(getattr(game, 'events', [])),
            "game_phase": self._determine_game_phase(game)
        }
        
        return {
            "priority_1_essential": level1,
            "priority_2_recent": level2, 
            "priority_3_historical": level3
        }

    def _determine_game_phase(self, game) -> str:
        """Determine what phase of the game we're in"""
        if not hasattr(game, 'players') or not game.players:
            return "initialization"
        
        max_points = max([p.points for p in game.players])
        turn_num = getattr(game, 'turn_number', 1)
        
        if max_points < 0:
            return "crisis"
        elif turn_num <= 3:
            return "opening"
        elif max_points < 70:
            return "midgame"
        elif max_points < 90:
            return "lategame"
        else:
            return "endgame"

    def intelligent_context_reduction(self, prompt: str, game_context: dict, target_reduction: float = 0.3, model: str = None) -> str:
        """Intelligently reduce context using hierarchical summaries and history reservation"""
        # Calculate context budget with history reservation
        current_tokens = self.estimate_tokens(prompt)
        target_tokens = int(current_tokens * (1 - target_reduction))
        
        # Calculate history reservation
        model_limit = self.get_model_limit(model) if model else current_tokens
        reserved_history_tokens = max(
            self.minimum_history_tokens,
            min(
                self.maximum_history_tokens,
                int(model_limit * self.history_reservation_percentage)
            )
        )
        
        # Adjust target tokens to account for history reservation
        available_for_current = target_tokens - reserved_history_tokens
        if available_for_current < current_tokens * 0.3:  # Safety check
            available_for_current = int(current_tokens * 0.3)
            reserved_history_tokens = target_tokens - available_for_current
        
        # Create hierarchical summary
        hierarchical = self.create_hierarchical_summary(game_context)
        
        # Build optimized prompt using hierarchical approach with reserved history
        optimized_sections = []
        
        # Always include level 1 (essential) - current context priority
        level1_text = f"ESSENTIAL GAME STATE: Turn {hierarchical['priority_1_essential']['current_turn']}, {hierarchical['priority_1_essential']['total_rules']} rules, leader at {hierarchical['priority_1_essential']['leader_points']} points"
        optimized_sections.append(level1_text)
        current_size = self.estimate_tokens(level1_text)
        
        # Include level 2 if space allows in current context budget
        level2_text = hierarchical['priority_2_recent']['recent_turns_summary']
        level2_tokens = self.estimate_tokens(level2_text)
        if current_size + level2_tokens <= available_for_current:
            optimized_sections.append(level2_text)
            current_size += level2_tokens
        
        # Dedicate reserved space to historical context (level 3)
        level3_text = hierarchical['priority_3_historical']['full_turn_summary']
        level3_tokens = self.estimate_tokens(level3_text)
        
        if level3_tokens <= reserved_history_tokens:
            # Full history fits in reservation
            optimized_sections.append(level3_text)
            history_used = level3_tokens
        else:
            # Trim history to fit reservation
            history_compression_ratio = reserved_history_tokens / level3_tokens
            compressed_history = self._compress_historical_context(
                level3_text, history_compression_ratio, game_context
            )
            optimized_sections.append(compressed_history)
            history_used = self.estimate_tokens(compressed_history)
        
        # Add enhanced context reduction notice
        total_reduction = 1 - (current_size + history_used) / current_tokens
        reduction_notice = f"[CONTEXT OPTIMIZED: {total_reduction:.1%} reduction | Current: {current_size:,} tokens | History: {history_used:,} tokens | Reserved: {reserved_history_tokens:,}]"
        optimized_sections.insert(0, reduction_notice)
        
        return "\n\n".join(optimized_sections)

    def _compress_historical_context(self, history_text: str, compression_ratio: float, game_context: dict) -> str:
        """Compress historical context to fit within reserved space"""
        target_length = int(len(history_text) * compression_ratio)
        
        if target_length <= 200:
            # Very aggressive compression - keep only key facts
            return f"COMPRESSED HISTORY: Game in progress, multiple rules proposed, current strategic situation developing."
        
        # Split into sentences and prioritize key information
        sentences = history_text.split('. ')
        important_sentences = []
        
        # Prioritize sentences with key game information
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # High priority patterns
            if any(keyword in sentence.lower() for keyword in ['rule', 'points', 'proposal', 'vote', 'win', 'leader']):
                important_sentences.append(sentence)
            # Medium priority patterns  
            elif any(keyword in sentence.lower() for keyword in ['turn', 'player', 'strategy']):
                if len('. '.join(important_sentences)) < target_length * 0.7:
                    important_sentences.append(sentence)
        
        # Combine and trim to target length
        compressed = '. '.join(important_sentences)
        if len(compressed) > target_length:
            compressed = compressed[:target_length-3] + "..."
        
        return f"HISTORY SUMMARY: {compressed}"

    def create_rolling_context_window(self, game_context: dict, window_size: int = 5, token_budget: int = None) -> str:
        """Create an adaptive rolling window of the most recent turns for consistent context"""
        if not game_context.get("game"):
            return "No game context available for rolling window."
        
        game = game_context["game"]
        current_turn = getattr(game, 'turn_number', 1)
        
        # Calculate adaptive window bounds based on token budget
        if token_budget:
            # Estimate tokens per turn and adjust window size accordingly
            estimated_tokens_per_turn = 400  # Conservative estimate
            max_turns_for_budget = max(3, token_budget // estimated_tokens_per_turn)
            window_size = min(window_size, max_turns_for_budget)
        
        window_start = max(1, current_turn - window_size + 1)
        window_end = current_turn
        
        # Build rolling context with priority-based content
        context_parts = []
        current_token_usage = 0
        
        # Priority 1: Current game state (always include)
        header = f"🎯 ADAPTIVE ROLLING WINDOW (T{window_start}-{window_end})"
        context_parts.append(header)
        current_token_usage += self.estimate_tokens(header)
        
        # Priority 2: Current standings
        if hasattr(game, 'players'):
            sorted_players = sorted(game.players, key=lambda p: p.points, reverse=True)
            standings = []
            for i, player in enumerate(sorted_players, 1):
                standings.append(f"{i}. {player.name}: {player.points}pts")
            standings_text = f"📊 STANDINGS: {' | '.join(standings[:4])}"
            
            if not token_budget or current_token_usage + self.estimate_tokens(standings_text) <= token_budget:
                context_parts.append(standings_text)
                current_token_usage += self.estimate_tokens(standings_text)
        
        # Priority 3: Recent rule changes in window (most critical for strategy)
        if hasattr(game, 'proposals'):
            recent_rules = []
            for proposal in game.proposals:
                if hasattr(proposal, 'turn_number') and window_start <= proposal.turn_number <= window_end:
                    status = "✅" if getattr(proposal, 'passed', False) else "❌"
                    # Adaptive rule text length based on available space
                    rule_length = 50 if not token_budget else min(50, max(20, (token_budget - current_token_usage) // 10))
                    recent_rules.append(f"T{proposal.turn_number}: {status} {proposal.rule_text[:rule_length]}...")
            
            if recent_rules:
                # Show most recent rules first, truncate if needed
                display_rules = recent_rules[-3:] if len(recent_rules) > 3 else recent_rules
                rules_text = f"📜 RECENT RULES ({len(recent_rules)}): {' | '.join(display_rules)}"
                
                if not token_budget or current_token_usage + self.estimate_tokens(rules_text) <= token_budget:
                    context_parts.append(rules_text)
                    current_token_usage += self.estimate_tokens(rules_text)
        
        # Priority 4: Key strategic events (if space allows)
        if hasattr(game, 'events'):
            key_events = []
            # Look for strategic events: point changes, voting patterns, rule effects
            for event in game.events[-window_size * 5:]:  # More events for better coverage
                if isinstance(event, str):
                    event_lower = event.lower()
                    if any(keyword in event_lower for keyword in [
                        'passed', 'failed', 'gained', 'lost', 'points', 'leads', 'won', 'transmut', 'amend'
                    ]):
                        event_text = event[:40] + "..." if len(event) > 40 else event
                        key_events.append(event_text)
            
            if key_events:
                # Show most recent strategic events
                display_events = key_events[-2:] if len(key_events) > 2 else key_events
                events_text = f"⚡ KEY EVENTS: {' | '.join(display_events)}"
                
                if not token_budget or current_token_usage + self.estimate_tokens(events_text) <= token_budget:
                    context_parts.append(events_text)
                    current_token_usage += self.estimate_tokens(events_text)
        
        # Priority 5: Context metadata (always include if space)
        total_rules = len(getattr(game, 'rules', {}).get('mutable', [])) + len(getattr(game, 'rules', {}).get('immutable', []))
        meta_text = f"🔧 META: Turn {current_turn}, {total_rules} rules, {current_token_usage} tokens used"
        
        if not token_budget or current_token_usage + self.estimate_tokens(meta_text) <= token_budget:
            context_parts.append(meta_text)
        
        result = "\n".join(context_parts)
        
        # Final safety check for token budget
        if token_budget and self.estimate_tokens(result) > token_budget:
            # Emergency compression - keep only essential parts
            emergency_parts = context_parts[:3]  # Header, standings, and rules
            result = "\n".join(emergency_parts)
            result += f"\n⚠️ CONTEXT COMPRESSED due to {token_budget} token limit"
        
        return result

    def update_rolling_context(self, game_context: dict, new_event: str) -> None:
        """Update the rolling context with a new event"""
        # This would be called when significant events happen
        # For now, we'll store events in the context manager
        if not hasattr(self, 'rolling_events'):
            self.rolling_events = []
        
        # Add timestamp and trim to rolling window
        timestamped_event = f"{datetime.now().strftime('%H:%M')} {new_event}"
        self.rolling_events.append(timestamped_event)
        
        # Keep only recent events (sliding window)
        max_events = 20  # Roughly 4 events per turn * 5 turns
        if len(self.rolling_events) > max_events:
            self.rolling_events = self.rolling_events[-max_events:]

    def get_optimized_context_for_model(self, model: str, base_prompt: str, game_context: dict) -> str:
        """Get optimized context specifically tailored for a model's capabilities"""
        usage = self.get_context_usage(model)
        game = game_context.get("game")
        
        # Get RAG-enhanced context if available
        rag_context = ""
        if game and hasattr(game, 'rag_system'):
            try:
                # Extract key terms from the prompt for RAG retrieval
                prompt_keywords = " ".join(base_prompt.split()[:50])  # First 50 words
                relevant_rules = game.rag_system.retrieve_relevant_rules(prompt_keywords, max_results=3)
                
                if relevant_rules:
                    rag_context = "🔍 RELEVANT GAME HISTORY:\n"
                    for rule in relevant_rules:
                        rag_context += f"  Rule {rule['rule_id']}: {rule['text']}\n"
                    rag_context += "\n"
            except Exception as e:
                print(f"RAG retrieval error: {e}")
        
        # Choose context strategy based on usage and model capabilities
        if usage["usage_percentage"] > 95:
            # Emergency: Use minimal context
            return self._emergency_context_reduction(base_prompt, game_context)
        
        elif usage["usage_percentage"] > 85:
            # High pressure: Use adaptive rolling window + RAG only with strict token budget
            remaining_tokens = usage["remaining_tokens"]
            rag_tokens = self.estimate_tokens(rag_context)
            rolling_budget = max(1500, remaining_tokens - rag_tokens - 1000)  # Reserve 1000 for prompt end
            rolling_context = self.create_rolling_context_window(game_context, window_size=3, token_budget=rolling_budget)
            return f"[HIGH CONTEXT PRESSURE: RAG + Adaptive sliding window]\n\n{rag_context}{rolling_context}\n\n{base_prompt[-1000:]}"
        
        elif usage["usage_percentage"] > 75:
            # Moderate pressure: Use adaptive rolling window + RAG + compressed history
            remaining_tokens = usage["remaining_tokens"]
            rag_tokens = self.estimate_tokens(rag_context)
            rolling_budget = max(2000, (remaining_tokens - rag_tokens) // 2)  # Split remaining space
            rolling_context = self.create_rolling_context_window(game_context, window_size=5, token_budget=rolling_budget)
            compressed_history = self.compress_turn_history(game_context, num_turns=3)
            return f"{rag_context}{rolling_context}\n\n{compressed_history}\n\n{base_prompt}"
        
        else:
            # Normal: Use full adaptive rolling window with RAG enhancement
            rolling_context = self.create_rolling_context_window(game_context, window_size=5)
            return f"{rag_context}{rolling_context}\n\n{base_prompt}"


class ComplexityDetectionEngine:
    """Detects game state complexity to inform hybrid model selection"""
    
    def __init__(self):
        self.complexity_weights = {
            "late_game": 0.3,           # Turn 15+ pressure
            "elimination_risk": 0.4,    # Player below 30 points
            "close_competition": 0.3,   # Within 10 points of leader
            "rule_complexity": 0.2,     # High number of active rules
            "close_vote": 0.4,          # Predicted margin ≤ 1 vote
            "meta_rule": 0.3,           # Rules about rules
            "failed_proposals": 0.3,    # Multiple failures increase stakes
            "alliance_formation": 0.2   # Strategic cooperation detected
        }
    
    def calculate_game_state_complexity(self, game_state, player) -> float:
        """Calculate complexity score for current game state (0.0-1.0)"""
        complexity = 0.0
        
        # Late game pressure
        if game_state.turn_number >= 15:
            complexity += self.complexity_weights["late_game"]
            
        # Elimination risk
        if player.points < 30:
            complexity += self.complexity_weights["elimination_risk"]
            
        # Close competition
        if hasattr(game_state, 'players'):
            leader_points = max(p.points for p in game_state.players)
            if abs(player.points - leader_points) <= 10:
                complexity += self.complexity_weights["close_competition"]
        
        # Rule complexity
        if hasattr(game_state, 'rules'):
            total_rules = len(game_state.rules.get("mutable", [])) + len(game_state.rules.get("immutable", []))
            if total_rules > 10:
                complexity += self.complexity_weights["rule_complexity"]
        
        # Failed proposal history
        if hasattr(player, 'failed_proposals') and player.failed_proposals > 2:
            complexity += self.complexity_weights["failed_proposals"]
            
        return min(complexity, 1.0)
    
    def calculate_task_complexity(self, task_type: str, context: dict) -> float:
        """Calculate complexity for specific task type"""
        complexity = 0.0
        
        if task_type == "proposal_deliberation":
            # Meta-rules are more complex
            if context.get("involves_meta_rules", False):
                complexity += self.complexity_weights["meta_rule"]
                
            # Amendment/transmutation complexity
            if context.get("is_amendment", False) or context.get("is_transmutation", False):
                complexity += 0.2
                
        elif task_type == "voting_decision":
            # Close votes require more careful analysis
            predicted_margin = context.get("predicted_vote_margin", 3)
            if predicted_margin <= 1:
                complexity += self.complexity_weights["close_vote"]
                
            # Direct impact on player
            if context.get("affects_player_directly", False):
                complexity += 0.3
                
        elif task_type == "alliance_analysis":
            if context.get("alliance_formation_detected", False):
                complexity += self.complexity_weights["alliance_formation"]
        
        return min(complexity, 1.0)
    
    def should_escalate_to_big_model(self, game_state, player, task_type: str, context: dict, budget_pressure: float) -> bool:
        """Determine if task should use big model based on complexity and budget"""
        game_complexity = self.calculate_game_state_complexity(game_state, player)
        task_complexity = self.calculate_task_complexity(task_type, context)
        
        # Combined complexity score
        total_complexity = (game_complexity + task_complexity) / 2
        
        # Adjust threshold based on budget pressure
        base_threshold = 0.6
        budget_adjusted_threshold = base_threshold + (budget_pressure * 0.3)
        
        # Always escalate for critical situations
        if task_complexity >= 0.8 or game_complexity >= 0.8:
            return True
            
        # Normal escalation logic
        return total_complexity >= budget_adjusted_threshold


class HybridModelSelector:
    """Selects appropriate model (big/small) for each task in hybrid mode"""
    
    def __init__(self, openrouter_client):
        self.client = openrouter_client
        self.complexity_engine = ComplexityDetectionEngine()
        self.escalation_history = {}  # Track escalation decisions for learning
        
    def get_model_for_player_task(self, player, task_type: str, game_state, context: dict = None) -> str:
        """Select optimal model for player task based on complexity and budget"""
        if not self.client.use_hybrid:
            return player.model  # Use assigned model if not in hybrid mode
            
        if context is None:
            context = {}
            
        # Get player's model family
        family = self._get_model_family(player.model)
        if family not in self.client.model_families:
            return player.model  # Fallback to assigned model
            
        # Calculate budget pressure
        total_cost = self.client.get_total_session_cost()
        budget_pressure = min(total_cost / self.client.session_budget, 1.0)
        
        # Check if we should escalate to big model
        should_escalate = self.complexity_engine.should_escalate_to_big_model(
            game_state, player, task_type, context, budget_pressure
        )
        
        if should_escalate:
            selected_model = self.client.model_families[family]["big"]
            self._log_escalation(player.id, task_type, "escalated", selected_model)
        else:
            selected_model = self.client.model_families[family]["small"]
            self._log_escalation(player.id, task_type, "small_model", selected_model)
            
        return selected_model
    
    def _get_model_family(self, model: str) -> str:
        """Determine which model family a model belongs to"""
        if "claude" in model or "anthropic" in model:
            return "claude"
        elif "gpt" in model or "openai" in model:
            return "openai"
        elif "gemini" in model or "google" in model:
            return "google"
        elif "grok" in model or "x-ai" in model:
            return "xai"
        else:
            return "unknown"
    
    def _log_escalation(self, player_id: int, task_type: str, decision: str, model: str):
        """Log escalation decisions for analysis"""
        if player_id not in self.escalation_history:
            self.escalation_history[player_id] = []
            
        self.escalation_history[player_id].append({
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "decision": decision,
            "model": model
        })
    
    def get_escalation_stats(self) -> dict:
        """Get statistics about escalation decisions"""
        stats = {
            "total_decisions": 0,
            "escalations": 0,
            "small_model_usage": 0,
            "by_task_type": {},
            "by_player": {}
        }
        
        for player_id, decisions in self.escalation_history.items():
            stats["by_player"][player_id] = {
                "total": len(decisions),
                "escalated": sum(1 for d in decisions if d["decision"] == "escalated"),
                "small_model": sum(1 for d in decisions if d["decision"] == "small_model")
            }
            
            for decision in decisions:
                stats["total_decisions"] += 1
                if decision["decision"] == "escalated":
                    stats["escalations"] += 1
                else:
                    stats["small_model_usage"] += 1
                    
                task_type = decision["task_type"]
                if task_type not in stats["by_task_type"]:
                    stats["by_task_type"][task_type] = {"escalated": 0, "small_model": 0}
                if decision["decision"] == "escalated":
                    stats["by_task_type"][task_type]["escalated"] += 1
                else:
                    stats["by_task_type"][task_type]["small_model"] += 1
        
        return stats


class BasicRAGSystem:
    """Basic Retrieval-Augmented Generation system for Nomic game context"""
    
    def __init__(self):
        self.rule_index = {}  # rule_id -> rule_text with metadata
        self.proposal_history = []  # List of past proposals with outcomes
        self.player_patterns = {}  # player_id -> patterns in their behavior
        self.keyword_index = {}  # keyword -> list of related rules/events
        
    def index_rule(self, rule_id: int, rule_text: str, metadata: dict):
        """Index a rule for later retrieval"""
        self.rule_index[rule_id] = {
            "text": rule_text,
            "metadata": metadata,
            "keywords": self._extract_keywords(rule_text)
        }
        
        # Update keyword index
        for keyword in self._extract_keywords(rule_text):
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(f"rule_{rule_id}")
    
    def index_proposal(self, proposal_id: int, proposal_text: str, player_id: int, outcome: bool, turn: int):
        """Index a proposal for pattern analysis"""
        proposal_data = {
            "id": proposal_id,
            "text": proposal_text,
            "player_id": player_id,
            "outcome": outcome,
            "turn": turn,
            "keywords": self._extract_keywords(proposal_text)
        }
        
        self.proposal_history.append(proposal_data)
        
        # Update player patterns
        if player_id not in self.player_patterns:
            self.player_patterns[player_id] = {"successful": [], "failed": []}
        
        if outcome:
            self.player_patterns[player_id]["successful"].append(proposal_data)
        else:
            self.player_patterns[player_id]["failed"].append(proposal_data)
    
    def retrieve_relevant_rules(self, query: str, max_results: int = 5) -> list:
        """Retrieve rules relevant to a query"""
        query_keywords = self._extract_keywords(query)
        
        # Score rules by keyword overlap
        rule_scores = {}
        for rule_id, rule_data in self.rule_index.items():
            score = len(set(query_keywords) & set(rule_data["keywords"]))
            if score > 0:
                rule_scores[rule_id] = score
        
        # Return top matches
        top_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)[:max_results]
        
        results = []
        for rule_id, score in top_rules:
            results.append({
                "rule_id": rule_id,
                "text": self.rule_index[rule_id]["text"][:200] + "...",
                "relevance_score": score
            })
        
        return results
    
    def get_player_patterns(self, player_id: int) -> dict:
        """Get behavioral patterns for a specific player"""
        if player_id not in self.player_patterns:
            return {"message": "No patterns found for this player"}
        
        patterns = self.player_patterns[player_id]
        successful_count = len(patterns["successful"])
        failed_count = len(patterns["failed"])
        total_count = successful_count + failed_count
        
        if total_count == 0:
            return {"message": "No proposal history for this player"}
        
        # Extract common keywords from successful proposals
        successful_keywords = []
        for proposal in patterns["successful"]:
            successful_keywords.extend(proposal["keywords"])
        
        # Find most common successful patterns
        from collections import Counter
        common_successful = Counter(successful_keywords).most_common(3)
        
        return {
            "success_rate": round((successful_count / total_count) * 100, 1),
            "total_proposals": total_count,
            "successful_patterns": [keyword for keyword, count in common_successful],
            "recent_successful": [p["text"][:60] + "..." for p in patterns["successful"][-2:]],
            "recent_failed": [p["text"][:60] + "..." for p in patterns["failed"][-2:]]
        }
    
    def get_contextual_suggestions(self, current_situation: str, player_id: int) -> dict:
        """Get AI-powered suggestions based on current situation and history"""
        situation_keywords = self._extract_keywords(current_situation)
        
        # Find similar past situations
        similar_proposals = []
        for proposal in self.proposal_history[-20:]:  # Recent history
            overlap = len(set(situation_keywords) & set(proposal["keywords"]))
            if overlap > 0:
                similar_proposals.append((proposal, overlap))
        
        # Sort by relevance
        similar_proposals.sort(key=lambda x: x[1], reverse=True)
        
        # Get player-specific insights
        player_patterns = self.get_player_patterns(player_id)
        
        suggestions = {
            "relevant_history": [
                f"T{p[0]['turn']}: {p[0]['text'][:60]}... ({'✅' if p[0]['outcome'] else '❌'})"
                for p in similar_proposals[:3]
            ],
            "player_success_rate": player_patterns.get("success_rate", 0),
            "recommended_approach": self._generate_approach_recommendation(situation_keywords, player_patterns)
        }
        
        return suggestions
    
    def _extract_keywords(self, text: str) -> list:
        """Extract relevant keywords from text"""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Common Nomic keywords
        nomic_keywords = [
            "points", "vote", "proposal", "rule", "player", "turn", "mutable", "immutable",
            "gain", "lose", "win", "pass", "fail", "unanimous", "majority", "transmute",
            "amend", "repeal", "leader", "last place", "negative", "positive"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in nomic_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        # Extract numbers (often important in Nomic)
        numbers = re.findall(r'\d+', text)
        found_keywords.extend([f"num_{num}" for num in numbers[:3]])  # Limit to first 3 numbers
        
        return found_keywords
    
    def _generate_approach_recommendation(self, keywords: list, player_patterns: dict) -> str:
        """Generate a strategic approach recommendation"""
        if not player_patterns or player_patterns.get("total_proposals", 0) == 0:
            return "Focus on mutual benefit proposals for first-time success"
        
        success_rate = player_patterns.get("success_rate", 0)
        successful_patterns = player_patterns.get("successful_patterns", [])
        
        if success_rate > 70:
            return f"High success rate! Continue using your effective patterns: {', '.join(successful_patterns[:2])}"
        elif success_rate > 40:
            return f"Moderate success. Try incorporating successful elements: {', '.join(successful_patterns[:2])}"
        else:
            return "Low success rate. Consider more collaborative, mutual-benefit approaches"


class FailureRecoverySystem:
    """System to detect and recover from critical game failures"""
    
    def __init__(self):
        self.failure_history = []
        self.recovery_strategies = []
        self.critical_thresholds = {
            "context_failures": 3,  # Too many context window failures
            "generation_failures": 5,  # Too many generation failures
            "all_negative_turns": 3,  # All players negative for too long
            "no_progress_turns": 10,  # No successful proposals for too long
        }
        self.active_interventions = []
    
    def detect_failure_patterns(self, game_context: dict) -> list:
        """Detect if the game is in a failure state"""
        failures = []
        game = game_context.get("game")
        
        if not game:
            return failures
        
        # Check for context window failures
        if hasattr(game, 'context_manager'):
            high_pressure_models = 0
            for model in game.context_manager.current_usage:
                usage = game.context_manager.get_context_usage(model)
                if usage["usage_percentage"] > 95:
                    high_pressure_models += 1
            
            if high_pressure_models >= 2:
                failures.append({
                    "type": "context_critical",
                    "severity": "high", 
                    "description": f"{high_pressure_models} models at critical context usage",
                    "suggested_intervention": "emergency_context_reset"
                })
        
        # Check for all players in negative points (death spiral)
        if hasattr(game, 'players'):
            all_negative = all(p.points < 0 for p in game.players)
            if all_negative:
                failures.append({
                    "type": "death_spiral",
                    "severity": "critical",
                    "description": "All players have negative points",
                    "suggested_intervention": "emergency_point_recovery"
                })
        
        # Check for generation failures pattern
        if hasattr(game, 'events'):
            recent_events = game.events[-20:] if len(game.events) >= 20 else game.events
            failure_count = sum(1 for event in recent_events if "failed" in str(event).lower() or "error" in str(event).lower())
            if failure_count > 10:  # More than 50% failure rate
                failures.append({
                    "type": "generation_failures",
                    "severity": "medium",
                    "description": f"High failure rate: {failure_count}/20 recent events",
                    "suggested_intervention": "simplify_prompts"
                })
        
        # Check for no progress (no successful proposals)
        if hasattr(game, 'proposals'):
            recent_successful = 0
            for proposal in game.proposals[-10:]:  # Last 10 proposals
                if hasattr(proposal, 'passed') and proposal.passed:
                    recent_successful += 1
            
            if len(game.proposals) >= 10 and recent_successful == 0:
                failures.append({
                    "type": "no_progress",
                    "severity": "medium", 
                    "description": "No successful proposals in last 10 attempts",
                    "suggested_intervention": "collaborative_emergency_rule"
                })
        
        return failures
    
    def implement_recovery_intervention(self, intervention_type: str, game_context: dict) -> dict:
        """Implement a specific recovery intervention"""
        game = game_context.get("game")
        if not game:
            return {"success": False, "error": "No game context"}
        
        intervention_result = {"success": False, "intervention": intervention_type, "actions": []}
        
        if intervention_type == "emergency_context_reset":
            # Reset context usage for all models
            if hasattr(game, 'context_manager'):
                game.context_manager.reset_usage()
                intervention_result["actions"].append("Reset all model context usage")
                intervention_result["success"] = True
        
        elif intervention_type == "emergency_point_recovery":
            # Emergency rule to help all players get positive points
            if hasattr(game, 'players'):
                for player in game.players:
                    if player.points < 0:
                        boost = abs(player.points) + 20  # Get to +20 points
                        player.points += boost
                        intervention_result["actions"].append(f"Emergency boost: {player.name} +{boost} points")
                
                # Add emergency rule to prevent future death spirals
                emergency_rule_text = "Emergency Recovery: If all players have negative points, all players immediately gain 30 points."
                intervention_result["actions"].append(f"Added emergency rule: {emergency_rule_text}")
                intervention_result["success"] = True
        
        elif intervention_type == "simplify_prompts":
            # Temporarily reduce context complexity
            if hasattr(game, 'context_manager'):
                # Force all models to use emergency context reduction
                original_method = game.context_manager.get_optimized_context_for_model
                def simplified_context(model, base_prompt, game_context):
                    return game.context_manager._emergency_context_reduction(base_prompt, game_context)
                
                game.context_manager.get_optimized_context_for_model = simplified_context
                intervention_result["actions"].append("Activated simplified prompt mode")
                intervention_result["success"] = True
        
        elif intervention_type == "collaborative_emergency_rule":
            # Suggest an emergency collaborative rule
            emergency_rule = "Emergency Collaboration: All players gain 5 points when any proposal passes unanimously."
            intervention_result["actions"].append(f"Suggested emergency rule: {emergency_rule}")
            intervention_result["emergency_rule_suggestion"] = emergency_rule
            intervention_result["success"] = True
        
        # Log the intervention
        if intervention_result["success"]:
            self.active_interventions.append({
                "type": intervention_type,
                "timestamp": datetime.now(),
                "actions": intervention_result["actions"]
            })
            
            if hasattr(game, 'add_event'):
                game.add_event(f"🚨 FAILURE RECOVERY: {intervention_type} activated")
        
        return intervention_result
    
    def monitor_game_health(self, game_context: dict) -> dict:
        """Continuous monitoring of game health with automated interventions"""
        failures = self.detect_failure_patterns(game_context)
        health_report = {
            "status": "healthy" if not failures else "critical" if any(f["severity"] == "critical" for f in failures) else "warning",
            "detected_failures": failures,
            "interventions_applied": [],
            "recommendations": []
        }
        
        # Auto-apply critical interventions
        for failure in failures:
            if failure["severity"] == "critical":
                intervention = self.implement_recovery_intervention(
                    failure["suggested_intervention"], game_context
                )
                if intervention["success"]:
                    health_report["interventions_applied"].append(intervention)
            else:
                # Add recommendations for non-critical issues
                health_report["recommendations"].append({
                    "issue": failure["description"],
                    "suggested_action": failure["suggested_intervention"]
                })
        
        return health_report


@dataclass
class RuleEffect:
    """Parsed rule effects that can be executed"""

    trigger: str  # "turn_start", "vote_pass", "vote_fail", "dice_roll", etc.
    condition: Optional[str] = None  # "points < 50", "roll == 6", etc.
    action: str = ""  # "add_points", "subtract_points", "steal_points", etc.
    target: str = "current_player"  # "current_player", "all_players", "proposer", etc.
    value: int = 0
    description: str = ""


@dataclass
class ParsedRule:
    """A rule with parsed executable effects"""

    id: int
    text: str
    mutable: bool
    effects: List[RuleEffect] = field(default_factory=list)
    author: Optional[int] = None
    turn_added: Optional[int] = None
    active: bool = True  # Allow rules to be dynamically enabled/disabled


class RuleParser:
    """Parses rule text to extract executable effects"""

    def __init__(self):
        # Patterns for different rule types
        self.patterns = {
            "point_gain": [
                r"(?:gain|receive|get|earn|award(?:ed)?)\s+(\d+)\s+(?:point|pt)",
                r"(\d+)\s+(?:point|pt)s?\s+(?:to|for)\s+(?:the\s+)?(\w+)",
                r"(?:add|give)\s+(\d+)\s+(?:point|pt)s?",
            ],
            "point_loss": [
                r"(?:lose|deduct|subtract|remove)\s+(\d+)\s+(?:point|pt)",
                r"(\d+)\s+(?:point|pt)s?\s+(?:penalty|deducted|removed)",
            ],
            "conditional": [
                r"(?:if|when|whenever)\s+(.+?)(?:,|then)",
                r"(?:players?\s+)?with\s+(?:fewer|less|more)\s+than\s+(\d+)\s+points",
                r"on\s+a?\s*(?:roll|dice)\s+of\s+(\d+)",
            ],
            "steal": [r"steal\s+(\d+)\s+points?\s+from", r"take\s+(\d+)\s+points?\s+from"],
            "turn_based": [
                r"(?:each|every)\s+turn",
                r"(?:at|during)\s+(?:the\s+)?(?:start|beginning|end)\s+of\s+(?:each\s+)?turn",
                r"per\s+turn",
            ],
            "voting": [
                r"vote\s+against\s+winning\s+proposals",
                r"unanimous\s+(?:vote|consent|approval)",
                r"majority\s+vote",
            ],
        }

    def parse_rule(self, rule_text: str) -> List[RuleEffect]:
        """Parse rule text and extract executable effects"""
        effects = []
        text_lower = rule_text.lower()

        # Check for point gains
        for pattern in self.patterns["point_gain"]:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                value = int(match.group(1))

                # Determine trigger
                trigger = "immediate"
                if any(phrase in text_lower for phrase in ["each turn", "every turn", "per turn"]):
                    trigger = "turn_start"
                elif "vote" in text_lower and "pass" in text_lower:
                    trigger = "vote_pass"
                elif "vote on" in text_lower or "voting" in text_lower:
                    trigger = "vote_cast"
                elif "propose" in text_lower:
                    trigger = "proposal_made"

                # Determine target
                target = "current_player"
                if "all players" in text_lower:
                    target = "all_players"
                elif "proposer" in text_lower:
                    target = "proposer"
                elif "voter" in text_lower:
                    target = "voters"
                elif "last place" in text_lower or "lowest" in text_lower:
                    target = "last_place_players"

                # Check conditions
                condition = None
                if "fewer than" in text_lower or "less than" in text_lower:
                    cond_match = re.search(r"(?:fewer|less)\s+than\s+(\d+)\s+points", text_lower)
                    if cond_match:
                        condition = f"points < {cond_match.group(1)}"
                elif "more than" in text_lower:
                    cond_match = re.search(r"more\s+than\s+(\d+)\s+points", text_lower)
                    if cond_match:
                        condition = f"points > {cond_match.group(1)}"
                elif "last place" in text_lower or "lowest" in text_lower:
                    condition = "is_last_place"

                effects.append(
                    RuleEffect(
                        trigger=trigger,
                        condition=condition,
                        action="add_points",
                        target=target,
                        value=value,
                        description=f"Gain {value} points",
                    )
                )

        return effects


@dataclass
class ModelMetrics:
    """Comprehensive model performance and behavioral metrics"""

    model_name: str

    # Performance Metrics
    total_games: int = 0
    proposals_made: int = 0
    proposals_passed: int = 0
    total_points_gained: int = 0
    games_won: int = 0

    # Coherence Metrics
    coherence_scores: List[float] = field(default_factory=list)
    syntax_errors: int = 0
    validation_failures: int = 0
    logic_inconsistencies: int = 0

    # Memory & Learning Metrics
    memory_reference_count: int = 0
    learning_evidence_count: int = 0
    repeated_failures: int = 0
    adaptation_score: float = 0.0

    # Engagement Metrics
    strategic_depth_scores: List[float] = field(default_factory=list)
    game_understanding_scores: List[float] = field(default_factory=list)
    random_behavior_flags: int = 0

    # Error Analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    correction_responses: int = 0
    improvement_trend: float = 0.0

    # Tie-Breaking Obsession Tracking
    tie_related_proposals: int = 0
    proposal_diversity_scores: List[float] = field(default_factory=list)

    # Timestamps for decay calculation
    last_updated: str = ""
    performance_history: List[Dict] = field(default_factory=list)

    def calculate_success_rate(self) -> float:
        return (self.proposals_passed / max(self.proposals_made, 1)) * 100

    def calculate_average_coherence(self) -> float:
        return sum(self.coherence_scores) / max(len(self.coherence_scores), 1)

    def calculate_strategic_engagement(self) -> float:
        strategic_avg = sum(self.strategic_depth_scores) / max(len(self.strategic_depth_scores), 1)
        understanding_avg = sum(self.game_understanding_scores) / max(len(self.game_understanding_scores), 1)
        return (strategic_avg + understanding_avg) / 2

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall performance score (0-100)"""
        success_rate = self.calculate_success_rate()
        coherence = self.calculate_average_coherence()
        engagement = self.calculate_strategic_engagement()
        diversity = sum(self.proposal_diversity_scores) / max(len(self.proposal_diversity_scores), 1)

        # Error penalty
        error_penalty = min(20, (self.syntax_errors + self.validation_failures) * 2)
        tie_penalty = min(15, self.tie_related_proposals * 3)

        score = (
            (success_rate * 0.3 + coherence * 0.25 + engagement * 0.25 + diversity * 0.2) - error_penalty - tie_penalty
        )
        return max(0, min(100, score))


@dataclass
class Player:
    id: int
    name: str
    role: str
    points: int = 50
    model: str = "llama3.2:3b"
    port: int = 11434
    assigned_model_metrics: Optional[ModelMetrics] = None
    is_human: bool = False
    character: Optional["ProtoCharacter"] = None


@dataclass
class CharacterTraits:
    """Character-specific traits and behaviors"""
    risk_tolerance: str  # "low", "medium", "high"
    cooperation_level: str  # "competitive", "balanced", "cooperative"
    deliberation_range: tuple  # (min_turns, max_turns)
    strategic_focus: str  # "analytical", "aggressive", "diplomatic", "creative", "adaptive"
    decision_speed: str  # "fast", "medium", "slow"
    complexity_preference: str  # "simple", "balanced", "complex"


@dataclass
class ProtoCharacter:
    """Enhanced persistent character with traits and family system"""
    name: str
    strategy: str
    model: str
    family: str  # Model family name
    traits: CharacterTraits
    persistent_thought: str = ""
    games_played: int = 0
    total_wins: int = 0
    total_points_scored: int = 0
    favorite_rule_types: List[str] = field(default_factory=list)
    nemesis_characters: List[str] = field(default_factory=list)
    last_game_reflection: str = ""
    games_without_decision: int = 0  # For rotation tracking
    last_played: str = ""  # Timestamp
    
    def get_win_rate(self) -> float:
        return (self.total_wins / self.games_played) if self.games_played > 0 else 0.0
    
    def update_after_game(self, points_scored: int, won: bool, reflection: str, game_outcome: str = "none"):
        """Update character stats after a game"""
        self.games_played += 1
        self.total_points_scored += points_scored
        if won:
            self.total_wins += 1
            self.games_without_decision = 0  # Reset counter
        elif game_outcome == "lost":
            self.games_without_decision = 0  # Reset counter  
        else:
            self.games_without_decision += 1  # Increment for neutral outcome
        self.last_game_reflection = reflection
        self.last_played = datetime.now().isoformat()
    
    def set_persistent_thought(self, thought: str):
        """Set the thought this character wants to remember for next game"""
        self.persistent_thought = thought[:200]  # Limit length
    
    def should_be_rotated_out(self) -> bool:
        """Check if character should be rotated out due to neutral games limit"""
        return self.games_without_decision >= 2
    
    def get_preferred_deliberation_turns(self, complexity: str = "medium", context_pressure: float = 0.0) -> int:
        """Get preferred number of deliberation turns based on traits, situation, and context pressure"""
        min_turns, max_turns = self.traits.deliberation_range
        
        # Reduce deliberation turns under context pressure
        if context_pressure > 0.95:  # Critical context pressure
            return 1  # Emergency single-turn deliberation
        elif context_pressure > 0.85:  # High context pressure
            return min(2, min_turns)  # Maximum 2 turns
        elif context_pressure > 0.75:  # Moderate context pressure
            return min_turns  # Use minimum turns
        
        # Adjust based on complexity preference and situation
        if complexity == "simple" and self.traits.decision_speed == "fast":
            return min_turns
        elif complexity == "complex" and self.traits.complexity_preference == "complex":
            return max_turns
        else:
            return min_turns + 1 if max_turns > min_turns else min_turns


class CharacterManager:
    """Enhanced character manager with model families and dynamic rotation"""
    
    def __init__(self, characters_file: str = "game_sessions/proto_characters.json"):
        self.characters_file = characters_file
        self.characters: Dict[str, ProtoCharacter] = {}
        self.last_game_winner: Optional[str] = None
        self.load_characters()
        
        # Enhanced character families (5 characters per model family)
        self.character_families = self._create_character_families()
        self.available_models = ["google/gemini-2.0-flash-001", "openai/gpt-4o-mini", 
                                "anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-haiku"]
    
    def _create_character_families(self) -> Dict[str, List[ProtoCharacter]]:
        """Create the full character families with traits"""
        families = {}
        
        # Gemini Family - Analytical and Strategic
        families["gemini"] = [
            ProtoCharacter(
                name="Dr. Maya Chen",
                strategy="Analytical Strategist - uses data and logic to find optimal rule combinations",
                model="google/gemini-2.0-flash-001",
                family="gemini",
                traits=CharacterTraits(
                    risk_tolerance="medium",
                    cooperation_level="balanced",
                    deliberation_range=(2, 3),
                    strategic_focus="analytical",
                    decision_speed="medium",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Lady Victoria",
                strategy="Elegant Tactician - uses sophisticated long-term planning with refined approach",
                model="google/gemini-2.0-flash-001",
                family="gemini",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="cooperative",
                    deliberation_range=(2, 3),
                    strategic_focus="diplomatic",
                    decision_speed="slow",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Professor Atlas",
                strategy="Academic Theorist - approaches rules with scientific method and research focus",
                model="google/gemini-2.0-flash-001",
                family="gemini",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="cooperative",
                    deliberation_range=(3, 3),
                    strategic_focus="analytical",
                    decision_speed="slow",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Captain Nova",
                strategy="Bold Risk-Taker - uses calculated risks and innovative strategies for breakthrough wins",
                model="google/gemini-2.0-flash-001",
                family="gemini",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 2),
                    strategic_focus="aggressive",
                    decision_speed="fast",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Sage Minerva",
                strategy="Wisdom-Focused Elder - relies on deep game understanding and patient strategic positioning",
                model="google/gemini-2.0-flash-001",
                family="gemini",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="balanced",
                    deliberation_range=(2, 3),
                    strategic_focus="adaptive",
                    decision_speed="medium",
                    complexity_preference="complex"
                )
            )
        ]
        
        # GPT-4o Mini Family - Direct and Action-Oriented
        families["gpt4o"] = [
            ProtoCharacter(
                name="Rex Thunder",
                strategy="Aggressive Competitor - favors bold moves and direct confrontation to dominate",
                model="openai/gpt-4o-mini",
                family="gpt4o",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 2),
                    strategic_focus="aggressive",
                    decision_speed="fast",
                    complexity_preference="simple"
                )
            ),
            ProtoCharacter(
                name="Rusty",
                strategy="Underdog Fighter - specializes in comeback strategies and never-give-up attitude",
                model="openai/gpt-4o-mini",
                family="gpt4o",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="balanced",
                    deliberation_range=(1, 2),
                    strategic_focus="adaptive",
                    decision_speed="fast",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Blitz Martinez",
                strategy="Speed & Efficiency Expert - makes rapid tactical decisions and exploits timing advantages",
                model="openai/gpt-4o-mini",
                family="gpt4o",
                traits=CharacterTraits(
                    risk_tolerance="medium",
                    cooperation_level="competitive",
                    deliberation_range=(1, 1),
                    strategic_focus="aggressive",
                    decision_speed="fast",
                    complexity_preference="simple"
                )
            ),
            ProtoCharacter(
                name="Alex Storm",
                strategy="Unpredictable Maverick - uses erratic patterns and surprising moves to confuse opponents",
                model="openai/gpt-4o-mini",
                family="gpt4o",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 3),
                    strategic_focus="creative",
                    decision_speed="medium",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Riley Phoenix",
                strategy="Comeback Specialist - thrives in difficult positions and turns defeats into victories",
                model="openai/gpt-4o-mini",
                family="gpt4o",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="balanced",
                    deliberation_range=(1, 2),
                    strategic_focus="adaptive",
                    decision_speed="medium",
                    complexity_preference="balanced"
                )
            )
        ]
        
        # Claude Sonnet Family - Thoughtful and Cooperative
        families["claude_sonnet"] = [
            ProtoCharacter(
                name="Professor Iris",
                strategy="Cooperation Theorist - seeks mutually beneficial rules and builds strategic alliances",
                model="anthropic/claude-3.5-sonnet",
                family="claude_sonnet",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="cooperative",
                    deliberation_range=(2, 3),
                    strategic_focus="diplomatic",
                    decision_speed="medium",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Judge Sterling",
                strategy="Rule Lawyer & Analyst - finds loopholes and ensures precise rule interpretation",
                model="anthropic/claude-3.5-sonnet",
                family="claude_sonnet",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="balanced",
                    deliberation_range=(2, 3),
                    strategic_focus="analytical",
                    decision_speed="slow",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Harmony Bell",
                strategy="Diplomatic Mediator - builds consensus and finds win-win solutions for complex conflicts",
                model="anthropic/claude-3.5-sonnet",
                family="claude_sonnet",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="cooperative",
                    deliberation_range=(2, 3),
                    strategic_focus="diplomatic",
                    decision_speed="medium",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Scholar Moon",
                strategy="Research & Logic Expert - applies systematic analysis and evidence-based decision making",
                model="anthropic/claude-3.5-sonnet",
                family="claude_sonnet",
                traits=CharacterTraits(
                    risk_tolerance="low",
                    cooperation_level="balanced",
                    deliberation_range=(3, 3),
                    strategic_focus="analytical",
                    decision_speed="slow",
                    complexity_preference="complex"
                )
            ),
            ProtoCharacter(
                name="Morgan Prime",
                strategy="System Builder - designs comprehensive rule frameworks and long-term game structures",
                model="anthropic/claude-3.5-sonnet",
                family="claude_sonnet",
                traits=CharacterTraits(
                    risk_tolerance="medium",
                    cooperation_level="balanced",
                    deliberation_range=(2, 3),
                    strategic_focus="analytical",
                    decision_speed="slow",
                    complexity_preference="complex"
                )
            )
        ]
        
        # Claude Haiku Family - Creative and Disruptive
        families["claude_haiku"] = [
            ProtoCharacter(
                name="Jack Chaos",
                strategy="Chaos Agent - creates unpredictable and disruptive rule changes to shake up the game",
                model="anthropic/claude-3.5-haiku",
                family="claude_haiku",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 2),
                    strategic_focus="creative",
                    decision_speed="fast",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Quinn Wild",
                strategy="Creative Disruptor - breaks conventional patterns with innovative and unexpected rule proposals",
                model="anthropic/claude-3.5-haiku",
                family="claude_haiku",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 2),
                    strategic_focus="creative",
                    decision_speed="fast",
                    complexity_preference="simple"
                )
            ),
            ProtoCharacter(
                name="Echo Dance",
                strategy="Pattern Breaker - identifies and disrupts established game patterns and player habits",
                model="anthropic/claude-3.5-haiku",
                family="claude_haiku",
                traits=CharacterTraits(
                    risk_tolerance="medium",
                    cooperation_level="balanced",
                    deliberation_range=(1, 2),
                    strategic_focus="creative",
                    decision_speed="medium",
                    complexity_preference="balanced"
                )
            ),
            ProtoCharacter(
                name="Rebel Knight",
                strategy="Anti-Establishment Fighter - challenges existing power structures and dominant strategies",
                model="anthropic/claude-3.5-haiku",
                family="claude_haiku",
                traits=CharacterTraits(
                    risk_tolerance="high",
                    cooperation_level="competitive",
                    deliberation_range=(1, 2),
                    strategic_focus="aggressive",
                    decision_speed="fast",
                    complexity_preference="simple"
                )
            ),
            ProtoCharacter(
                name="Trickster Fox",
                strategy="Cunning Opportunist - exploits temporary advantages and creates clever rule combinations",
                model="anthropic/claude-3.5-haiku",
                family="claude_haiku",
                traits=CharacterTraits(
                    risk_tolerance="medium",
                    cooperation_level="balanced",
                    deliberation_range=(1, 2),
                    strategic_focus="creative",
                    decision_speed="medium",
                    complexity_preference="balanced"
                )
            )
        ]
        
        return families
    
    def load_characters(self):
        """Load characters from persistent storage with enhanced format support"""
        try:
            if os.path.exists(self.characters_file):
                with open(self.characters_file, 'r') as f:
                    data = json.load(f)
                    for char_data in data:
                        # Handle both old and new character formats
                        if 'traits' in char_data and 'family' in char_data:
                            # New enhanced format
                            traits_data = char_data['traits']
                            char_data['traits'] = CharacterTraits(**traits_data)
                            char = ProtoCharacter(**char_data)
                        else:
                            # Legacy format - upgrade to new format
                            char = self._upgrade_legacy_character(char_data)
                        self.characters[char.name] = char
        except Exception as e:
            print(f"Warning: Could not load characters: {e}")
    
    def _upgrade_legacy_character(self, char_data: dict) -> ProtoCharacter:
        """Upgrade legacy character data to new enhanced format"""
        name = char_data['name']
        
        # Find matching character in families to get traits
        for family_name, characters in self.character_families.items():
            for char_template in characters:
                if char_template.name == name:
                    # Copy template traits and merge with existing data
                    enhanced_char = ProtoCharacter(
                        name=char_data['name'],
                        strategy=char_data['strategy'],
                        model=char_data['model'],
                        family=char_template.family,
                        traits=char_template.traits,
                        persistent_thought=char_data.get('persistent_thought', ''),
                        games_played=char_data.get('games_played', 0),
                        total_wins=char_data.get('total_wins', 0),
                        total_points_scored=char_data.get('total_points_scored', 0),
                        favorite_rule_types=char_data.get('favorite_rule_types', []),
                        nemesis_characters=char_data.get('nemesis_characters', []),
                        last_game_reflection=char_data.get('last_game_reflection', ''),
                        games_without_decision=0,  # Reset for new system
                        last_played=char_data.get('last_played', '')
                    )
                    return enhanced_char
        
        # Fallback if character not found in families
        default_traits = CharacterTraits(
            risk_tolerance="medium",
            cooperation_level="balanced", 
            deliberation_range=(1, 2),
            strategic_focus="adaptive",
            decision_speed="medium",
            complexity_preference="balanced"
        )
        return ProtoCharacter(
            name=char_data['name'],
            strategy=char_data['strategy'], 
            model=char_data['model'],
            family="unknown",
            traits=default_traits,
            persistent_thought=char_data.get('persistent_thought', ''),
            games_played=char_data.get('games_played', 0),
            total_wins=char_data.get('total_wins', 0),
            total_points_scored=char_data.get('total_points_scored', 0)
        )
    
    def save_characters(self):
        """Save characters to persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.characters_file), exist_ok=True)
            with open(self.characters_file, 'w') as f:
                char_list = []
                for char in self.characters.values():
                    char_dict = asdict(char)
                    # Convert traits to dict for JSON serialization
                    if isinstance(char_dict['traits'], CharacterTraits):
                        char_dict['traits'] = asdict(char_dict['traits'])
                    char_list.append(char_dict)
                json.dump(char_list, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save characters: {e}")
    
    def get_character(self, name: str) -> ProtoCharacter:
        """Get character by name, creating from families if needed"""
        if name in self.characters:
            return self.characters[name]
        
        # Look for character in families
        for family_name, characters in self.character_families.items():
            for char_template in characters:
                if char_template.name == name:
                    # Copy template to our active characters
                    new_char = ProtoCharacter(
                        name=char_template.name,
                        strategy=char_template.strategy,
                        model=char_template.model,
                        family=char_template.family,
                        traits=char_template.traits
                    )
                    self.characters[name] = new_char
                    return new_char
        
        # Fallback - shouldn't happen with proper selection
        raise ValueError(f"Character '{name}' not found in any family")
    
    def get_all_characters(self) -> List[ProtoCharacter]:
        """Get list of all available characters from all families"""
        all_chars = []
        for family_chars in self.character_families.values():
            all_chars.extend(family_chars)
        return all_chars
    
    def set_winner(self, winner_name: str):
        """Set the winner of the last game for rotation purposes"""
        self.last_game_winner = winner_name
    
    def assign_characters_to_players(self, num_players: int) -> List[ProtoCharacter]:
        """Enhanced character assignment with dynamic rotation system"""
        return self._select_characters_with_rotation(num_players)
    
    def _select_characters_with_rotation(self, num_players: int) -> List[ProtoCharacter]:
        """Select characters using the dynamic rotation system"""
        selected_characters = []
        family_names = list(self.character_families.keys())
        
        # Step 1: Handle winner retention
        if self.last_game_winner and num_players >= 1:
            winner_char = self.get_character(self.last_game_winner)
            selected_characters.append(winner_char)
            print(f" Winner {self.last_game_winner} retained for next game")
            
            # Remove winner's family from selection pool for this round
            winner_family = winner_char.family
            if winner_family in family_names:
                family_names.remove(winner_family)
        
        # Step 2: Select one character from each remaining family
        families_needed = min(num_players - len(selected_characters), len(family_names))
        selected_families = family_names[:families_needed]
        
        for family_name in selected_families:
            char = self._select_character_from_family(family_name)
            selected_characters.append(char)
        
        # Step 3: If we need more characters (larger games), add from families
        while len(selected_characters) < num_players:
            # Pick from families that aren't represented yet, or cycle through
            remaining_families = [f for f in self.character_families.keys() 
                                if f not in [c.family for c in selected_characters]]
            if not remaining_families:
                remaining_families = list(self.character_families.keys())
            
            family_name = random.choice(remaining_families)
            char = self._select_character_from_family(family_name, exclude=[c.name for c in selected_characters])
            selected_characters.append(char)
        
        # Load persistent data for all selected characters
        result = []
        for char_template in selected_characters:
            persistent_char = self.get_character(char_template.name)
            result.append(persistent_char)
        
        return result
    
    def _select_character_from_family(self, family_name: str, exclude: List[str] = None) -> ProtoCharacter:
        """Select a character from a specific family using rotation rules"""
        if exclude is None:
            exclude = []
            
        family_chars = self.character_families[family_name]
        eligible_chars = [c for c in family_chars if c.name not in exclude]
        
        if not eligible_chars:
            # All characters excluded, pick any from family
            eligible_chars = family_chars
        
        # Priority 1: Characters that need to be rotated out (games_without_decision >= 2)
        chars_needing_rotation = []
        for char in eligible_chars:
            persistent_char = self.characters.get(char.name)
            if persistent_char and persistent_char.should_be_rotated_out():
                chars_needing_rotation.append(char)
        
        if chars_needing_rotation:
            return random.choice(chars_needing_rotation)
        
        # Priority 2: Characters that haven't played recently
        chars_by_recency = []
        for char in eligible_chars:
            persistent_char = self.characters.get(char.name)
            if persistent_char:
                last_played = persistent_char.last_played if persistent_char.last_played else "1970-01-01"
                chars_by_recency.append((char, last_played))
            else:
                # Never played - high priority
                chars_by_recency.append((char, "1970-01-01"))
        
        # Sort by last played date (oldest first)
        chars_by_recency.sort(key=lambda x: x[1])
        
        # Select from the 2-3 least recently played characters
        top_candidates = chars_by_recency[:min(3, len(chars_by_recency))]
        selected_char, _ = random.choice(top_candidates)
        
        return selected_char


@dataclass
class Proposal:
    id: int
    player_id: int
    rule_text: str
    explanation: str
    internal_thoughts: str
    turn: int
    votes: Dict[int, bool] = None
    is_transmutation: bool = False
    transmute_target_rule: int = None
    is_amendment: bool = False
    amend_target_rule: int = None
    original_rule_text: str = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
        
        # Auto-detect transmutation from rule text
        if "transmute rule" in self.rule_text.lower():
            self.is_transmutation = True
            # Extract rule number
            import re
            match = re.search(r'transmute rule (\d+)', self.rule_text.lower())
            if match:
                self.transmute_target_rule = int(match.group(1))
        
        # Auto-detect amendment from rule text
        elif "amend rule" in self.rule_text.lower():
            self.is_amendment = True
            # Extract rule number
            import re
            match = re.search(r'amend rule (\d+)', self.rule_text.lower())
            if match:
                self.amend_target_rule = int(match.group(1))


@dataclass
class PlayerInternalState:
    """Comprehensive internal state for a player"""

    player_id: int
    strategic_focus: str = ""
    victory_path: str = ""
    threat_assessments: Dict[int, str] = field(default_factory=dict)
    alliance_considerations: Dict[int, str] = field(default_factory=dict)
    planned_proposals: List[str] = field(default_factory=list)
    voting_strategies: Dict[str, str] = field(default_factory=dict)
    learning_observations: List[str] = field(default_factory=list)
    rule_effectiveness_notes: Dict[int, str] = field(default_factory=dict)
    scratchpad: List[str] = field(default_factory=list)  # Persistent notes across turns
    last_updated: str = ""
    idle_turn_count: int = 0


@dataclass
class IdleTurnLog:
    """Log entry for idle turn strategic thinking"""

    player_id: int
    turn_number: int
    idle_turn_index: int
    timestamp: str
    strategic_analysis: str
    threat_reassessment: str
    plan_refinement: str
    learned_insights: str


@dataclass
class GameSession:
    """Complete game session data"""

    session_id: str
    start_time: str
    end_time: Optional[str] = None
    players: List[Dict] = field(default_factory=list)
    final_scores: Dict[int, int] = field(default_factory=dict)
    winner_id: Optional[int] = None
    total_turns: int = 0
    rules_created: int = 0
    session_summary: str = ""


class GameSessionManager:
    """Manages persistent game session storage and retrieval"""

    def __init__(self, sessions_dir: str = "game_sessions"):
        self.sessions_dir = sessions_dir
        self.current_session: Optional[GameSession] = None
        self.ensure_sessions_directory()

    def ensure_sessions_directory(self):
        """Create sessions directory if it doesn't exist"""
        os.makedirs(self.sessions_dir, exist_ok=True)

    def start_new_session(self, players: List[Player]) -> str:
        """Start a new game session"""
        session_id = f"session_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"

        self.current_session = GameSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            players=[
                {"id": p.id, "name": p.name, "role": p.role, "model": p.model, "starting_points": p.points}
                for p in players
            ],
        )

        # Create session directory
        session_path = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_path, exist_ok=True)

        # Save initial session metadata
        self.save_session_metadata()

        return session_id

    def end_session(self, final_scores: Dict[int, int], winner_id: Optional[int], total_turns: int):
        """End the current session"""
        if self.current_session:
            self.current_session.end_time = datetime.now().isoformat()
            self.current_session.final_scores = final_scores
            self.current_session.winner_id = winner_id
            self.current_session.total_turns = total_turns
            self.save_session_metadata()

    def save_session_metadata(self):
        """Save session metadata to file"""
        if self.current_session:
            session_path = os.path.join(self.sessions_dir, self.current_session.session_id)
            metadata_path = os.path.join(session_path, "game_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(asdict(self.current_session), f, indent=2)

    def save_turn_log(self, turn_data: Dict):
        """Save detailed turn log"""
        if self.current_session:
            session_path = os.path.join(self.sessions_dir, self.current_session.session_id)
            logs_path = os.path.join(session_path, "turn_logs.json")

            # Load existing logs
            logs = []
            if os.path.exists(logs_path):
                with open(logs_path, "r") as f:
                    logs = json.load(f)

            # Add new turn
            logs.append(turn_data)

            # Save updated logs
            with open(logs_path, "w") as f:
                json.dump(logs, f, indent=2)

    def save_player_states(self, player_states: Dict[int, PlayerInternalState]):
        """Save current player internal states"""
        if self.current_session:
            session_path = os.path.join(self.sessions_dir, self.current_session.session_id)
            states_path = os.path.join(session_path, "player_states.json")

            states_data = {
                "timestamp": datetime.now().isoformat(),
                "states": {str(pid): asdict(state) for pid, state in player_states.items()},
            }

            with open(states_path, "w") as f:
                json.dump(states_data, f, indent=2)

    def save_idle_turn_log(self, idle_turn: IdleTurnLog):
        """Save idle turn log entry"""
        if self.current_session:
            session_path = os.path.join(self.sessions_dir, self.current_session.session_id)
            idle_path = os.path.join(session_path, "idle_turns.json")

            # Load existing logs
            logs = []
            if os.path.exists(idle_path):
                with open(idle_path, "r") as f:
                    logs = json.load(f)

            # Add new idle turn
            logs.append(asdict(idle_turn))

            # Save updated logs
            with open(idle_path, "w") as f:
                json.dump(logs, f, indent=2)

    def get_session_history(self) -> List[GameSession]:
        """Get list of all completed sessions"""
        sessions = []
        if os.path.exists(self.sessions_dir):
            for session_dir in os.listdir(self.sessions_dir):
                metadata_path = os.path.join(self.sessions_dir, session_dir, "game_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        sessions.append(GameSession(**data))
        return sorted(sessions, key=lambda s: s.start_time, reverse=True)

    def find_latest_in_progress_session(self) -> Optional[str]:
        """Find the most recent session that is still in progress (no end_time)"""
        if not os.path.exists(self.sessions_dir):
            return None
        
        in_progress_sessions = []
        for session_dir in os.listdir(self.sessions_dir):
            metadata_path = os.path.join(self.sessions_dir, session_dir, "game_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        # Check if session is in progress (no end_time)
                        if not data.get("end_time"):
                            in_progress_sessions.append((session_dir, data.get("start_time", "")))
                except Exception as e:
                    print(f"Warning: Could not read session {session_dir}: {e}")
                    continue
        
        if not in_progress_sessions:
            return None
        
        # Return the most recent in-progress session
        in_progress_sessions.sort(key=lambda x: x[1], reverse=True)
        return in_progress_sessions[0][0]

    def load_session_data(self, session_id: str) -> Optional[Dict]:
        """Load complete session data for resuming"""
        session_path = os.path.join(self.sessions_dir, session_id)
        if not os.path.exists(session_path):
            return None
        
        session_data = {}
        
        # Load metadata
        metadata_path = os.path.join(session_path, "game_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                session_data["metadata"] = json.load(f)
        
        # Load turn logs
        turn_logs_path = os.path.join(session_path, "turn_logs.json")
        if os.path.exists(turn_logs_path):
            with open(turn_logs_path, "r") as f:
                session_data["turn_logs"] = json.load(f)
        
        # Load player states
        player_states_path = os.path.join(session_path, "player_states.json")
        if os.path.exists(player_states_path):
            with open(player_states_path, "r") as f:
                player_states_data = json.load(f)
                session_data["player_states"] = player_states_data.get("states", {})
        
        # Load idle turns
        idle_turns_path = os.path.join(session_path, "idle_turns.json")
        if os.path.exists(idle_turns_path):
            with open(idle_turns_path, "r") as f:
                session_data["idle_turns"] = json.load(f)
        
        return session_data

    def resume_session(self, session_id: str) -> bool:
        """Resume an existing session"""
        session_data = self.load_session_data(session_id)
        if not session_data or not session_data.get("metadata"):
            return False
        
        metadata = session_data["metadata"]
        self.current_session = GameSession(**metadata)
        print(f"✅ Resumed session: {session_id}")
        print(f"   Started: {metadata.get('start_time', 'Unknown')}")
        print(f"   Players: {len(metadata.get('players', []))}")
        print(f"   Total turns: {metadata.get('total_turns', 0)}")
        return True


class InternalStateTracker:
    """Manages comprehensive player internal state tracking"""

    def __init__(self):
        self.player_states: Dict[int, PlayerInternalState] = {}

    def initialize_player_state(self, player: Player):
        """Initialize internal state for a player"""
        self.player_states[player.id] = PlayerInternalState(
            player_id=player.id, last_updated=datetime.now().isoformat()
        )

    def update_strategic_focus(self, player_id: int, focus: str):
        """Update player's strategic focus"""
        if player_id in self.player_states:
            self.player_states[player_id].strategic_focus = focus
            self.player_states[player_id].last_updated = datetime.now().isoformat()

    def update_threat_assessment(self, player_id: int, target_id: int, assessment: str):
        """Update threat assessment of another player"""
        if player_id in self.player_states:
            self.player_states[player_id].threat_assessments[target_id] = assessment
            self.player_states[player_id].last_updated = datetime.now().isoformat()

    def add_learning_observation(self, player_id: int, observation: str):
        """Add a learning observation"""
        if player_id in self.player_states:
            self.player_states[player_id].learning_observations.append(observation)
            self.player_states[player_id].last_updated = datetime.now().isoformat()

    def add_planned_proposal(self, player_id: int, proposal: str):
        """Add a planned proposal"""
        if player_id in self.player_states:
            self.player_states[player_id].planned_proposals.append(proposal)
            self.player_states[player_id].last_updated = datetime.now().isoformat()

    def update_rule_effectiveness(self, player_id: int, rule_id: int, note: str):
        """Update notes on rule effectiveness"""
        if player_id in self.player_states:
            self.player_states[player_id].rule_effectiveness_notes[rule_id] = note
            self.player_states[player_id].last_updated = datetime.now().isoformat()

    def get_player_state(self, player_id: int) -> Optional[PlayerInternalState]:
        """Get player's internal state"""
        return self.player_states.get(player_id)

    def get_all_states(self) -> Dict[int, PlayerInternalState]:
        """Get all player states"""
        return self.player_states.copy()


class IdleTurnProcessor:
    """Handles background strategic thinking during idle turns"""

    def __init__(self, ollama_manager, deliberation_manager, game_instance=None):
        self.ollama = ollama_manager
        self.deliberation = deliberation_manager
        self.game_instance = game_instance  # Reference to main game for unified_generate
        self.max_idle_turns = 3

    def can_process_idle_turn(self, player: Player, state: PlayerInternalState) -> bool:
        """Check if player can process an idle turn"""
        return state.idle_turn_count < self.max_idle_turns

    def process_idle_turn(self, player: Player, game_state: Dict, current_turn: int) -> IdleTurnLog:
        """Process an idle turn for strategic thinking"""
        context_header = self.deliberation.generate_nomic_context_header(game_state, player)

        # Get current internal state
        state = game_state.get("player_states", {}).get(player.id, {})

        prompt = f"""{context_header}

 IDLE TURN STRATEGIC ANALYSIS (Background Thinking)

You have idle time while other players take their turns. Use this time for deep strategic thinking.

Your Current Strategic State:
- Focus: {state.get('strategic_focus', 'Not set')}
- Victory Path: {state.get('victory_path', 'Not defined')}
- Current Threats: {state.get('threat_assessments', {})}

IDLE TURN ANALYSIS TASKS:

1. STRATEGIC POSITION REASSESSMENT:
Analyze how the recent turns have changed your position and update your threat analysis.

2. PLAN REFINEMENT:
Based on recent rule changes and player actions, refine your victory strategy.

3. LEARNING INSIGHTS:
What new patterns or opportunities have you observed? What has worked/failed?

4. FUTURE PREPARATION:
What proposals should you consider for your next turn? How should you vote?

OUTPUT FORMAT:
STRATEGIC_ANALYSIS: [Updated analysis of your position and threats]
THREAT_REASSESSMENT: [How other players' threat levels have changed]
PLAN_REFINEMENT: [Updated strategy and victory path]
LEARNED_INSIGHTS: [New observations and lessons learned]
"""

        # Use game instance's unified_generate method if available, otherwise fallback to ollama
        if self.game_instance and hasattr(self.game_instance, "unified_generate"):
            response = self.game_instance.unified_generate(player, prompt)
        else:
            # Fallback to direct ollama call
            response = self.ollama.generate(player.model, prompt, player.port)

        # Parse response
        strategic_analysis = self._extract_section(response, "STRATEGIC_ANALYSIS")
        threat_reassessment = self._extract_section(response, "THREAT_REASSESSMENT")
        plan_refinement = self._extract_section(response, "PLAN_REFINEMENT")
        learned_insights = self._extract_section(response, "LEARNED_INSIGHTS")

        return IdleTurnLog(
            player_id=player.id,
            turn_number=current_turn,
            idle_turn_index=state.get("idle_turn_count", 0) + 1,
            timestamp=datetime.now().isoformat(),
            strategic_analysis=strategic_analysis,
            threat_reassessment=threat_reassessment,
            plan_refinement=plan_refinement,
            learned_insights=learned_insights,
        )

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract a section from the response"""
        pattern = f"{section_name}:\\s*(.*?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "No response"


class GameLogger:
    """Structured logging for game events, strategies, and system metrics"""

    def __init__(self, session_manager: GameSessionManager):
        self.session_manager = session_manager
        self.current_turn_log = []

    def log_turn_start(self, turn_number: int, player: Player):
        """Log start of a turn"""
        self.current_turn_log = []
        self.current_turn_log.append(
            {
                "type": "turn_start",
                "timestamp": datetime.now().isoformat(),
                "turn_number": turn_number,
                "player_id": player.id,
                "player_name": player.name,
                "player_model": player.model,
            }
        )

    def log_deliberation_step(self, player_id: int, step: int, prompt: str, response: str):
        """Log a deliberation step"""
        self.current_turn_log.append(
            {
                "type": "deliberation",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "deliberation_step": step,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "response": response,
            }
        )

    def log_proposal(self, player_id: int, proposal: Proposal):
        """Log a proposal"""
        self.current_turn_log.append(
            {
                "type": "proposal",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "rule_text": proposal.rule_text,
                "explanation": proposal.explanation,
                "internal_thoughts": proposal.internal_thoughts,
            }
        )

    def log_vote(self, voter_id: int, vote: bool, reasoning: str):
        """Log a vote"""
        self.current_turn_log.append(
            {
                "type": "vote",
                "timestamp": datetime.now().isoformat(),
                "voter_id": voter_id,
                "vote": vote,
                "reasoning": reasoning,
            }
        )

    def log_idle_turn(self, idle_turn: IdleTurnLog):
        """Log an idle turn"""
        self.current_turn_log.append(
            {
                "type": "idle_turn",
                "timestamp": idle_turn.timestamp,
                "player_id": idle_turn.player_id,
                "idle_turn_index": idle_turn.idle_turn_index,
                "strategic_analysis": idle_turn.strategic_analysis,
                "insights": idle_turn.learned_insights,
            }
        )

    def log_proposal_outcome(
        self, player_id: int, rule_text: str, passed: bool, ayes: int, total: int, percentage: float
    ):
        """Log the outcome of a proposal"""
        self.current_turn_log.append(
            {
                "type": "proposal_outcome",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "rule_text": rule_text,
                "passed": passed,
                "votes_for": ayes,
                "total_votes": total,
                "percentage": percentage,
            }
        )

    def log_rule_execution(self, trigger: str, context: dict):
        """Log rule execution events"""
        self.current_turn_log.append(
            {"type": "rule_execution", "timestamp": datetime.now().isoformat(), "trigger": trigger, "context": context}
        )

    def log_game_start(self, players: List[Player]):
        """Log game initialization"""
        self.current_turn_log.append(
            {
                "type": "game_start",
                "timestamp": datetime.now().isoformat(),
                "players": [{"id": p.id, "name": p.name, "model": p.model, "points": p.points} for p in players],
            }
        )

    def log_game_end(self, winner_id: Optional[int], final_turn: int, final_scores: Dict[int, int]):
        """Log game completion"""
        self.current_turn_log.append(
            {
                "type": "game_end",
                "timestamp": datetime.now().isoformat(),
                "winner_id": winner_id,
                "final_turn": final_turn,
                "final_scores": final_scores,
            }
        )
    
    def log_scratchpad_update(self, player_id: int, new_notes: List[str], total_notes: int):
        """Log scratchpad note updates"""
        self.current_turn_log.append(
            {
                "type": "scratchpad_update",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "new_notes": new_notes,
                "total_notes": total_notes
            }
        )
    
    def log_failed_proposal(self, player_id: int, error_reason: str, retry_count: int):
        """Log failed proposal attempts"""
        self.current_turn_log.append(
            {
                "type": "failed_proposal",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "error_reason": error_reason,
                "retry_count": retry_count
            }
        )
    
    def log_deliberation_choice(self, player_id: int, chosen_turns: int, reasoning: str):
        """Log flexible deliberation choices"""
        self.current_turn_log.append(
            {
                "type": "deliberation_choice",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "chosen_turns": chosen_turns,
                "reasoning": reasoning
            }
        )
    
    def log_insight(self, player_id: int, insight: str):
        """Log player insights"""
        self.current_turn_log.append(
            {
                "type": "insight",
                "timestamp": datetime.now().isoformat(),
                "player_id": player_id,
                "insight": insight
            }
        )

    def finalize_turn_log(self, turn_number: int):
        """Finalize and save the turn log"""
        turn_data = {
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat(),
            "events": self.current_turn_log.copy(),
        }
        self.session_manager.save_turn_log(turn_data)
        self.current_turn_log = []


class ModelPerformanceManager:
    """Cross-game model performance tracking and management"""

    def __init__(self, storage_path: str = "model_performance.json"):
        self.storage_path = storage_path
        self.metrics: Dict[str, ModelMetrics] = {}
        self.load_metrics()

        # Tie-breaking keywords for detection
        self.tie_keywords = {
            "tie",
            "tied",
            "equal",
            "same",
            "break",
            "breaker",
            "tiebreak",
            "deadlock",
            "even",
            "identical",
            "resolve",
            "determine",
            "winner",
        }

        # Strategic keywords for depth analysis
        self.strategic_keywords = {
            "strategy",
            "tactical",
            "advantage",
            "position",
            "opportunity",
            "leverage",
            "timing",
            "future",
            "long-term",
            "consequence",
            "alliance",
            "cooperation",
        }

    def load_metrics(self):
        """Load performance metrics from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for model_name, metrics_data in data.items():
                        # Convert back to ModelMetrics object
                        metrics = ModelMetrics(**metrics_data)
                        self.metrics[model_name] = metrics
            except Exception as e:
                print(f"Error loading metrics: {e}")
                self.metrics = {}

    def save_metrics(self):
        """Save performance metrics to storage"""
        try:
            data = {}
            for model_name, metrics in self.metrics.items():
                data[model_name] = asdict(metrics)
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def get_or_create_metrics(self, model_name: str) -> ModelMetrics:
        """Get existing metrics or create new ones for a model"""
        if model_name not in self.metrics:
            self.metrics[model_name] = ModelMetrics(model_name=model_name)
        return self.metrics[model_name]

    def analyze_proposal_coherence(self, rule_text: str, explanation: str, internal_thought: str) -> float:
        """Analyze coherence of a proposal (0-100 score)"""
        score = 50.0  # Base score

        # Text clarity and completeness
        if len(rule_text.strip()) < 10:
            score -= 20
        if len(explanation.strip()) < 20:
            score -= 15
        if len(internal_thought.strip()) < 10:
            score -= 10

        # Check for syntax issues
        if rule_text.count(".") == 0:  # No ending punctuation
            score -= 5
        if not rule_text[0].isupper():  # Doesn't start with capital
            score -= 5

        # Check for logical consistency between internal thought and explanation
        thought_words = set(internal_thought.lower().split())
        explanation_words = set(explanation.lower().split())
        overlap = len(thought_words.intersection(explanation_words))
        consistency_bonus = min(15, overlap * 2)
        score += consistency_bonus

        # Check for strategic depth
        strategic_words = sum(1 for word in rule_text.lower().split() if word in self.strategic_keywords)
        score += min(10, strategic_words * 3)

        return max(0, min(100, score))

    def analyze_memory_usage(self, internal_thought: str, explanation: str, game_context: str) -> float:
        """Analyze evidence of memory and learning (0-100 score)"""
        score = 0.0
        text_combined = (internal_thought + " " + explanation).lower()

        # Look for references to past events
        memory_indicators = ["previous", "last", "before", "earlier", "remember", "learned", "failed", "worked"]
        memory_refs = sum(1 for indicator in memory_indicators if indicator in text_combined)
        score += min(30, memory_refs * 5)

        # Look for specific game references
        if "turn" in text_combined and any(char.isdigit() for char in text_combined):
            score += 20
        if "rule" in text_combined and any(char.isdigit() for char in text_combined):
            score += 15

        # Check for learning evidence
        learning_words = ["learned", "noticed", "realized", "understand", "see that", "because"]
        learning_evidence = sum(1 for word in learning_words if word in text_combined)
        score += min(25, learning_evidence * 8)

        return min(100, score)

    def detect_tie_obsession(self, rule_text: str, explanation: str, internal_thought: str) -> bool:
        """Detect if proposal is obsessed with tie-breaking"""
        text_combined = (rule_text + " " + explanation + " " + internal_thought).lower()

        tie_word_count = sum(1 for word in self.tie_keywords if word in text_combined)
        word_count = len(text_combined.split())

        # If more than 3% of words are tie-related, flag as obsessed
        return tie_word_count > 0 and (tie_word_count / word_count) > 0.03

    def calculate_proposal_diversity(self, rule_text: str, all_previous_proposals: List[str]) -> float:
        """Calculate diversity score against all previous proposals (0-100)"""
        if not all_previous_proposals:
            return 100.0

        current_words = set(rule_text.lower().split())
        similarities = []

        for prev_proposal in all_previous_proposals:
            prev_words = set(prev_proposal.lower().split())
            if len(current_words) == 0 or len(prev_words) == 0:
                continue

            # Calculate Jaccard similarity
            intersection = len(current_words.intersection(prev_words))
            union = len(current_words.union(prev_words))
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)

        if not similarities:
            return 100.0

        # Diversity = 100 - (max similarity * 100)
        max_similarity = max(similarities)
        return max(0, 100 - (max_similarity * 100))

    def analyze_strategic_engagement(
        self, rule_text: str, explanation: str, internal_thought: str, game_state: Dict
    ) -> float:
        """Analyze strategic engagement and game understanding (0-100)"""
        score = 0.0

        # Strategic depth analysis
        strategic_word_count = 0
        all_text = (rule_text + " " + explanation + " " + internal_thought).lower()
        for word in self.strategic_keywords:
            strategic_word_count += all_text.count(word)
        score += min(30, strategic_word_count * 5)

        # Game state awareness
        if "points" in all_text:
            score += 15
        if "vote" in all_text or "voting" in all_text:
            score += 10
        if "rule" in all_text:
            score += 10

        # Complexity and thoughtfulness
        if len(internal_thought.split()) > 15:
            score += 15
        if len(explanation.split()) > 10:
            score += 10

        # Check for goal-oriented thinking
        goal_words = ["win", "winning", "advantage", "benefit", "help", "improve", "gain"]
        goal_thinking = sum(1 for word in goal_words if word in all_text)
        score += min(20, goal_thinking * 4)

        return min(100, score)

    def update_model_performance(self, model_name: str, proposal_data: Dict, game_outcome: Dict):
        """Update comprehensive performance metrics for a model"""
        metrics = self.get_or_create_metrics(model_name)

        # Basic performance updates
        metrics.proposals_made += 1
        if proposal_data.get("passed", False):
            metrics.proposals_passed += 1

        # Coherence analysis
        coherence_score = self.analyze_proposal_coherence(
            proposal_data["rule_text"], proposal_data["explanation"], proposal_data["internal_thought"]
        )
        metrics.coherence_scores.append(coherence_score)

        # Memory analysis
        memory_score = self.analyze_memory_usage(
            proposal_data["internal_thought"], proposal_data["explanation"], proposal_data.get("game_context", "")
        )
        if memory_score > 50:
            metrics.memory_reference_count += 1

        # Strategic engagement
        engagement_score = self.analyze_strategic_engagement(
            proposal_data["rule_text"],
            proposal_data["explanation"],
            proposal_data["internal_thought"],
            proposal_data.get("game_state", {}),
        )
        metrics.strategic_depth_scores.append(engagement_score)

        # Tie obsession detection
        if self.detect_tie_obsession(
            proposal_data["rule_text"], proposal_data["explanation"], proposal_data["internal_thought"]
        ):
            metrics.tie_related_proposals += 1

        # Diversity scoring
        diversity_score = self.calculate_proposal_diversity(
            proposal_data["rule_text"], proposal_data.get("all_previous_proposals", [])
        )
        metrics.proposal_diversity_scores.append(diversity_score)

        # Error tracking
        if proposal_data.get("validation_failed", False):
            metrics.validation_failures += 1
        if proposal_data.get("syntax_error", False):
            metrics.syntax_errors += 1

        # Update timestamp
        metrics.last_updated = datetime.now().isoformat()

        # Save metrics
        self.save_metrics()

    def get_weighted_model_selection(self, available_models: List[str]) -> List[str]:
        """Return models weighted by performance for random selection"""
        if not available_models:
            return available_models

        weights = []
        for model in available_models:
            if model in self.metrics:
                score = self.metrics[model].calculate_overall_score()
                # Convert score to weight (minimum weight of 10 for recovery)
                weight = max(10, score)
            else:
                # New models get average weight
                weight = 50
            weights.append(weight)

        # Create weighted list
        weighted_models = []
        for i, model in enumerate(available_models):
            count = int(weights[i] / 10)  # Scale down weights
            weighted_models.extend([model] * max(1, count))

        return weighted_models

    def get_model_statistics(self) -> Dict:
        """Get comprehensive statistics for all models"""
        stats = {}
        for model_name, metrics in self.metrics.items():
            stats[model_name] = {
                "overall_score": metrics.calculate_overall_score(),
                "success_rate": metrics.calculate_success_rate(),
                "coherence": metrics.calculate_average_coherence(),
                "engagement": metrics.calculate_strategic_engagement(),
                "games_played": metrics.total_games,
                "proposals_made": metrics.proposals_made,
                "tie_obsession": metrics.tie_related_proposals,
                "error_count": metrics.syntax_errors + metrics.validation_failures,
                "last_updated": metrics.last_updated,
            }
        return stats


class DeliberationManager:
    """Manages internal deliberation loops for strategic thinking"""

    def __init__(self, game_instance=None):
        self.game_instance = game_instance
        # Define 8 distinct proposal categories for diversity enforcement
        self.proposal_categories = {
            "point_distribution": {
                "keywords": ["points", "score", "gain", "lose", "award", "penalty", "distribute"],
                "description": "Rules about how points are gained, lost, or distributed",
                "examples": ["bonus points for voting", "point penalties for failed proposals", "point redistribution"],
            },
            "turn_mechanics": {
                "keywords": ["turn", "order", "skip", "extra", "sequence", "round"],
                "description": "Rules about turn order, timing, or turn-taking mechanics",
                "examples": ["change turn order", "extra turns", "skip turns", "simultaneous actions"],
            },
            "victory_conditions": {
                "keywords": ["win", "victory", "goal", "target", "achieve", "finish"],
                "description": "Rules that modify winning conditions or game end states",
                "examples": ["new victory conditions", "multiple win paths", "victory point modifications"],
            },
            "player_interaction": {
                "keywords": ["alliance", "team", "cooperation", "communication", "negotiate", "share"],
                "description": "Rules about player communication, alliances, or cooperation",
                "examples": ["alliance formation", "information sharing", "joint proposals"],
            },
            "resource_management": {
                "keywords": ["resource", "currency", "tokens", "cards", "items", "assets"],
                "description": "Rules introducing resources, currencies, or assets to manage",
                "examples": ["virtual currencies", "resource collection", "item trading"],
            },
            "information_systems": {
                "keywords": ["secret", "hidden", "reveal", "information", "knowledge", "private"],
                "description": "Rules about information visibility, secrets, or knowledge sharing",
                "examples": ["hidden information", "secret voting", "information reveals"],
            },
            "penalty_mechanisms": {
                "keywords": ["penalty", "punishment", "consequence", "fine", "restriction", "limit"],
                "description": "Rules that create penalties or restrictions for certain actions",
                "examples": ["voting penalties", "proposal restrictions", "action limits"],
            },
            "meta_rules": {
                "keywords": ["rule", "vote", "voting", "proposal", "change", "modify"],
                "description": "Rules about how rules are created, modified, or voted on",
                "examples": ["voting requirements", "proposal formats", "rule modification"],
            },
        }

        self.recent_categories = []  # Track recently used categories
        self.category_usage_history = {cat: 0 for cat in self.proposal_categories.keys()}

    def generate_nomic_context_header(self, game_state: Dict, player=None) -> str:
        """Generate enhanced Nomic game context header with temporal context and game history"""
        sorted_players = sorted(game_state["players"], key=lambda p: p["points"], reverse=True)
        
        # Calculate points needed to win for each player and handle ties better
        player_progress = []
        tied_players = []
        
        # Check if everyone is tied (early game scenario)
        all_same_points = all(p["points"] == sorted_players[0]["points"] for p in sorted_players)
        
        for i, p in enumerate(sorted_players):
            points_needed = max(0, 100 - p["points"])
            
            if all_same_points:
                # Everyone tied - show as equal footing
                rank_indicator = "" 
                status = "TIED"
            else:
                # Normal ranking
                rank_indicator = "" if i == 0 else "" if i == 1 else "" if i == 2 else f"#{i + 1}"
                status = f"Rank #{i + 1}"
                
            player_progress.append(f"{rank_indicator} {p['name']}: {p['points']}/100 pts (need {points_needed})")

        # Get current leader info
        leader = sorted_players[0]
        if all_same_points:
            leader_advantage = 0
            leader_status = "Everyone tied"
        else:
            leader_advantage = leader["points"] - sorted_players[1]["points"] if len(sorted_players) > 1 else 0
            leader_status = f"{leader['name']} leads"

        # Generate temporal context - recent game history
        temporal_context = self._generate_temporal_context(game_state, player)
        
        # Player-specific context with historical awareness
        player_context = ""
        if player:
            my_rank = next(i for i, p in enumerate(sorted_players, 1) if p["name"] == player.name)
            my_points = player.points
            points_to_win = 100 - my_points
            
            # Get player's turn position in the current round
            current_turn = game_state.get('turn', 1)
            turn_order_info = self._get_turn_order_context(player, game_state)
            
            # Add character-specific context
            character_context = ""
            if hasattr(player, 'character') and player.character:
                char = player.character
                character_context = f"""
 YOUR CHARACTER: {char.name} ({char.strategy})
 YOUR TRACK RECORD: {char.games_played} games played, {char.total_wins} wins ({char.get_win_rate()*100:.1f}% win rate)
 YOUR MEMORY: {char.persistent_thought if char.persistent_thought else "No persistent thoughts from previous games"}"""
                
            # Better status description for early game
            if all_same_points:
                status_description = f" TIED with all players at {my_points}/100 points ({points_to_win} points needed to WIN)"
                competitive_note = "- Everyone starts on equal footing - focus on good proposals that can pass"
            else:
                status_description = f" YOUR STATUS: Rank #{my_rank} with {my_points}/100 points ({points_to_win} points needed to WIN)"
                competitive_note = "- Focus on gaining advantage while getting unanimous support"
                
            player_context = f"""
{status_description}
 YOUR GOAL: Get to 100 points before anyone else does
 TURN ORDER: {turn_order_info}
 TEMPORAL POSITION: Turn {current_turn}, {self._get_game_phase_description(current_turn, leader['points'])}
{competitive_note}{character_context}"""

        context_header = f"""

 GAME: NOMIC (Self-Modifying Competitive Rule Game)
 VICTORY CONDITION: First player to reach 100 POINTS wins the game
 WIN OBJECTIVE: Propose and vote strategically to reach 100 points first


 CURRENT GAME STATUS (Turn {game_state.get('turn', 1)}):
{chr(10).join(player_progress)}

 VICTORY ANALYSIS:
- Current Situation: {leader_status} {"(everyone equal)" if all_same_points else f"with {leader['points']} points (ahead by {leader_advantage})"}
- Distance to Victory: {leader['name']} needs {100 - leader['points']} more points to WIN
- Game Stage: {(" Early game - all players equal, focus on fundament" if all_same_points else "Early stage") if leader['points'] < 70 else "LATE STAGE - Victory approaching!" if leader['points'] < 90 else "CRITICAL - Someone about to WIN!"}
{player_context}

{temporal_context}

 NOMIC RULES PRIMER:
- NOMIC is a game where players compete by changing the rules to help themselves win
- You propose RULES that affect: point scoring, turn order, voting, player interactions
- Valid Nomic rules affect GAMEPLAY: how points are gained/lost, how turns work, how voting works
- INVALID rules: Real-world topics (city planning, energy audits, external policies)

 EXAMPLES OF VALID NOMIC RULES (with mutual benefit appeal):
- "Players in last place gain 3 bonus points each turn" (helps trailing players - likely to get support)
- "All players gain 1 point for voting on proposals" (benefits everyone - encourages participation)  
- "Players gain 2 points for proposing creative rules" (incentivizes good proposals from all)
- "If no proposals pass for 2 turns, all players lose 2 points" (motivates cooperation)
- "Players who reach 90+ points lose 1 point per turn" (anti-runaway, appeals to non-leaders)

 INVALID (NOT NOMIC RULES):
- City council policies, energy audits, building regulations
- Real-world governance, environmental policies
- Rules about external games or non-Nomic activities

 NOMIC STRATEGIC REALITY:
- You are competing to WIN - reach 100 points first
- BUT: All proposals need UNANIMOUS SUPPORT to pass (100% agreement required)
- THEREFORE: You must propose rules that others will vote FOR, not against
- WINNING STRATEGY: Create rules that benefit you while being acceptable to others
- MUTUAL BENEFIT is key - purely selfish rules always fail in Nomic
- Consider: "How can this rule help me while also being appealing to others?"

 RULE ECOSYSTEM AWARENESS:
- Rules come in two types: MUTABLE (changeable) and IMMUTABLE (unchangeable)
- Mutable rules can be: amended (changed/neutralized) or transmuted
- Immutable rules can only be transmuted to mutable (then amended)
- STRATEGIC QUESTION: Are harmful rules mutable? Can beneficial rules be protected?

{self._get_rule_status_display(game_state)}

"""
        return context_header

    def _get_rule_status_display(self, game_state: Dict) -> str:
        """Display current rules with their mutable/immutable status"""
        try:
            if hasattr(self, 'game_instance') and self.game_instance and hasattr(self.game_instance, 'rules'):
                display = " CURRENT RULE STATUS:\n"
                
                # Show key immutable rules
                display += " IMMUTABLE RULES (cannot be changed directly):\n"
                key_immutable = [
                    ("117", "Zero points = everyone loses immediately"),
                    ("109", "Transmuting immutable→mutable needs unanimous vote"),
                    ("112", "Victory at 100 points cannot be changed")
                ]
                for rule_id, desc in key_immutable:
                    display += f"   Rule {rule_id}: {desc}\n"
                
                # Show all mutable rules with emphasis on harmful ones
                display += "\n MUTABLE RULES (CAN be amended!):\n"
                
                # Check for Rule 206 specifically
                for rule in self.game_instance.rules.get('mutable', []):
                    if rule.id == 206:
                        display += f"   Rule 206: Failed proposals lose 10 points [CHANGEABLE - CAUSING PAIN!]\n"
                        break
                
                # Show other mutable rules
                for rule in self.game_instance.rules.get('mutable', []):
                    if rule.id >= 301:  # Player-created rules
                        display += f"   Rule {rule.id}: {rule.text[:60]}...\n"
                
                display += "\n STRATEGIC INSIGHT: Rules causing problems can be changed if they're mutable!\n"
                return display
            else:
                return ""
        except:
            return ""

    def _generate_temporal_context(self, game_state: Dict, player=None) -> str:
        """Generate temporal context showing recent game history and trends"""
        current_turn = game_state.get('turn', 1)
        
        # Get recent proposal history (last 3-5 proposals)
        recent_proposals = self._get_recent_proposals(game_state)
        
        # Get voting patterns from recent turns
        voting_trends = self._get_voting_trends(game_state)
        
        # Generate player-specific temporal insights
        player_history = ""
        if player and hasattr(self, 'game_instance') and self.game_instance:
            player_history = self._get_player_temporal_insights(player, game_state)
        
        temporal_section = f""" TEMPORAL CONTEXT & GAME HISTORY:
{recent_proposals}
{voting_trends}
{player_history}"""
        
        return temporal_section
    
    def _get_recent_proposals(self, game_state: Dict) -> str:
        """Get summary of recent proposals and outcomes with failure analysis"""
        try:
            # Access the actual game instance for full proposal history
            if hasattr(self, 'game_instance') and self.game_instance and hasattr(self.game_instance, 'proposals'):
                proposals = self.game_instance.proposals[-5:]  # Last 5 proposals
                
                if not proposals:
                    return " RECENT PROPOSALS: No proposals have been made yet"
                
                proposal_summary = " RECENT PROPOSAL HISTORY (Learn from failures!):\n"
                
                for proposal in proposals:
                    # Calculate vote results
                    if proposal.votes:
                        ayes = sum(proposal.votes.values())
                        total = len(proposal.votes)
                        percentage = (ayes / total) * 100 if total > 0 else 0
                        passed = percentage >= 100  # Assuming unanimous requirement
                        
                        result_emoji = "" if passed else ""
                        vote_summary = f"({ayes}/{total} = {percentage:.0f}%)"
                        
                        # Analyze why failed proposals failed
                        failure_reason = ""
                        if not passed:
                            if percentage < 25:
                                failure_reason = " [REJECTED: Too selfish/harmful to others]"
                            elif percentage < 50:
                                failure_reason = " [REJECTED: Benefits proposer more than others]"
                            elif percentage < 75:
                                failure_reason = " [REJECTED: Not enough mutual benefit]"
                            else:
                                failure_reason = " [REJECTED: Almost passed - small tweaks needed]"
                        
                        proposal_summary += f"   {result_emoji} \"{proposal.rule_text[:50]}...\" {vote_summary}{failure_reason}\n"
                    else:
                        proposal_summary += f"    \"{proposal.rule_text[:50]}...\" (voting in progress)\n"
                
                # Add learning note
                failed_count = len([p for p in proposals if p.votes and sum(p.votes.values()) < len(p.votes)])
                if failed_count > 0:
                    proposal_summary += f"\n LEARNING: {failed_count} recent proposals failed - focus on MUTUAL BENEFIT to get unanimous support!"
                
                return proposal_summary
            else:
                # Fallback to simple display
                rules = game_state.get('mutable_rules', [])
                if not rules:
                    return " RECENT PROPOSALS: No rules have been successfully passed yet"
                
                recent_rules_text = " RECENT SUCCESSFUL RULES:\n"
                for i, rule in enumerate(rules[-3:], 1):  # Last 3 rules
                    recent_rules_text += f"   {i}. {rule.get('text', 'Unknown rule')[:60]}...\n"
                
                return recent_rules_text
                
        except Exception as e:
            return " RECENT PROPOSALS: Unable to load proposal history"
    
    def _get_voting_trends(self, game_state: Dict) -> str:
        """Analyze recent voting patterns"""
        # Basic voting trend analysis
        return "  VOTING TRENDS: Players showing strategic voting patterns based on position"
    
    def _get_player_temporal_insights(self, player, game_state: Dict) -> str:
        """Get player-specific historical insights"""
        current_turn = game_state.get('turn', 1)
        
        insights = f""" YOUR RECENT PERFORMANCE:
   - Turn History: You are in turn {current_turn} of the game
   - Strategy Notes: Consider what worked/failed in previous proposals
   - Learning Opportunity: Adapt based on other players' voting patterns"""
        
        return insights
    
    def _get_turn_order_context(self, player, game_state: Dict) -> str:
        """Generate turn order context for the player"""
        current_turn = game_state.get('turn', 1)
        players = game_state.get('players', [])
        
        # Simple turn order description
        if len(players) >= 4:
            player_names = [p['name'] for p in players]
            try:
                my_index = player_names.index(player.name)
                next_index = (my_index + 1) % len(players)
                prev_index = (my_index - 1) % len(players)
                
                return f"After {player_names[prev_index]}, before {player_names[next_index]}"
            except ValueError:
                return "Turn order position unknown"
        
        return "Turn order being determined"
    
    def _get_game_phase_description(self, current_turn: int, leader_points: int) -> str:
        """Describe what phase of the game we're in"""
        # Check for crisis mode (everyone in negative points)
        if hasattr(self, 'game_instance') and self.game_instance:
            all_negative = all(p.points < 0 for p in self.game_instance.players)
            if all_negative:
                return "CRISIS MODE - All players negative! Rule fixes needed urgently!"
        
        if leader_points < 0:
            return "CRISIS PHASE - Negative points detected, recovery strategies essential"
        elif current_turn <= 2:
            return "Opening phase - establishing early strategy"
        elif current_turn <= 5:
            return "Early game - rule foundation building"
        elif leader_points < 70:
            return "Mid-game - competitive positioning"
        elif leader_points < 90:
            return "Late game - victory approaching"
        else:
            return "End game - critical final moves"

    def _get_player_scratchpad_content(self, player, game_state: Dict) -> str:
        """Get player's scratchpad content for display in prompts"""
        if not hasattr(self, 'game_instance') or not self.game_instance:
            return ""
        
        # Try to get scratchpad from player state
        player_states = game_state.get('player_states', {})
        player_state = player_states.get(str(player.id), {})
        scratchpad = player_state.get('scratchpad', [])
        
        if not scratchpad:
            return " SCRATCHPAD: Empty - start taking notes for future reference!"
        
        scratchpad_text = " YOUR SCRATCHPAD NOTES:\n"
        for i, note in enumerate(scratchpad[-5:], 1):  # Show last 5 notes
            scratchpad_text += f"   {i}. {note}\n"
        
        if len(scratchpad) > 5:
            scratchpad_text += f"   ... and {len(scratchpad) - 5} older notes\n"
        
        return scratchpad_text
    
    def _get_temporal_reminders(self, player, game_state: Dict) -> str:
        """Generate temporal reminders about recent game events"""
        current_turn = game_state.get('turn', 1)
        
        # Get player's recent learning observations
        player_states = game_state.get('player_states', {})
        player_state = player_states.get(str(player.id), {})
        recent_observations = player_state.get('learning_observations', [])
        
        # Get recent failed proposals
        recent_failures = [obs for obs in recent_observations if 'Failed proposal' in obs]
        
        temporal_text = f" TEMPORAL REMINDERS (Turn {current_turn}):\n"
        
        if recent_failures:
            temporal_text += "    Recent lessons learned:\n"
            for failure in recent_failures[-3:]:  # Last 3 failures
                temporal_text += f"      - {failure[:80]}...\n"
        else:
            temporal_text += "   - This is early in your strategic development\n"
        
        # Add turn progression context
        if current_turn == 1:
            temporal_text += "   - Opening turn - set foundation for your strategy\n"
        elif current_turn <= 3:
            temporal_text += "   - Early game - establish your competitive position\n"
        else:
            temporal_text += "   - Game progressing - adapt based on what you've learned\n"
        
        return temporal_text

    def categorize_proposal(self, rule_text: str, explanation: str) -> str:
        """Categorize a proposal into one of the 8 categories"""
        combined_text = (rule_text + " " + explanation).lower()

        category_scores = {}
        for category, data in self.proposal_categories.items():
            score = 0
            for keyword in data["keywords"]:
                score += combined_text.count(keyword)
            category_scores[category] = score

        # Return category with highest score, or 'meta_rules' as default
        if max(category_scores.values()) == 0:
            return "meta_rules"
        return max(category_scores, key=category_scores.get)

    def is_category_overused(self, category: str, turns_since_last: int = 3) -> bool:
        """Check if a category has been used too recently"""
        if len(self.recent_categories) < turns_since_last:
            return False
        return category in self.recent_categories[-turns_since_last:]

    def get_underused_categories(self) -> List[str]:
        """Get categories that haven't been used much"""
        min_usage = min(self.category_usage_history.values())
        return [cat for cat, usage in self.category_usage_history.items() if usage == min_usage]

    def record_category_usage(self, category: str):
        """Record that a category has been used"""
        self.recent_categories.append(category)
        self.recent_categories = self.recent_categories[-10:]  # Keep last 10
        self.category_usage_history[category] += 1

    def _customize_prompt_for_character(self, base_prompt: str, player) -> str:
        """Customize prompts based on character traits and personality"""
        if not (hasattr(player, 'character') and player.character and hasattr(player.character, 'traits')):
            return base_prompt
        
        char = player.character
        traits = char.traits
        
        # Add character-specific personality adjustments
        personality_adjustments = ""
        
        # Risk tolerance adjustments
        if traits.risk_tolerance == "high":
            personality_adjustments += " As a HIGH RISK character, favor bold, aggressive moves that could pay off big. Don't be afraid of failure - big risks bring big rewards.\n"
        elif traits.risk_tolerance == "low":
            personality_adjustments += " As a LOW RISK character, prioritize safe, reliable strategies. Focus on steady progress over risky gambles.\n"
        
        # Cooperation level adjustments  
        if traits.cooperation_level == "cooperative":
            personality_adjustments += " As a COOPERATIVE character, seek win-win solutions. Frame proposals to benefit multiple players.\n"
        elif traits.cooperation_level == "competitive":
            personality_adjustments += " As a COMPETITIVE character, focus on getting ahead of opponents. Look for advantages others might miss.\n"
        
        # Strategic focus adjustments
        if traits.strategic_focus == "analytical":
            personality_adjustments += " Use your ANALYTICAL nature - examine data, patterns, and logical connections in detail.\n"
        elif traits.strategic_focus == "creative":
            personality_adjustments += " Use your CREATIVE instincts - think outside the box and propose unconventional solutions.\n"
        elif traits.strategic_focus == "aggressive":
            personality_adjustments += " Use your AGGRESSIVE approach - make decisive moves and put pressure on opponents.\n"
        elif traits.strategic_focus == "diplomatic":
            personality_adjustments += " Use your DIPLOMATIC skills - build consensus and find solutions that appeal to everyone.\n"
        
        # Decision speed adjustments
        if traits.decision_speed == "fast":
            personality_adjustments += " Trust your quick instincts - don't overthink obvious opportunities.\n"
        elif traits.decision_speed == "slow":
            personality_adjustments += " Take time to thoroughly consider all angles before deciding.\n"
        
        # Add personality section to prompt
        if personality_adjustments:
            character_section = f"""
 YOUR CHARACTER TRAITS ({char.name}):
{personality_adjustments}
Remember: Play true to your character's personality while pursuing victory.
"""
            # Insert after the role description
            base_prompt = base_prompt.replace(
                f"Your Role: {player.role}",
                f"Your Role: {player.role}\n{character_section}"
            )
        
        return base_prompt

    def generate_turn_choice_prompt(self, player, game_state: Dict) -> str:
        """Let the model choose how many deliberation turns it wants (1-3)"""
        context_header = self.generate_nomic_context_header(game_state, player)

        base_prompt = f"""{context_header}

 DELIBERATION PLANNING PHASE

Your Role: {player.role}
Current Position: Rank #{game_state.get('rank', '?')} in the 100-point race to victory

You can choose how many deliberation turns to use for your proposal strategy:

 **1 TURN** - Quick Tactical Decision
- Use when you see an immediate opportunity
- Fast reaction to current game state
- Good for simple, obvious moves

 **2 TURNS** - Balanced Strategic Planning  
- Standard approach for most situations
- Time to consider alternatives and opponents
- Good balance of depth vs efficiency

 **3 TURNS** - Deep Strategic Analysis
- Use for complex game situations
- Comprehensive analysis of all factors
- When you need maximum strategic depth

SITUATION ANALYSIS:
- Game Turn: {game_state.get('turn', 1)}
- Leader has {max([p['points'] for p in game_state.get('players', [])])} points
- You need {100 - player.points} points to win
- Recent rule changes: {len(game_state.get('proposals_history', []))} proposals so far

Choose your approach and explain your reasoning. You'll get 2000+ tokens per turn you choose.

RESPOND WITH:
- Your chosen number of turns (1, 2, or 3)
- Brief reasoning for your choice
- Any initial strategic direction"""
        
        # Apply character customization
        return self._customize_prompt_for_character(base_prompt, player)

    def generate_flexible_deliberation_prompt(
        self, player, game_state: Dict, turn_num: int, total_turns: int, previous_results: Dict
    ) -> str:
        """Generate enhanced deliberation prompts with temporal context and scratchpad support"""
        context_header = self.generate_nomic_context_header(game_state, player)

        # Include previous deliberation insights
        previous_insights = ""
        if previous_results:
            previous_insights = "\n YOUR PREVIOUS DELIBERATION:\n"
            for key, value in previous_results.items():
                if key.startswith("turn_"):
                    turn_number = key.split("_")[1]
                    previous_insights += f"Turn {turn_number}: {value[:200]}...\n"

        # Get player's scratchpad content and game history
        scratchpad_content = self._get_player_scratchpad_content(player, game_state)
        temporal_reminders = self._get_temporal_reminders(player, game_state)

        base_prompt = f"""{context_header}
{previous_insights}
{scratchpad_content}
{temporal_reminders}

 DELIBERATION TURN {turn_num}/{total_turns}

Your Role: {player.role}
Current Approach: {total_turns}-turn deliberation

FREE-FORM STRATEGIC THINKING:
You have 2000+ tokens to think deeply about your Nomic strategy. Consider any aspects you find relevant:

- Current game dynamics and player positions
- Rule opportunities and gaps in the current system  
- How your role gives you unique advantages
- Short-term vs long-term strategic considerations
- What other players might be planning
- Creative rule ideas that could help you win
- Learn from your previous proposal successes/failures
- Adapt to voting patterns you've observed

THINK HOWEVER WORKS BEST FOR YOU. No rigid format required.

 SCRATCHPAD NOTES:
Feel free to add notes to your persistent scratchpad by using:
SCRATCHPAD_NOTE: [important insight to remember for future turns]

This will be saved and shown to you in future deliberations. Use it for:
- Patterns you've noticed in other players' behavior
- Ideas for future proposals
- Lessons learned from failed proposals
- Strategic insights worth remembering

Optional: If you want to preserve an insight for the current deliberation, use:
INSIGHT: [key insight you want to remember]

Focus on developing a winning strategy to reach 100 points first."""
        
        # Apply character customization
        return self._customize_prompt_for_character(base_prompt, player)

    def generate_final_proposal_prompt(
        self, player, game_state: Dict, deliberation_results: Dict, total_turns: int
    ) -> str:
        """Generate final proposal prompt using all deliberation context"""
        context_header = self.generate_nomic_context_header(game_state, player)

        # Compile all deliberation thinking
        deliberation_summary = " YOUR COMPLETE DELIBERATION:\n"
        for key, value in deliberation_results.items():
            if key.startswith("turn_"):
                turn_number = key.split("_")[1]
                deliberation_summary += f"\nTurn {turn_number}: {value}\n"

        return f"""{context_header}

{deliberation_summary}

 FINAL PROPOSAL CREATION

Based on your {total_turns}-turn deliberation above, now create your actual rule proposal.

You have 2000+ tokens for final proposal crafting. Think creatively and strategically.

REQUIREMENTS - You MUST end with this exact format:

INTERNAL_THOUGHT: [Your private strategic reasoning and thought process]
RULE_TEXT: [The exact rule text for the game system]  
EXPLANATION: [Your public explanation for why this rule is good]

The game system requires these three sections in this format to process your proposal.
Everything before these sections can be free-form thinking."""

    def generate_final_proposal_prompt_with_checkboxes(
        self, player, game_state: Dict, deliberation_results: Dict, total_turns: int
    ) -> str:
        """Generate final proposal prompt with checkbox system for rule effects"""
        context_header = self.generate_nomic_context_header(game_state, player)

        # Compile all deliberation thinking
        deliberation_summary = " YOUR COMPLETE DELIBERATION:\n"
        for key, value in deliberation_results.items():
            if key.startswith("turn_"):
                turn_number = key.split("_")[1]
                deliberation_summary += f"\nTurn {turn_number}: {value}\n"

        return f"""{context_header}

{deliberation_summary}

 FINAL PROPOSAL CREATION WITH RULE EFFECTS CHECKBOXES

Based on your {total_turns}-turn deliberation above, now create your actual rule proposal.

You have 2000+ tokens for final proposal crafting. Think creatively and strategically.

 PROPOSAL TYPES AVAILABLE:
1. NEW RULE: Create a completely new rule for the game
2. TRANSMUTATION: Change an existing rule from mutable ↔ immutable
   - Format: "Transmute rule [number]" (e.g., "Transmute rule 301")
   - STRATEGIC POWER: This changes rule mutability status!
   - Mutable rules can be amended, immutable rules cannot
   - EXAMPLES: 
     * Transmute Rule 117 to mutable (then amend to disable zero-point rule)
     * Transmute beneficial rules to immutable (protect them from changes)
     * Transmute opponent's favorable rules to mutable (then amend them)
   - Consider which rules would benefit you if made mutable/immutable
3. AMENDMENT: Modify an existing MUTABLE rule
   - Format: "Amend rule [number] to: [new text]"
   - Can only amend MUTABLE rules!
   - Strategic uses: Fix harmful rules, adjust point values, change conditions
   - Note: Amendments can completely change a rule's meaning (de facto repeal)

 CRISIS MODE STRATEGIC THINKING:
When everyone has negative points, consider:
- What rule is causing the most pain? (Is it mutable?)
- Would everyone vote to fix a rule that hurts everyone?
- Sometimes the best move is fixing broken rules, not adding new ones
- Amendment can neutralize harmful rules or change their effects entirely

REQUIREMENTS - You MUST end with this exact format:

INTERNAL_THOUGHT: [Your private strategic reasoning and thought process]
RULE_TEXT: [The exact rule text for the game system]  
EXPLANATION: [Your public explanation for why this rule is good]
EFFECTS: [Fill out this checkbox system to help the engine understand your rule]

 RULE EFFECTS CHECKLIST (Mark with  for YES,  for NO):

POINTS SYSTEM:
 Add points (how many? __ to whom? __)
 Subtract points (how many? __ from whom? __)
 Steal points (how many? __ from/to whom? __)
 Redistribute points between players

VOTING MECHANICS:
 Change voting threshold (to what percentage? __)
 Require unanimous consent
 Change majority requirements
 Add voting restrictions

TURN MECHANICS:
 Skip turns (whose? __ when? __)
 Extra turns (for whom? __ when? __)
 Change turn order
 Add time limits

WIN CONDITIONS:
 Modify win threshold (to what? __)
 Add alternative win conditions
 Change victory requirements

SPECIAL MECHANICS:
 Add dice/random elements
 Add conditional triggers (if/when)
 Create compound effects
 Add resource systems

The game system requires these sections in this format to process your proposal.
Everything before these sections can be free-form thinking."""

    def generate_deliberation_prompts(self, player, game_state: Dict, deliberation_turn: int) -> str:
        """Legacy method - generate prompts for each deliberation turn"""

        if deliberation_turn == 1:
            return self._generate_state_analysis_prompt(player, game_state)
        elif deliberation_turn == 2:
            return self._generate_gap_analysis_prompt(player, game_state)
        elif deliberation_turn == 3:
            return self._generate_category_selection_prompt(player, game_state)
        elif deliberation_turn == 4:
            return self._generate_impact_modeling_prompt(player, game_state)
        elif deliberation_turn == 5:
            return self._generate_final_selection_prompt(player, game_state)
        else:
            return self._generate_final_selection_prompt(player, game_state)

    def _generate_state_analysis_prompt(self, player, game_state: Dict) -> str:
        """Turn 1: Analyze current position and threats - optimized for efficiency"""
        context_header = self.generate_nomic_context_header(game_state, player)
        
        # Check if this is a simple game state that doesn't need deep analysis
        players_data = game_state.get('players', [])
        early_game = len(players_data) <= 4 and all(abs(p.get('points', 50) - 50) <= 10 for p in players_data)
        
        if early_game:
            return f"""{context_header}

 QUICK ANALYSIS: Early game, similar positions.

Your Role: {player.role}

FOCUS: What rule would help YOUR role most right now?

OUTPUT FORMAT:
POSITION_ANALYSIS: [Your competitive situation - be brief]
PATTERN_OBSERVATION: [What gaps exist in current rules]

Keep it concise - save deep strategy for your actual proposal."""

        return f"""{context_header}

 DELIBERATION TURN 1: STRATEGIC ANALYSIS

Your Role: {player.role}
Position: #{game_state.get('rank', '?')} with {player.points} points

FOCUS ON:
1. Who's your biggest threat to reaching 100?
2. What's your path to victory?
3. What patterns can you exploit?

OUTPUT FORMAT:
THREAT_ASSESSMENT: [Key threat and why - one sentence]
POSITION_ANALYSIS: [Your situation - brief]
PATTERN_OBSERVATION: [Exploitable trends - brief]

 Goal: Win efficiently. Be strategic, not cooperative."""

    def _generate_gap_analysis_prompt(self, player, game_state: Dict) -> str:
        """Turn 2: Identify strategic gaps and opportunities - simplified"""
        context_header = self.generate_nomic_context_header(game_state, player)

        return f"""{context_header}

 DELIBERATION TURN 2: RULE GAP ANALYSIS

Your Role: {player.role}

EXISTING RULES:
{chr(10).join([f"Rule {r['id']}: {r['text']}" for r in game_state.get('mutable_rules', [])[-5:]])}

QUICK QUESTIONS:
1. What's missing that your role could exploit?
2. What point-scoring opportunities exist?
3. Which category suits your strategy?

Categories: Point Distribution, Turn Mechanics, Victory Conditions, Player Interaction, Resource Management, Information Systems, Penalty Mechanisms

OUTPUT FORMAT:
EXPLOITABLE_GAPS: [One specific opportunity]
TOP_CATEGORIES: [Pick 1-2 that help YOU most]
ROLE_ADVANTAGE: [How your role helps - brief]

 Focus on YOUR advantage, not game balance."""

    def _generate_category_selection_prompt(self, player, game_state: Dict) -> str:
        """Turn 3: Select specific category and approach"""
        context_header = self.generate_nomic_context_header(game_state, player)
        underused = self.get_underused_categories()
        overused = [cat for cat in self.proposal_categories.keys() if self.is_category_overused(cat)]

        return f"""{context_header}

 DELIBERATION TURN 3: NOMIC CATEGORY SELECTION & STRATEGIC APPROACH

Your Role in this NOMIC game: {player.role}
Your Strategic Position: {game_state.get('position_analysis', 'Unknown')}

NOMIC CATEGORY AVAILABILITY ANALYSIS:
UNDERUSED NOMIC CATEGORIES (Recommended): {', '.join(underused)}
OVERUSED NOMIC CATEGORIES (Avoid): {', '.join(overused)}

STRATEGIC NOMIC CATEGORY EVALUATION:
For each available Nomic category, consider:

1. POINT DISTRIBUTION SYSTEMS (in Nomic):
   - Could you create Nomic rules that favor your position?
   - Are there point-scoring opportunities others missed in Nomic?

2. TURN MECHANICS (in Nomic):
   - Could you gain extra turns or disrupt others' turns in Nomic?
   - Are there timing advantages you could exploit in Nomic?

3. VICTORY CONDITIONS (in Nomic):
   - Could you create alternative Nomic win conditions that favor you?
   - Are there shortcuts to 100 points others haven't considered?

4. PLAYER INTERACTION (in Nomic):
   - Could you force alliances that benefit you in Nomic?
   - Are there Nomic cooperation rules that help your position?

5. RESOURCE MANAGEMENT (in Nomic):
   - Could you introduce Nomic resources you'd be good at managing?
   - Are there scarcity mechanics that favor your Nomic strategy?

6. INFORMATION SYSTEMS (in Nomic):
   - Could you create information advantages for yourself in Nomic?
   - Are there secret Nomic mechanisms that help your role?

7. PENALTY MECHANISMS (in Nomic):
   - Could you penalize Nomic behaviors that hurt your chances?
   - Are there Nomic restrictions that favor your playstyle?

OUTPUT FORMAT:
SELECTED_CATEGORY: [Your chosen Nomic category]
STRATEGIC_RATIONALE: [Why this Nomic category gives YOU the best advantage]
COMPETITIVE_ANGLE: [How this Nomic rule hurts your competitors or helps your position]
SPECIFIC_APPROACH: [The general type of Nomic rule you'll propose]

 Choose based on YOUR winning chances in NOMIC, not fairness to others."""

    def _generate_impact_modeling_prompt(self, player, game_state: Dict) -> str:
        """Turn 4: Model impacts of potential approaches"""
        context_header = self.generate_nomic_context_header(game_state, player)

        return f"""{context_header}

 DELIBERATION TURN 4: NOMIC IMPACT MODELING & STRATEGIC OPTIMIZATION

Your Selected Nomic Category: {game_state.get('selected_category', 'Unknown')}
Your Current Position: #{game_state.get('rank', '?')} with {player.points} points in the 100-point race

NOMIC IMPACT MODELING EXERCISE:
For your chosen Nomic approach, analyze the following impacts:

IMPACT ON YOU IN NOMIC:
- How many points could you potentially gain in this Nomic game?
- How does this improve your winning chances to reach 100 points?
- What strategic advantages does this give you in Nomic?
- Are there any risks or downsides for you in this Nomic rule?

IMPACT ON COMPETITORS IN NOMIC:
- Who gets hurt most by this Nomic rule?
- Who benefits least from this Nomic rule?
- Will this Nomic rule slow down the current leader?
- Does this create problems for your biggest threats in the 100-point race?

VOTING LIKELIHOOD ANALYSIS FOR NOMIC:
- Which players have incentive to vote FOR this Nomic rule?
- Which players have incentive to vote AGAINST this Nomic rule?
- How can you frame this Nomic rule to get the required votes?
- What arguments would convince the most players in Nomic?

LONG-TERM NOMIC CONSEQUENCES:
- How does this affect the rest of the Nomic game?
- Could this Nomic rule backfire on you later?
- Does this set up future advantages for you in Nomic?
- Are there follow-up Nomic rules you could propose next?

OUTPUT FORMAT:
PERSONAL_BENEFIT: [Specific advantages you gain in Nomic]
COMPETITIVE_DAMAGE: [How this Nomic rule hurts your rivals]
VOTE_STRATEGY: [How to frame this Nomic rule to win votes]
RISK_ASSESSMENT: [Potential downsides and mitigation in Nomic]
LONG_TERM_PLAN: [How this fits your overall Nomic strategy]

 Focus on maximizing YOUR advantage in NOMIC while getting enough votes to pass."""

    def _generate_final_selection_prompt(self, player, game_state: Dict) -> str:
        """Turn 5: Craft the final proposal - with successful examples"""
        context_header = self.generate_nomic_context_header(game_state, player)

        return f"""{context_header}

 FINAL PROPOSAL: Create your rule

Position: #{game_state.get('rank', '?')} with {player.points} points

SUCCESSFUL PROPOSAL EXAMPLES (study these patterns):
 "Players gain 1 point when they vote for a successful proposal." - Universal benefit
 "When a rule is repealed, all players gain 2 points." - Shared incentive
 "Players in last place gain 2 bonus points at turn start." - Helps underdogs
 "If no proposals pass for 2 turns, all players gain 3 points." - Prevents deadlock

WHY THESE WORK:
- Give EVERYONE something (crucial for unanimous votes)
- Help trailing players (most players aren't winning)
- Don't obviously target the leader
- Simple and clear

YOUR PROPOSAL STRATEGY:
1. Include a benefit for ALL players
2. Give extra help to non-leaders
3. Keep rule under 25 words
4. Avoid obvious self-dealing

OUTPUT FORMAT:
INTERNAL_THOUGHT: [Your real strategy - be honest]
RULE_TEXT: [Clear rule under 25 words - must benefit others too]
EXPLANATION: [Why this helps everyone and improves the game]

 Key: Get everyone's vote by giving everyone something."""

    def _parse_scratchpad_notes(self, response: str) -> List[str]:
        """Parse scratchpad notes from AI response"""
        scratchpad_notes = []
        
        # Look for SCRATCHPAD_NOTE: patterns
        scratchpad_pattern = r"SCRATCHPAD_NOTE:\s*(.*?)(?:\n|$)"
        matches = re.findall(scratchpad_pattern, response, re.MULTILINE)
        
        for match in matches:
            note = match.strip()
            if note and len(note) > 10:  # Only meaningful notes
                scratchpad_notes.append(note[:200])  # Limit length
        
        return scratchpad_notes
    
    def _save_scratchpad_notes(self, player_id: int, notes: List[str]):
        """Save scratchpad notes to player state"""
        if not notes or not self.game_instance:
            return
        
        try:
            # Get or create player state
            if not hasattr(self.game_instance, 'state_tracker'):
                return
            
            state = self.game_instance.state_tracker.get_player_state(player_id)
            if state is None:
                # Initialize state if it doesn't exist
                from dataclasses import dataclass
                player = next((p for p in self.game_instance.players if p.id == player_id), None)
                if player:
                    self.game_instance.state_tracker.initialize_player_state(player)
                    state = self.game_instance.state_tracker.get_player_state(player_id)
                else:
                    return  # Can't find player
            
            if not hasattr(state, 'scratchpad'):
                state.scratchpad = []
            
            # Add new notes
            for note in notes:
                state.scratchpad.append(f"Turn {getattr(self.game_instance, 'turn_number', '?')}: {note}")
            
            # Keep only last 20 notes to prevent bloat
            state.scratchpad = state.scratchpad[-20:]
            
            # Log the scratchpad update
            if hasattr(self.game_instance, 'game_logger'):
                self.game_instance.game_logger.log_scratchpad_update(
                    player_id, notes, len(state.scratchpad)
                )
            
            # State is automatically updated since we modified the object directly
            
        except Exception as e:
            print(f"Error saving scratchpad notes: {e}")


class BehavioralAnalyzer:
    """Advanced behavioral analysis for model quality assessment"""

    def __init__(self):
        self.suspicious_patterns = {
            "random_text": ["lorem", "ipsum", "test", "example", "placeholder"],
            "nonsense": ["asdf", "qwerty", "xyz", "123", "abc"],
            "repetitive": [],  # Will be populated dynamically
        }

    def detect_random_behavior(self, proposal_text: str, explanation: str, internal_thought: str) -> bool:
        """Detect if model is throwing random stuff instead of playing strategically"""
        combined_text = (proposal_text + " " + explanation + " " + internal_thought).lower()

        # Check for suspicious patterns
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in combined_text:
                    return True

        # Check for extremely short or long responses (likely random)
        if len(proposal_text.strip()) < 5 or len(proposal_text) > 200:
            return True

        # Check for lack of game-related words
        game_words = ["rule", "vote", "player", "point", "turn", "game", "win", "propose"]
        game_word_count = sum(1 for word in game_words if word in combined_text)
        total_words = len(combined_text.split())

        if total_words > 10 and game_word_count == 0:
            return True

        return False

    def assess_game_understanding(self, proposal: str, game_context: str) -> float:
        """Assess how well the model understands Nomic mechanics (0-100)"""
        score = 0.0
        proposal_lower = proposal.lower()

        # Understanding of basic concepts
        if "rule" in proposal_lower:
            score += 20
        if "vote" in proposal_lower or "voting" in proposal_lower:
            score += 15
        if "point" in proposal_lower or "points" in proposal_lower:
            score += 15

        # Understanding of game mechanics
        if "unanimous" in proposal_lower or "majority" in proposal_lower:
            score += 20
        if "mutable" in proposal_lower or "immutable" in proposal_lower:
            score += 25
        if "propose" in proposal_lower or "proposal" in proposal_lower:
            score += 10

        # Sophistication indicators
        if "if" in proposal_lower and "then" in proposal_lower:
            score += 15
        if len(proposal.split()) > 20:  # Detailed rules
            score += 10

        return min(100, score)


class OllamaManager:
    """Manages multiple Ollama instances on different ports"""

    def __init__(self, validator_port: int = 11435):
        self.base_port = 11434  # Main Ollama service (shared)
        self.validator_port = validator_port  # Dedicated validator port
        self.player_ports = {}

    def assign_ports(self, players):
        """Assign unique ports to players"""
        for i, player in enumerate(players):
            port = self.validator_port + i + 1  # Start after validator port
            self.player_ports[player.id] = port
            player.port = port

    def generate(self, model: str, prompt: str, port: int = 11434, temperature: float = 0.7, max_tokens: int = 500):
        """Generate text using specific Ollama instance"""
        try:
            response = requests.post(
                f"http://localhost:{port}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=15,
            )
            return response.json().get("response", f"Error from port {port}")
        except Exception as e:
            return f"Error calling {port}: {str(e)}"

    def validate_rule(self, rule_text: str, context: str):
        """Use larger model to validate rule clarity and completeness"""
        prompt = f"""RULE VALIDATION TASK

PROPOSED RULE: "{rule_text}"

GAME CONTEXT:
{context}

As a Nomic rule validator, assess this rule:

1. CLARITY: Is the rule clearly written and unambiguous?
2. COMPLETENESS: Does it specify all necessary conditions?
3. CONFLICTS: Does it conflict with existing rules?
4. ENFORCEABILITY: Can this rule be clearly enforced?

Respond in this format:
VALID: Yes/No
ISSUES: [List any problems, or "None"]
SUGGESTION: [How to improve, or "Rule is acceptable"]
"""

        try:
            # Use larger model on validator port for better analysis
            response = self.generate("llama3:8b", prompt, self.validator_port, temperature=0.3, max_tokens=300)
            return self.parse_validation_response(response)
        except Exception as e:
            return {"valid": True, "issues": f"Validation error: {str(e)}", "suggestion": ""}

    def parse_validation_response(self, response: str):
        """Parse validation response into structured format"""
        valid_match = re.search(r"VALID:\s*(Yes|No)", response, re.IGNORECASE)
        issues_match = re.search(r"ISSUES:\s*(.*?)(?:SUGGESTION:|$)", response, re.DOTALL)
        suggestion_match = re.search(r"SUGGESTION:\s*(.*)", response, re.DOTALL)

        return {
            "valid": valid_match.group(1).lower() == "yes" if valid_match else True,
            "issues": issues_match.group(1).strip() if issues_match else "None",
            "suggestion": suggestion_match.group(1).strip() if suggestion_match else "",
        }


class ProperNomicGame:
    def __init__(
        self, num_players=6, provider: Union[OllamaClient, OllamaManager, OpenRouterClient] = None, include_human=False, session_dir: str = None
    ):
        self.num_players = num_players
        self.include_human = include_human
        self.turn_number = 1
        self.game_over = False
        self.winner = None
        self.current_player_idx = 0
        self.next_rule_number = 301
        self.waiting_for_human = False  # Flag to prevent infinite loops
        self.processing_proposal_result = False  # Prevent double processing

        # Initialize LLM provider (Ollama or OpenRouter)
        if provider is None:
            # Default to existing OllamaManager for backward compatibility
            self.ollama = OllamaManager()
            self.llm_provider = None  # Legacy mode
        else:
            self.llm_provider = provider
            self.ollama = provider  # For compatibility with existing code

        # Track provider type for analytics
        if isinstance(provider, OpenRouterClient):
            self.provider_type = "openrouter"
        elif isinstance(provider, OllamaClient):
            self.provider_type = "ollama_new"
        else:
            self.provider_type = "ollama_legacy"

        # Initialize proto character system
        self.character_manager = CharacterManager()

        # Initialize performance tracking
        self.performance_manager = ModelPerformanceManager()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.deliberation_manager = DeliberationManager(self)

        # Initialize new systems with port-specific session directory
        if session_dir is None:
            session_dir = "game_sessions"  # Default for backward compatibility
        self.session_manager = GameSessionManager(sessions_dir=session_dir)
        self.state_tracker = InternalStateTracker()
        self.idle_processor = IdleTurnProcessor(self.ollama, self.deliberation_manager, self)
        self.game_logger = GameLogger(self.session_manager)
        self.input_sanitizer = InputSanitizer()  # Security hardening for human input
        self.context_manager = ContextWindowManager()  # Context window optimization and tracking
        self.context_manager.reset_for_new_game()  # Reset context for fresh game start
        self.rag_system = BasicRAGSystem()  # Retrieval-Augmented Generation for game history
        self.failure_recovery = FailureRecoverySystem()  # Automated failure detection and recovery
        
        # Initialize Game Master system
        self.game_master = GameMaster(self.llm_provider or self.ollama)
        self.context_restoration_manager = ContextRestorationManager(self.game_master, self.llm_provider or self.ollama)
        self.dynamic_rule_engine = DynamicRuleEngine()
        self.gm_enabled = True  # Can be toggled for testing
        
        # Initialize Advanced Game Master API
        self.advanced_gm = create_game_master_engine(self)
        self.gm_api_enabled = True  # Toggle for advanced GM features
        
        # Initialize Hybrid Model Selector (if using OpenRouter with hybrid mode)
        self.hybrid_selector = None
        if isinstance(self.llm_provider, OpenRouterClient) and self.llm_provider.use_hybrid:
            self.hybrid_selector = HybridModelSelector(self.llm_provider)
        
        # Add threading lock to prevent race conditions
        self.turn_lock = threading.Lock()
        self.turn_in_progress = False

        # Create players and assign models/ports
        self.players = self._create_players()
        if self.llm_provider:
            self.llm_provider.assign_ports(self.players)
        else:
            self.ollama.assign_ports(self.players)

        # Initialize player internal states
        for player in self.players:
            self.state_tracker.initialize_player_state(player)

        # Start game session
        self.session_id = self.session_manager.start_new_session(self.players)

        # Log game start
        self.game_logger.log_game_start(self.players)

        self.rules = self._initialize_rules()
        self.proposals = []
        self.events = []
        self.current_proposal = None

        # Enhanced memory to prevent repetition
        self.global_proposed_rules = []  # All rules ever proposed
        self.turn_history = []  # Complete turn-by-turn history
        self.vote_history = []  # All votes with outcomes
        self.agent_memory = {
            p.id: {
                "proposed_rules": [],
                "voting_history": [],
                "internal_thoughts": [],
                "successful_rules": [],
                "failed_rules": [],
            }
            for p in self.players
        }

        # Rule execution tracking
        self.active_effects = []  # Currently active rule effects
        self.dice_value = 0  # Last dice roll
        self.current_voting_threshold = 100  # Start with unanimous

    @classmethod
    def resume_last_game(cls, provider: Union[OllamaClient, OllamaManager, OpenRouterClient] = None, session_dir: str = None):
        """Class method to resume the most recent in-progress game"""
        # Create a temporary session manager to find in-progress sessions
        if session_dir is None:
            session_dir = "game_sessions"
        temp_session_manager = GameSessionManager(sessions_dir=session_dir)
        latest_session_id = temp_session_manager.find_latest_in_progress_session()
        
        if not latest_session_id:
            print("❌ No in-progress sessions found to resume")
            return None
        
        print(f"🔄 Found in-progress session: {latest_session_id}")
        
        # Load session data
        session_data = temp_session_manager.load_session_data(latest_session_id)
        if not session_data or not session_data.get("metadata"):
            print("❌ Could not load session data")
            return None
        
        metadata = session_data["metadata"]
        players_data = metadata.get("players", [])
        
        print(f"📋 Session details:")
        print(f"   Started: {metadata.get('start_time', 'Unknown')}")
        print(f"   Players: {len(players_data)}")
        for p in players_data:
            print(f"     - {p.get('name', 'Unknown')} ({p.get('model', 'Unknown model')})")
        print(f"   Total turns so far: {metadata.get('total_turns', 0)}")
        print(f"   Rules created: {metadata.get('rules_created', 0)}")
        
        # Create new game instance with resume logic
        game = cls(
            num_players=len(players_data), 
            provider=provider, 
            include_human=any(p.get('model') == 'human' for p in players_data),
            session_dir=session_dir
        )
        
        # Resume the session in the session manager
        if not game.session_manager.resume_session(latest_session_id):
            print("❌ Failed to resume session")
            return None
        
        # Restore game state from session data
        game._restore_game_state(session_data)
        
        print("✅ Game resumed successfully!")
        return game

    def _restore_game_state(self, session_data: Dict):
        """Restore game state from saved session data"""
        metadata = session_data.get("metadata", {})
        turn_logs = session_data.get("turn_logs", [])
        player_states = session_data.get("player_states", {})
        
        # Restore basic game state
        self.turn_number = metadata.get("total_turns", 0) + 1  # Next turn to play
        self.next_rule_number = 301 + metadata.get("rules_created", 0)
        
        # Restore player points from final_scores or calculate from turn logs
        final_scores = metadata.get("final_scores", {})
        if final_scores:
            for player in self.players:
                if str(player.id) in final_scores:
                    player.points = final_scores[str(player.id)]
        
        # Restore player internal states
        for player_id_str, state_data in player_states.items():
            player_id = int(player_id_str)
            if player_id in self.state_tracker.player_states:
                # Restore scratchpad and other internal state
                player_state = self.state_tracker.player_states[player_id]
                player_state.scratchpad = state_data.get("scratchpad", [])
                player_state.strategic_focus = state_data.get("strategic_focus", "")
                player_state.victory_path = state_data.get("victory_path", "")
                player_state.threat_assessments = state_data.get("threat_assessments", {})
                player_state.learning_observations = state_data.get("learning_observations", [])
                player_state.planned_proposals = state_data.get("planned_proposals", [])
                player_state.last_updated = state_data.get("last_updated", "")
        
        # Calculate current player index based on turn number
        if self.turn_number > 1:
            self.current_player_idx = (self.turn_number - 1) % len(self.players)
        
        # Restore rules from turn logs if available
        self._restore_rules_from_logs(turn_logs)
        
        print(f"🔄 Restored game state:")
        print(f"   Next turn: {self.turn_number}")
        print(f"   Current player: {self.players[self.current_player_idx].name}")
        print(f"   Next rule number: {self.next_rule_number}")
        print(f"   Player scores:")
        for player in self.players:
            print(f"     {player.name}: {player.points} points")

    def _restore_rules_from_logs(self, turn_logs: List[Dict]):
        """Restore rules from turn logs to reconstruct game state"""
        # This is a simplified restoration - in a full implementation,
        # you might want to save the actual rules in the session data
        rules_passed = 0
        for turn_log in turn_logs:
            for event in turn_log.get("events", []):
                if event.get("type") == "rule_execution" and event.get("trigger") == "rule_adopted":
                    rules_passed += 1
        
        # Add placeholder rules for now - in practice, you'd want to save actual rule text
        for i in range(rules_passed):
            rule_id = 301 + i
            self.rules.append({
                "id": rule_id,
                "text": f"Rule {rule_id} (restored from session)",
                "type": "mutable",
                "active": True
            })

    def _create_players(self):
        players = []

        # Create human player first if requested
        if self.include_human:
            human_player = Player(
                id=1, name="Human", role="Human Player - strategic decision maker", model="human", is_human=True
            )
            players.append(human_player)

        # Get proto characters for AI players
        ai_player_count = self.num_players - (1 if self.include_human else 0)
        selected_characters = self.character_manager.assign_characters_to_players(ai_player_count)
        
        # Create AI players using proto characters
        for i, character in enumerate(selected_characters):
            player_id = len(players) + 1
            player = Player(
                id=player_id,
                name=character.name,  # Use character name instead of "Player X"
                role=character.strategy,  # Use character strategy as role
                model=character.model,  # Use character's locked model
                is_human=False,
                character=character  # Link the character
            )
            
            # Assign model metrics for tracking
            player.assigned_model_metrics = self.performance_manager.get_or_create_metrics(character.model)
            players.append(player)

        # Log character assignments
        character_assignments = {}
        for p in players:
            if p.character:
                win_rate = p.character.get_win_rate() * 100
                character_assignments[p.name] = f"{p.model} (Win rate: {win_rate:.1f}%, Games: {p.character.games_played})"
            else:
                character_assignments[p.name] = p.model
                
        print(f" Proto Character assignments for this game:")
        for name, info in character_assignments.items():
            print(f"  {name}: {info}")

        return players

    def unified_generate(self, player, prompt, **kwargs):
        """Unified text generation method that works with both Ollama and OpenRouter"""
        
        # Handle pending context restoration from previous degradation detection
        if hasattr(player, 'pending_context_restoration'):
            restoration_package = player.pending_context_restoration
            delattr(player, 'pending_context_restoration')  # Remove after use
            
            # Prepend restoration package to current prompt for immediate context restoration
            self.add_event(f" 🔧 Applying pending GM restoration package to {player.name}")
            prompt = f"{restoration_package}\n\n---\n\n{prompt}"
        
        # Get optimized context using advanced rolling window system
        original_prompt = prompt
        game_context = {"game": self}
        optimized_prompt = self.context_manager.get_optimized_context_for_model(player.model, prompt, game_context)

        # Track if optimization occurred
        if len(optimized_prompt) != len(original_prompt):
            reduction_percent = int((1 - len(optimized_prompt) / len(original_prompt)) * 100) if len(original_prompt) > 0 else 0
            self.add_event(
                f" 🎯 Advanced context optimization for {player.name}: {len(original_prompt)}→{len(optimized_prompt)} chars ({reduction_percent}% reduction)"
            )

        response = ""
        if self.llm_provider:
            # Using new provider system (Ollama or OpenRouter)
            if isinstance(self.llm_provider, OpenRouterClient):
                # Determine which model to use (hybrid selector if available)
                model_to_use = player.model  # Default to assigned model
                
                if self.hybrid_selector:
                    # Use hybrid model selection based on task complexity
                    task_type = kwargs.get('task_type', 'general')
                    task_context = kwargs.get('task_context', {})
                    model_to_use = self.hybrid_selector.get_model_for_player_task(
                        player, task_type, self, task_context
                    )
                    
                    # Log model selection for transparency
                    if model_to_use != player.model:
                        model_name = model_to_use.split('/')[-1]
                        self.add_event(f" 🔄 {player.name} escalated to {model_name} for {task_type}")
                
                # OpenRouter API call with selected model
                response = self.llm_provider.generate(model_to_use, optimized_prompt, **kwargs)
            else:
                # OllamaClient call
                response = self.llm_provider.generate(player.model, optimized_prompt, player.port, **kwargs)
        else:
            # Legacy OllamaManager call
            response = self.ollama.generate(player.model, optimized_prompt, player.port, **kwargs)

        # Track usage for context window management
        self.context_manager.track_usage(player.model, optimized_prompt, response)

        # Log context usage if approaching limits
        context_status = self.context_manager.get_context_usage(player.model)
        if context_status["usage_percentage"] > 75:
            self.add_event(
                f" Context usage for {player.model}: {context_status['usage_percentage']}% ({context_status['status']})"
            )
        
        # Track context degradation and apply interventions if needed
        degradation_status = self.context_manager.track_degradation(
            player.model, player.id, context_status["usage_percentage"]
        )
        
        # Check for intervention needs
        if degradation_status["needs_intervention"]:
            intervention_applied = False
            
            if degradation_status["intervention_urgency"] == "critical":
                self.add_event(f" 🚨 CRITICAL context degradation for {player.name} - applying emergency intervention")
                
                # Apply critical intervention: GM restoration package + context reset
                try:
                    game_context = {"game": self}
                    restoration_package = self.context_restoration_manager.apply_restoration(
                        player, game_context, intervention_urgency="critical"
                    )
                    
                    # Force context reset for this player's model
                    self.context_manager.reset_usage([player.model])
                    
                    # Replace current response with restoration package for immediate effect
                    response = restoration_package
                    intervention_applied = True
                    
                    self.add_event(f" ✅ Critical intervention applied: GM restoration package delivered to {player.name}")
                    
                except Exception as e:
                    self.add_event(f" ❌ Critical intervention failed for {player.name}: {e}")
                    
            elif degradation_status["intervention_urgency"] == "moderate":
                self.add_event(f" ⚠️ Context degradation detected for {player.name} - applying moderate intervention")
                
                # Apply moderate intervention: GM restoration package only
                try:
                    game_context = {"game": self}
                    restoration_package = self.context_restoration_manager.apply_restoration(
                        player, game_context, intervention_urgency="moderate"
                    )
                    
                    # Add restoration package as context for next turn (don't replace current response)
                    if hasattr(player, 'pending_context_restoration'):
                        player.pending_context_restoration = restoration_package
                    else:
                        # Store in player object for next generation
                        setattr(player, 'pending_context_restoration', restoration_package)
                    
                    intervention_applied = True
                    self.add_event(f" ✅ Moderate intervention queued: GM restoration package for {player.name}")
                    
                except Exception as e:
                    self.add_event(f" ❌ Moderate intervention failed for {player.name}: {e}")
            
            # Mark intervention as applied if successful
            if intervention_applied:
                self.context_manager.mark_intervention_applied(player.id)
        
        # Log degradation statistics occasionally 
        elif degradation_status["consecutive_high_usage"] > 0:
            self.add_event(
                f" 📊 Context tracking for {player.name}: {degradation_status['consecutive_high_usage']} consecutive high-usage events"
            )

        return response

    def _initialize_rules(self):
        """Initialize with parsed rules that have effects"""
        rules = {"immutable": [], "mutable": []}

        # Immutable rules (text only, no effects)
        immutable_texts = [
            "101: All players must always abide by all the rules then in effect, in the form in which they are then in effect.",
            "102: Initially, rules in the 100's are immutable and rules in the 200's are mutable.",
            "103: A rule change is: (a) enactment/repeal/amendment of a mutable rule; (b) transmutation of immutablemutable.",
            "104: All rule changes proposed in the proper way shall be voted on.",
            "105: Every player is an eligible voter.",
            "106: Any proposed rule change must be written down before it is voted on.",
            "107: No rule change may take effect earlier than the moment of the completion of the vote that adopted it.",
            "108: Each proposed rule change shall be given a rank-order number starting with 301.",
            "109: Rule changes that transmute immutable rules into mutable rules require unanimous vote.",
            "110: Mutable rules that are inconsistent with immutable rules are wholly void.",
            "111: If a rule change is unclear, other players may suggest amendments.",
            "112: The state of affairs that constitutes winning may not be changed from achieving N points.",
            "113: A player always has the option to forfeit the game.",
            "114: There must always be at least one mutable rule.",
            "115: Rule changes that affect rules needed to allow or apply rule changes are permissible.",
            "116: Whatever is not explicitly prohibited by a rule is permitted, except changing rules.",
            "117: If any player reaches zero points before a rule proposal passes, all players lose the game immediately.",
        ]

        for text in immutable_texts:
            match = re.match(r"(\d+):\s*(.+)", text)
            if match:
                rules["immutable"].append(
                    ParsedRule(id=int(match.group(1)), text=match.group(2), mutable=False, effects=[])
                )

        # Mutable rules with effects
        mutable_with_effects = [
            ("201: Players shall alternate in clockwise order, taking one whole turn apiece.", []),
            ("202: One turn consists of proposing one rule change and having it voted on.", []),
            (
                "203: A rule change is adopted if and only if the vote is unanimous among eligible voters.",
                [
                    RuleEffect(
                        trigger="rule_change",
                        action="change_voting",
                        target="system",
                        value=100,
                        description="Unanimous voting required",
                    )
                ],
            ),
            (
                "204: When rule changes can be adopted without unanimity, players who vote against winning proposals receive 10 points.",
                [
                    RuleEffect(
                        trigger="vote_pass",
                        condition="vote == nay",
                        action="add_points",
                        target="nay_voters",
                        value=10,
                        description="Reward dissenting voters",
                    )
                ],
            ),
            ("205: An adopted rule change takes full effect at the moment of completion of the vote.", []),
            (
                "206: When a proposed rule change is defeated, the player who proposed it loses 10 points.",
                [
                    RuleEffect(
                        trigger="vote_fail",
                        action="subtract_points",
                        target="proposer",
                        value=10,
                        description="Penalty for failed proposals",
                    )
                ],
            ),
            ("207: Each player has exactly one vote.", []),
            ("208: The winner is the first person to achieve 100 (positive) points.", []),
            ("209: At no time may there be more than 25 mutable rules.", []),
            ("210: Players may not conspire or consult on making future rule changes unless they are teammates.", []),
        ]

        for text, effects in mutable_with_effects:
            match = re.match(r"(\d+):\s*(.+)", text)
            if match:
                rules["mutable"].append(
                    ParsedRule(id=int(match.group(1)), text=match.group(2), mutable=True, effects=effects)
                )

        return rules

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def execute_transmutation(self, proposal) -> bool:
        """Execute transmutation of a rule between mutable and immutable"""
        target_rule_id = proposal.transmute_target_rule
        
        # Find the rule in mutable rules
        target_rule = None
        source_list = None
        target_list = None
        
        for rule in self.rules["mutable"]:
            if rule.id == target_rule_id:
                target_rule = rule
                source_list = self.rules["mutable"]
                target_list = self.rules["immutable"]
                break
        
        # If not in mutable, check immutable
        if not target_rule:
            for rule in self.rules["immutable"]:
                if rule.id == target_rule_id:
                    target_rule = rule
                    source_list = self.rules["immutable"]
                    target_list = self.rules["mutable"]
                    break
        
        if not target_rule:
            self.add_event(f" Rule {target_rule_id} not found for transmutation")
            return False
        
        # Perform the transmutation
        source_list.remove(target_rule)
        target_rule.mutable = not target_rule.mutable  # Flip mutability
        target_list.append(target_rule)
        
        # Log the transmutation
        status_change = "mutable  immutable" if target_rule.mutable == False else "immutable  mutable"
        self.add_event(f" TRANSMUTATION: Rule {target_rule_id} changed from {status_change}")
        self.add_event(f" Rule {target_rule_id}: {target_rule.text}")
        
        return True

    def execute_amendment(self, proposal) -> bool:
        """Execute amendment of a mutable rule"""
        target_rule_id = proposal.amend_target_rule
        
        # Find the rule in mutable rules (can only amend mutable rules)
        target_rule = None
        for rule in self.rules["mutable"]:
            if rule.id == target_rule_id:
                target_rule = rule
                break
        
        if not target_rule:
            # Check if rule exists in immutable rules
            for rule in self.rules["immutable"]:
                if rule.id == target_rule_id:
                    self.add_event(f" Rule {target_rule_id} is IMMUTABLE and cannot be amended")
                    self.add_event(f" Use transmutation to make it mutable first")
                    return False
            
            # Rule doesn't exist at all
            self.add_event(f" Rule {target_rule_id} not found for amendment")
            return False
        
        # Store original text for history
        original_text = target_rule.text
        
        # Perform the amendment
        target_rule.text = proposal.rule_text
        
        # Log the amendment with a clear before/after
        self.add_event(f" AMENDMENT: Rule {target_rule_id} modified")
        self.add_event(f" OLD: {original_text}")
        self.add_event(f" NEW: {target_rule.text}")
        
        # Update rule metadata
        if hasattr(target_rule, 'author') and hasattr(target_rule, 'turn_added'):
            # Keep original author but note the amendment
            target_rule.turn_added = self.turn_number  # Update to amendment turn
            self.add_event(f" Rule {target_rule_id} last modified turn {self.turn_number}")
        
        # Reparse effects for the amended rule
        if hasattr(proposal, 'parsed_effects'):
            target_rule.effects = proposal.parsed_effects
        
        # Update RAG system with amended rule
        if hasattr(self, 'rag_system'):
            self.rag_system.index_rule(
                target_rule.id, target_rule.text,
                {"type": "mutable", "amended_by": proposal.player_id, "turn_amended": self.turn_number, "original_text": original_text}
            )
        
        return True

    def execute_rule_effects(self, trigger: str, **kwargs):
        """Execute all rule effects for a given trigger"""
        executed = []

        # Create serializable context for logging
        serializable_context = {}
        for key, value in kwargs.items():
            if hasattr(value, "id") and hasattr(value, "name"):  # Player object
                serializable_context[key] = {"id": value.id, "name": value.name, "points": getattr(value, "points", 0)}
            elif hasattr(value, "id") and hasattr(value, "text"):  # Rule object
                serializable_context[key] = {"id": value.id, "text": value.text}
            elif isinstance(value, (str, int, float, bool, list, dict)):
                serializable_context[key] = value
            else:
                serializable_context[key] = str(value)
        
        # GM System has full control - executes first and can override/block regular rules
        gm_executed_effects = []
        if self.gm_enabled and self.dynamic_rule_engine:
            matching_rules = self.dynamic_rule_engine.get_rules_for_trigger(trigger)
            for rule_id in matching_rules:
                # Prepare context for GM rules
                gm_context = {
                    'trigger': trigger,
                    'turn_number': self.turn_number,
                    'current_player_id': self.get_current_player().id if hasattr(self, 'get_current_player') else None,
                    'all_players': [{'id': p.id, 'name': p.name, 'points': p.points} for p in self.players]
                }
                
                # Add specific context based on trigger
                if trigger == "proposal_made" and "proposer" in kwargs:
                    gm_context['proposer_id'] = kwargs['proposer'].id
                    gm_context['proposal_text'] = kwargs.get('proposal', {}).rule_text if 'proposal' in kwargs else ""
                elif trigger == "vote_pass" and "proposer" in kwargs:
                    gm_context['proposer_id'] = kwargs['proposer'].id
                elif trigger == "vote_fail" and "proposer" in kwargs:
                    gm_context['proposer_id'] = kwargs['proposer'].id
                    
                # Execute the GM rule
                result = self.dynamic_rule_engine.execute_rule(rule_id, serializable_context, gm_context)
                
                # Apply effects returned by GM rule
                if 'effects' in result:
                    for effect in result['effects']:
                        if effect.get('type') == 'add_points':
                            player = next((p for p in self.players if p.id == effect['target']), None)
                            if player:
                                player.points += effect['value']
                                self.add_event(f" GM Rule {rule_id}: {player.name} gains {effect['value']} points  {player.points}")
                                executed.append(f"GM Rule {rule_id}: Added {effect['value']} points")
                                gm_executed_effects.append(('add_points', effect['target'], effect['value']))
                        elif effect.get('type') == 'subtract_points':
                            player = next((p for p in self.players if p.id == effect['target']), None)
                            if player:
                                player.points -= effect['value']
                                self.add_event(f" GM Rule {rule_id}: {player.name} loses {effect['value']} points  {player.points}")
                                executed.append(f"GM Rule {rule_id}: Subtracted {effect['value']} points")
                                gm_executed_effects.append(('subtract_points', effect['target'], effect['value']))

        # Log rule execution start
        self.game_logger.log_rule_execution(trigger, serializable_context)

        # Collect all active effects from mutable rules
        all_effects = []
        for rule in self.rules["mutable"]:
            for effect in rule.effects:
                if effect.trigger == trigger:
                    all_effects.append((rule, effect))

        # Execute each matching effect
        for rule, effect in all_effects:
            # Check conditions
            if effect.condition:
                if not self.evaluate_condition(effect.condition, **kwargs):
                    continue

            # Execute action
            if effect.action == "add_points":
                targets = self.get_effect_targets(effect.target, **kwargs)
                for player in targets:
                    # Check if GM already handled this effect
                    gm_already_handled = any(
                        gm_effect[0] == 'add_points' and 
                        gm_effect[1] == player.id and 
                        gm_effect[2] == effect.value
                        for gm_effect in gm_executed_effects
                    )
                    if gm_already_handled:
                        continue
                        
                    player.points += effect.value
                    self.add_event(f" Rule {rule.id}: {player.name} gains {effect.value} points  {player.points}")
                    executed.append(f"{effect.description} for {len(targets)} player(s)")

            elif effect.action == "subtract_points":
                targets = self.get_effect_targets(effect.target, **kwargs)
                for player in targets:
                    # Check if GM already handled this effect
                    gm_already_handled = any(
                        gm_effect[0] == 'subtract_points' and 
                        gm_effect[1] == player.id and 
                        gm_effect[2] == effect.value
                        for gm_effect in gm_executed_effects
                    )
                    if gm_already_handled:
                        continue
                        
                    player.points -= effect.value
                    self.add_event(f" Rule {rule.id}: {player.name} loses {effect.value} points  {player.points}")
                    executed.append(f"{effect.description} for {len(targets)} player(s)")
                    # Check for zero points rule violation
                    if player.points <= 0:
                        self.check_zero_points_rule()

            elif effect.action == "steal_points":
                if "stealer" in kwargs and "victim" in kwargs:
                    stealer = kwargs["stealer"]
                    victim = kwargs["victim"]
                    stolen = min(effect.value, victim.points)
                    victim.points -= stolen
                    stealer.points += stolen
                    self.add_event(f" Rule {rule.id}: {stealer.name} steals {stolen} points from {victim.name}")
                    executed.append(f"Stole {stolen} points")
                    # Check for zero points rule violation
                    if victim.points <= 0:
                        self.check_zero_points_rule()

            elif effect.action == "change_voting":
                self.current_voting_threshold = effect.value
                self.add_event(f" Rule {rule.id}: Voting threshold changed to {effect.value}%")
                executed.append(effect.description)

        return executed

    def get_rule_by_id(self, rule_id: int) -> Optional[ParsedRule]:
        """Get a rule by its ID from immutable or mutable rules"""
        # Check immutable rules first
        for rule in self.rules["immutable"]:
            if rule.id == rule_id:
                return rule
        # Check mutable rules
        for rule in self.rules["mutable"]:
            if rule.id == rule_id:
                return rule
        return None

    def disable_rule(self, rule_id: int) -> bool:
        """Disable a rule by setting its active flag to False"""
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.active = False
            self.add_event(f" Rule {rule_id} has been disabled")
            return True
        return False

    def enable_rule(self, rule_id: int) -> bool:
        """Enable a rule by setting its active flag to True"""
        rule = self.get_rule_by_id(rule_id)
        if rule:
            rule.active = True
            self.add_event(f" Rule {rule_id} has been enabled")
            return True
        return False

    def check_zero_points_rule(self):
        """Check if any player reached zero points, ending game for all (Rule 117)"""
        # First check if Rule 117 is still active (could be transmuted/disabled)
        rule_117 = self.get_rule_by_id(117)
        if not rule_117 or not rule_117.active:
            return False  # Rule 117 has been disabled by transmutation
        
        # Rule 117: "If any player reaches zero points BEFORE a rule proposal passes, all players lose"
        # Only applies before first rule passes
        if self.next_rule_number > 301:
            return False  # Rules have already passed, Rule 117 no longer applies
        
        zero_point_players = [p for p in self.players if p.points <= 0]
        if zero_point_players:
            player_names = ", ".join([p.name for p in zero_point_players])
            self.add_event(f" GAME OVER: {player_names} reached 0 points! Rule 117 triggered - ALL PLAYERS LOSE!")
            self.add_event(f" Rule 117 applies because no rule proposals have passed yet (still at rule #{self.next_rule_number})")
            self.game_over = True
            self.game_over_reason = f"Rule 117 violation: {player_names} reached 0 points before any rule passed"
            # Log game end
            self.game_logger.log_game_end(None, self.turn_number, {p.id: p.points for p in self.players})
            return True
        return False

    def evaluate_condition(self, condition: str, **kwargs):
        """Evaluate a rule condition"""
        try:
            # Simple condition evaluation
            if "points <" in condition:
                match = re.search(r"points < (\d+)", condition)
                if match and "player" in kwargs:
                    return kwargs["player"].points < int(match.group(1))
            elif "points >" in condition:
                match = re.search(r"points > (\d+)", condition)
                if match and "player" in kwargs:
                    return kwargs["player"].points > int(match.group(1))
            elif "dice ==" in condition:
                match = re.search(r"dice == (\d+)", condition)
                if match:
                    return self.dice_value == int(match.group(1))
            elif "vote == nay" in condition:
                return kwargs.get("vote") is False
        except ValueError:
            pass
        return True  # Default to true if we can't evaluate

    def get_effect_targets(self, target_type: str, **kwargs):
        """Get players affected by a rule effect"""
        if target_type == "all_players":
            return self.players
        elif target_type == "current_player":
            return [self.get_current_player()]
        elif target_type == "proposer" and "proposer" in kwargs:
            return [kwargs["proposer"]]
        elif target_type == "nay_voters" and "votes" in kwargs:
            return [p for p in self.players if kwargs["votes"].get(p.id) is False]
        elif target_type == "voters" and "votes" in kwargs:
            return [p for p in self.players if p.id in kwargs["votes"]]
        else:
            return []

    def analyze_rule_gaps(self):
        """Analyze current rules to find gaps and opportunities"""
        analysis = []

        # Check what mechanics exist
        has_dice = any("dice" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_steal = any("steal" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_trade = any("trade" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_alliance = any(
            "alliance" in str(rule.text).lower() or "team" in str(rule.text).lower() for rule in self.rules["mutable"]
        )

        if not has_dice:
            analysis.append("No dice mechanics exist - opportunity for randomness")
        if not has_steal:
            analysis.append("No stealing mechanics - opportunity for player interaction")
        if not has_trade:
            analysis.append("No trading systems - opportunity for negotiation")
        if not has_alliance:
            analysis.append("No alliance mechanics - opportunity for cooperation")

        # Check point ranges
        if self.turn_number > 5:
            analysis.append(f"Turn {self.turn_number} - consider acceleration mechanics")

        return "\n".join(analysis) if analysis else "All basic mechanics covered - be creative!"

    def get_system_prompt_for_player(self, player):
        """Get comprehensive system prompt for a player"""
        return f"""You are {player.name}, a {player.role} in a game of Nomic.

GAME OVERVIEW:
Nomic is a self-modifying rule game where players propose changes to the rules themselves. The goal is to reach 100 points while strategically modifying the game to your advantage.

YOUR ROLE: {player.role}
As a {player.role.split(' - ')[0]}, you should focus on {player.role.split(' - ')[1] if ' - ' in player.role else 'strategic gameplay'}.

RULE STRUCTURE EXAMPLES:
- Point-based: "Players with fewer than 40 points gain 5 points each turn"
- Conditional: "If a player rolls a 6, they may steal 8 points from another player"
- Voting: "Players who vote Aye on successful proposals gain 3 points"
- Turn-based: "Every 3rd turn, all players gain 2 points"
- Meta-rules: "Players may spend 20 points to propose two rules in one turn"

IMPORTANT MECHANICS:
- Triggers: turn_start, vote_pass, vote_fail, dice_roll, proposal_made
- Conditions: point thresholds, dice values, turn numbers, player positions
- Effects: add/subtract points, change voting rules, special actions
- Targets: current_player, all_players, proposer, voters, specific conditions

YOUR STRATEGY:
- Propose rules that advance YOUR winning chances
- Consider what others will vote for (need {self.current_voting_threshold}% approval)
- Avoid repeating previously proposed rules
- Think about immediate vs long-term benefits
- Use your role personality to guide decisions

CRITICAL WARNINGS:
- Each rule must be UNIQUE and not previously proposed. Be creative!
- Rule 117: If ANY player reaches 0 points BEFORE a rule passes, ALL PLAYERS LOSE!
  (Note: Rule 117 only applies before first rule passes, and can be transmuted!)
- Overly destructive strategies can backfire and end the game for everyone"""

    def get_game_history_summary(self):
        """Get comprehensive game history"""
        if not self.vote_history:
            return "GAME HISTORY: No votes yet"

        history = "RECENT GAME HISTORY:\n"
        for vote in self.vote_history[-5:]:  # Last 5 votes
            history += f"Turn {vote['turn']}: {vote['proposer']} - '{vote['rule'][:50]}...'  {vote['outcome']} ({vote['ayes']}/{vote['total']})\n"

        return history

    def record_vote_outcome(self, proposal, ayes, total, passed):
        """Record vote outcome for history tracking"""
        proposer = next(p for p in self.players if p.id == proposal.player_id)
        self.vote_history.append(
            {
                "turn": self.turn_number,
                "proposer": proposer.name,
                "rule": proposal.rule_text,
                "ayes": ayes,
                "total": total,
                "outcome": "PASSED" if passed else "FAILED",
                "effects": (
                    [e.description for e in proposal.parsed_effects] if hasattr(proposal, "parsed_effects") else []
                ),
            }
        )

    def record_turn_state(self, player, proposal, votes):
        """Record complete turn state"""
        self.turn_history.append(
            {
                "turn": self.turn_number,
                "player": player.name,
                "proposal": proposal.rule_text,
                "votes": votes,
                "player_points": {p.name: p.points for p in self.players},
                "effects_detected": (
                    [e.description for e in proposal.parsed_effects] if hasattr(proposal, "parsed_effects") else []
                ),
            }
        )

    def get_game_context(self):
        """Get current game context for validation"""
        sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
        standings = "; ".join([f"{p.name}: {p.points}pts" for p in sorted_players])

        # Get key immutable rules with status
        key_immutable_rules = [r for r in self.rules['immutable'] if r.id in [117, 109, 110, 114]]
        immutable_status = chr(10).join([f"Rule {r.id}: {r.text} {'(ACTIVE)' if r.active else '(DISABLED)'}" for r in key_immutable_rules])
        
        context = f"""GAME STATE:
Turn: {self.turn_number}
Standings: {standings}

CURRENT MUTABLE RULES:
{chr(10).join([f"Rule {r.id}: {r.text}" for r in self.rules['mutable'][-8:]])}

KEY IMMUTABLE RULES (can be transmuted):
{immutable_status}

RECENT RULE ADDITIONS:
{chr(10).join([f"Turn {p.turn}: {p.rule_text}" for p in self.proposals[-3:]]) if self.proposals else "None yet"}
"""
        return context

    def generate_proposal_with_deliberation(self, player):
        """Generate proposal with flexible 1-3 turn deliberation for SOTA models"""

        self.add_event(f" {player.name} entering deliberation phase...")

        # Prepare comprehensive game state for deliberation
        sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
        my_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == player.id)

        game_state = {
            "turn": self.turn_number,
            "players": [{"id": p.id, "name": p.name, "points": p.points} for p in sorted_players],
            "mutable_rules": [{"id": r.id, "text": r.text, "active": r.active} for r in self.rules["mutable"]],
            "immutable_rules": [{"id": r.id, "text": r.text, "active": r.active} for r in self.rules["immutable"]],
            "rank": my_rank,
            "proposals_history": self.proposals,
            "vote_history": self.vote_history[-10:] if hasattr(self, "vote_history") else [],
        }

        # Calculate context pressure for adaptive deliberation
        context_pressure = 0.0
        if hasattr(self, 'context_manager') and self.context_manager:
            try:
                # Get context usage for the player's model
                player_model = getattr(player, 'model', 'unknown')
                context_status = self.context_manager.get_context_usage(player_model)
                context_pressure = context_status.get('usage_percentage', 0.0) / 100.0
                
                if context_pressure > 0.8:
                    self.add_event(f" ⚠️ {player.name} context pressure: {context_pressure:.1%}")
            except Exception as e:
                print(f"Error calculating context pressure for {player.name}: {e}")
                context_pressure = 0.0

        # Use character traits to determine deliberation approach
        if hasattr(player, 'character') and player.character and hasattr(player.character, 'traits'):
            # Use character-specific deliberation preferences with context pressure
            game_complexity = self._assess_game_complexity(game_state)
            chosen_turns = player.character.get_preferred_deliberation_turns(game_complexity, context_pressure)
            
            pressure_note = ""
            if context_pressure > 0.85:
                pressure_note = f" (reduced due to {context_pressure:.1%} context pressure)"
            self.add_event(f" {player.name} ({player.character.traits.decision_speed} decision maker) chooses {chosen_turns}-turn deliberation{pressure_note}")
        else:
            # Fallback to original choice method for characters without traits
            choice_prompt = self.deliberation_manager.generate_turn_choice_prompt(player, game_state)
            choice_response = self.unified_generate(player, choice_prompt, temperature=0.3, max_tokens=2000)
            chosen_turns = self._parse_deliberation_turns(choice_response)
            
            # Apply context pressure to fallback method too
            if context_pressure > 0.95:
                chosen_turns = 1
                self.add_event(f" {player.name} chooses {chosen_turns}-turn deliberation (emergency context reduction)")
            elif context_pressure > 0.85:
                chosen_turns = min(2, chosen_turns)
                self.add_event(f" {player.name} chooses {chosen_turns}-turn deliberation (context pressure applied)")
            else:
                self.add_event(f" {player.name} chooses {chosen_turns}-turn deliberation approach")
        
        # Log the deliberation choice
        choice_reasoning = choice_response[:200] if 'choice_response' in locals() else f"Character-based: {chosen_turns} turns via {player.character.traits.decision_speed} speed"
        self.game_logger.log_deliberation_choice(
            player.id, chosen_turns, choice_reasoning
        )

        # Store deliberation results for building final proposal
        deliberation_results = {}

        # Run flexible deliberation loop (1-3 turns)
        for deliberation_turn in range(1, chosen_turns + 1):
            self.add_event(f" {player.name} deliberation turn {deliberation_turn}/{chosen_turns}")

            # Generate flexible deliberation prompt
            prompt = self.deliberation_manager.generate_flexible_deliberation_prompt(
                player, game_state, deliberation_turn, chosen_turns, deliberation_results
            )

            # Get deliberation response with much higher token limit
            response = self.unified_generate(
                player, prompt, 
                temperature=0.7, max_tokens=2000,
                task_type="proposal_deliberation",
                task_context={
                    "deliberation_turn": deliberation_turn,
                    "total_turns": chosen_turns,
                    "involves_meta_rules": "rule" in prompt.lower() and ("proposal" in prompt.lower() or "voting" in prompt.lower()),
                    "late_game": self.turn_number >= 15
                }
            )

            # Log deliberation step
            self.game_logger.log_deliberation_step(player.id, deliberation_turn, prompt, response)

            # Store free-form deliberation results (no rigid parsing)
            deliberation_results[f"turn_{deliberation_turn}"] = response

            # Optional: Extract any insights the model wants to preserve
            if "INSIGHT:" in response:
                insight = response.split("INSIGHT:")[1].split("\n")[0].strip()
                self.add_event(f" {player.name} insight: {insight[:60]}...")
                self.game_logger.log_insight(player.id, insight)
            
            # Parse and save scratchpad notes from deliberation
            try:
                scratchpad_notes = self.deliberation_manager._parse_scratchpad_notes(response)
                if scratchpad_notes:
                    self.deliberation_manager._save_scratchpad_notes(player.id, scratchpad_notes)
                    self.add_event(f" {player.name} added {len(scratchpad_notes)} notes to scratchpad")
            except Exception as e:
                print(f"Error processing scratchpad notes for {player.name}: {e}")
                # Continue without breaking the game

            # Small delay between deliberation turns
            time.sleep(0.5)

        # Now generate final proposal using deliberation insights
        self.add_event(f" {player.name} deliberation complete, crafting proposal...")

        # Build final proposal prompt with all deliberation context and checkbox system
        final_prompt = self.deliberation_manager.generate_final_proposal_prompt_with_checkboxes(
            player, game_state, deliberation_results, chosen_turns
        )

        # Generate final proposal with high token limit
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.unified_generate(player, final_prompt, temperature=0.8, max_tokens=2000)

            # Parse the structured response
            parsed = self.parse_proposal_response(response)
            if not parsed:
                continue

            # Parse rule effects with enhanced checkbox system
            parsed_effects = self.parse_rule_effects_with_checkboxes(parsed["rule_text"])
            parsed["parsed_effects"] = parsed_effects

            # Check proposal category and enforce diversity
            proposed_category = self.deliberation_manager.categorize_proposal(
                parsed["rule_text"], parsed["explanation"]
            )

            # If category is overused, reject and try again
            if self.deliberation_manager.is_category_overused(proposed_category) and attempt < max_attempts - 1:
                self.add_event(f" {player.name} proposal rejected for overused category: {proposed_category}")
                continue

            # Validate the rule
            validation = self.ollama.validate_rule(parsed["rule_text"], self.get_game_context())

            if validation["valid"]:
                # Record category usage
                self.deliberation_manager.record_category_usage(proposed_category)

                # Enhanced behavioral analysis
                proposal_data = {
                    "rule_text": parsed["rule_text"],
                    "explanation": parsed["explanation"],
                    "internal_thought": parsed["internal_thought"],
                    "game_context": self.get_game_context(),
                    "game_state": game_state,
                    "all_previous_proposals": [p.rule_text for p in self.proposals],
                    "validation_failed": False,
                    "syntax_error": False,
                    "passed": False,
                    "deliberation_results": deliberation_results,
                    "proposal_category": proposed_category,
                }

                # Behavioral checks
                is_random = self.behavioral_analyzer.detect_random_behavior(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"]
                )
                if is_random:
                    self.add_event(f" Warning: {player.name} showing signs of random behavior")
                    proposal_data["random_behavior"] = True

                # Analysis scores
                coherence_score = self.performance_manager.analyze_proposal_coherence(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"]
                )
                memory_score = self.performance_manager.analyze_memory_usage(
                    parsed["internal_thought"], parsed["explanation"], self.get_game_context()
                )
                engagement_score = self.performance_manager.analyze_strategic_engagement(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"], game_state
                )

                self.add_event(
                    f" {player.name} final analysis: Coherence: {coherence_score:.0f}, Memory: {memory_score:.0f}, Engagement: {engagement_score:.0f}"
                )
                self.add_event(f" {player.name} proposal category: {proposed_category}")

                # Update memory
                memory = self.agent_memory[player.id]
                memory["proposed_rules"].append(parsed["rule_text"])
                memory["internal_thoughts"].append(parsed["internal_thought"])
                memory["proposed_rules"] = memory["proposed_rules"][-5:]
                memory["internal_thoughts"] = memory["internal_thoughts"][-5:]

                # Store comprehensive data
                parsed["_performance_data"] = proposal_data
                parsed["_deliberation_results"] = deliberation_results

                self.add_event(f" {player.name} proposes Rule {self.next_rule_number}: {parsed['rule_text']}")
                self.add_event(f" Explanation: {parsed['explanation']}")
                self.add_event(f" Internal thought: {parsed['internal_thought']}")

                return parsed

        # NO MORE STRATEGIC FALLBACKS! Force AI to be creative
        self.add_event(f" {player.name} failed to generate valid proposal - encouraging creative retry")
        self.game_logger.log_failed_proposal(player.id, "Failed to parse proposal format", 1)
        
        # Try ONE more time with enhanced temporal context prompt
        retry_prompt = f"""
PROPOSAL GENERATION RETRY - Your previous attempt failed to parse correctly.

{self.deliberation_manager.generate_nomic_context_header(game_state, player)}

LEARNING FROM FAILURE: Your last proposal attempt could not be parsed. This means you need to:
- Follow the EXACT format: INTERNAL_THOUGHT: / RULE_TEXT: / EXPLANATION:
- Use your scratchpad notes and temporal context
- Avoid repeating failed proposals from your history
- Be creative but clear

YOUR RECENT NOTES: Check your scratchpad for patterns of what worked/failed

SIMPLE SUCCESS TEMPLATE (adapt creatively):
INTERNAL_THOUGHT: [Your honest strategic thinking about the current game state]
RULE_TEXT: [Clear, enforceable rule under 30 words]
EXPLANATION: [Why others should vote for this]

Focus on the current game situation and be creative!
"""
        
        retry_response = self.unified_generate(player, retry_prompt, temperature=0.5, max_tokens=1000)
        retry_parsed = self.parse_proposal_response(retry_response)
        
        if retry_parsed:
            self.add_event(f" {player.name} successful retry with enhanced temporal context")
            return retry_parsed
        
        # If retry also fails, provide minimal fallback but log learning opportunity
        self.add_event(f" {player.name} complete generation failure - needs better temporal awareness")
        self.game_logger.log_failed_proposal(player.id, "Failed retry - complete generation failure", 2)
        
        # Add this failure to their learning observations for future scratchpad use
        if hasattr(self, 'state_tracker'):
            try:
                state = self.state_tracker.get_player_state(player.id)
                failure_note = f"Turn {self.turn_number}: Failed to generate valid proposal twice - need to follow format better"
                if hasattr(state, 'learning_observations'):
                    state.learning_observations.append(failure_note)
                    state.learning_observations = state.learning_observations[-10:]  # Keep recent
                    # State is automatically updated since we modified the object directly
            except:
                pass
        
        # Absolute minimal fallback
        return {
            "rule_text": "Players gain 1 point when they successfully follow the proposal format.",
            "explanation": "Encourages proper rule formatting for better game flow.",
            "internal_thought": "Generated after parsing failure - must improve format compliance next time.",
        }

    def _assess_game_complexity(self, game_state: Dict) -> str:
        """Assess the complexity of the current game state"""
        players = game_state.get('players', [])
        mutable_rules = game_state.get('mutable_rules', [])
        turn = game_state.get('turn', 1)
        
        # Calculate complexity factors
        point_spread = max(p['points'] for p in players) - min(p['points'] for p in players) if players else 0
        rules_count = len(mutable_rules)
        
        # Simple early game with little differentiation
        if turn <= 3 and point_spread <= 20 and rules_count <= 12:
            return "simple"
        # Complex late game with high stakes
        elif turn >= 10 or max(p['points'] for p in players if players) >= 80 or point_spread >= 40:
            return "complex"
        else:
            return "medium"

    def _parse_deliberation_turns(self, response: str) -> int:
        """Parse the number of deliberation turns chosen by the model"""
        response_lower = response.lower()

        # Look for explicit numbers
        if "1 turn" in response_lower or "one turn" in response_lower:
            return 1
        elif "3 turn" in response_lower or "three turn" in response_lower:
            return 3
        elif "2 turn" in response_lower or "two turn" in response_lower:
            return 2

        # Look for keywords indicating complexity
        if any(word in response_lower for word in ["quick", "simple", "immediate", "obvious"]):
            return 1
        elif any(word in response_lower for word in ["complex", "deep", "thorough", "comprehensive"]):
            return 3

        # Default to 2 turns if unclear
        return 2


    def parse_proposal_response(self, response: str):
        """Parse structured proposal response"""
        internal_match = re.search(r"INTERNAL_THOUGHT:\s*(.*?)(?:RULE_TEXT:|$)", response, re.DOTALL)
        rule_match = re.search(r"RULE_TEXT:\s*(.*?)(?:EXPLANATION:|$)", response, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:EFFECTS:|$)", response, re.DOTALL)

        # New: Parse checkbox effects
        effects_match = re.search(r"EFFECTS:\s*(.*?)(?:$)", response, re.DOTALL)

        if not rule_match:
            return None

        result = {
            "internal_thought": internal_match.group(1).strip() if internal_match else "No internal thought provided",
            "rule_text": rule_match.group(1).strip(),
            "explanation": explanation_match.group(1).strip() if explanation_match else "No explanation provided",
        }

        # Parse checkbox effects if provided
        if effects_match:
            checkbox_text = effects_match.group(1).strip()
            result["checkbox_effects"] = self.parse_checkbox_effects(checkbox_text)

        return result

    def parse_checkbox_effects(self, checkbox_text: str) -> dict:
        """Parse checkbox effect declarations from model output"""
        effects = {
            "points": {"add": 0, "subtract": 0, "steal": 0, "target": "none"},
            "voting": {"change_threshold": False, "unanimous": False, "majority": False},
            "turn_mechanics": {"skip": False, "extra": False, "order_change": False},
            "win_conditions": {"modify": False, "new_threshold": 0},
            "special": {"dice": False, "conditional": False, "compound": False},
        }

        lines = checkbox_text.lower().split("\n")
        for line in lines:
            # Points effects
            if "[x]" in line or "" in line or "yes" in line:
                if "add" in line and "point" in line:
                    match = re.search(r"(\d+)", line)
                    if match:
                        effects["points"]["add"] = int(match.group(1))
                        if "all" in line:
                            effects["points"]["target"] = "all_players"
                        elif "last" in line or "lowest" in line:
                            effects["points"]["target"] = "last_place"
                        else:
                            effects["points"]["target"] = "current_player"

                elif "subtract" in line or "lose" in line:
                    match = re.search(r"(\d+)", line)
                    if match:
                        effects["points"]["subtract"] = int(match.group(1))

                elif "steal" in line:
                    match = re.search(r"(\d+)", line)
                    if match:
                        effects["points"]["steal"] = int(match.group(1))

                # Voting effects
                elif "unanimous" in line:
                    effects["voting"]["unanimous"] = True
                elif "majority" in line:
                    effects["voting"]["majority"] = True
                elif "threshold" in line:
                    effects["voting"]["change_threshold"] = True

                # Turn mechanics
                elif "skip" in line and "turn" in line:
                    effects["turn_mechanics"]["skip"] = True
                elif "extra" in line and "turn" in line:
                    effects["turn_mechanics"]["extra"] = True

                # Special conditions
                elif "dice" in line:
                    effects["special"]["dice"] = True
                elif "condition" in line or "if" in line:
                    effects["special"]["conditional"] = True

        return effects

    def parse_rule_effects_with_checkboxes(self, rule_text: str) -> list:
        """Enhanced rule parsing using both traditional parsing and checkbox hints"""
        # Use traditional parser as baseline
        parser = RuleParser()
        traditional_effects = parser.parse_rule(rule_text)

        # Check for special cases that traditional parser might miss
        text_lower = rule_text.lower()

        # Enhanced detection for "last place" or "lowest scoring" players
        if any(phrase in text_lower for phrase in ["last place", "lowest", "fewest points", "trailing"]):
            # Look for point values
            point_match = re.search(r"(\d+)\s*(?:point|pt)", text_lower)
            if point_match:
                value = int(point_match.group(1))

                # Determine trigger
                trigger = "turn_start"
                if "each turn" in text_lower or "every turn" in text_lower:
                    trigger = "turn_start"
                elif "when" in text_lower or "if" in text_lower:
                    trigger = "conditional"

                # Add effect for last place bonus
                traditional_effects.append(
                    RuleEffect(
                        trigger=trigger,
                        condition="is_last_place",
                        action="add_points",
                        target="last_place_players",
                        value=value,
                        description=f"Player(s) in last place gain {value} points",
                    )
                )

        # Check for complex voting changes
        if "vote" in text_lower and any(word in text_lower for word in ["require", "need", "must"]):
            if "unanimous" in text_lower:
                traditional_effects.append(
                    RuleEffect(
                        trigger="rule_change",
                        action="change_voting",
                        target="system",
                        value=100,
                        description="Require unanimous consent for proposals",
                    )
                )

        # Check for turn order changes
        if "reverse" in text_lower and "turn" in text_lower:
            traditional_effects.append(
                RuleEffect(
                    trigger="immediate",
                    action="reverse_turn_order",
                    target="system",
                    value=1,
                    description="Reverse turn order",
                )
            )

        return traditional_effects

    def _parse_voting_turns(self, response: str) -> int:
        """Parse the number of voting turns chosen by the model (1 or 2)"""
        response_lower = response.lower()

        # Look for explicit numbers
        if "1 turn" in response_lower or "one turn" in response_lower:
            return 1
        elif "2 turn" in response_lower or "two turn" in response_lower:
            return 2

        # Look for keywords indicating quick vs thorough
        if any(word in response_lower for word in ["quick", "obvious", "clear", "simple"]):
            return 1
        elif any(word in response_lower for word in ["analyze", "consider", "evaluate", "complex"]):
            return 2

        # Default to 2 turns if unclear
        return 2

    def _generate_voting_choice_prompt(self, player, proposal, proposer, my_rank, proposer_rank) -> str:
        """Generate prompt for choosing voting deliberation approach"""
        return f"""VOTING DELIBERATION PLANNING
        
PROPOSAL TO EVALUATE:
Rule: "{proposal.rule_text}"
Proposed by: {proposer.name} (Rank #{proposer_rank})
Your Position: {player.name} (Rank #{my_rank})

Choose your voting analysis approach:

**1 TURN** - Quick Decision
- Use when the impact is obvious
- Clear benefit/harm to your position
- Simple rule effects

**2 TURNS** - Thorough Analysis  
- Standard approach for complex rules
- Need to evaluate multiple factors
- Strategic impact requires deeper thought

This rule complexity level and how it affects your victory chances should guide your choice.

RESPOND WITH:
- Your chosen number of turns (1 or 2)
- Brief reasoning for your approach"""

    def _get_simple_scratchpad_content(self, player) -> str:
        """Get simplified scratchpad content for voting prompts"""
        if not hasattr(self, 'state_tracker'):
            return " SCRATCHPAD: Not available during voting"
        
        try:
            state = self.state_tracker.get_player_state(player.id)
            if hasattr(state, 'scratchpad') and state.scratchpad:
                recent_notes = state.scratchpad[-3:]  # Last 3 notes only
                scratchpad_text = " YOUR RECENT NOTES:\n"
                for i, note in enumerate(recent_notes, 1):
                    scratchpad_text += f"   {i}. {note[:100]}...\n"
                return scratchpad_text
            else:
                return " SCRATCHPAD: No recent notes"
        except:
            return " SCRATCHPAD: Not accessible"
    
    def _get_simple_temporal_reminders(self) -> str:
        """Get simplified temporal reminders for voting"""
        return " TEMPORAL CONTEXT: Consider your strategic position and recent game events when voting"

    def _generate_voting_turn_prompt(
        self,
        player,
        proposal,
        proposer,
        my_rank,
        proposer_rank,
        turn_num,
        total_turns,
        previous_results,
        async_analysis=None,
        votes_so_far=None,
    ) -> str:
        """Generate enhanced voting turn prompts with temporal context and game awareness"""

        # Get player's scratchpad and temporal reminders (simplified for voting)
        scratchpad_content = self._get_simple_scratchpad_content(player)
        temporal_reminders = self._get_simple_temporal_reminders()

        # Prepare voting context
        voting_context = ""
        if votes_so_far:
            voting_context = f"\nCURRENT VOTES: {', '.join(votes_so_far)}\n"

        # Prepare async analysis section
        async_section = ""
        if async_analysis:
            async_section = f"""
YOUR EARLIER ANALYSIS:
{async_analysis}

Now with voting context and other players votes, finalize your decision:
"""

        if turn_num == 1 and total_turns == 1:
            # Single turn comprehensive analysis
            return f""" COMPREHENSIVE VOTING ANALYSIS{" (WITH ASYNC CONTEXT)" if async_analysis else ""}
{scratchpad_content}
{temporal_reminders}

PROPOSAL: "{proposal.rule_text}"
Proposed by: {proposer.name} (Rank #{proposer_rank}, {proposer.points} points)
Your Position: {player.name} (Rank #{my_rank}, {player.points} points){voting_context}
{async_section}
You have 1500+ tokens for {'final ' if async_analysis else ''}analysis. Consider all relevant factors:

- How this rule affects YOUR path to 100 points
- Whether this helps or hinders your competitive position
- Long-term vs short-term strategic implications
- Risk of auto-lose conditions (Rule 117: all players lose if anyone hits 0)
- Game balance and whether this creates healthy competition
{"- How other players' votes affect the outcome" if votes_so_far else ""}

Play to WIN, but remember that overly destructive rules can trigger auto-lose conditions. Balance competitive advantage with game stability.

REQUIRED ENDING FORMAT:
VOTE: AYE/NAY
REASONING: [Your strategic reasoning for this vote]"""

        elif turn_num == 1 and total_turns == 2:
            # First turn of two-turn analysis
            return f""" VOTING ANALYSIS TURN 1/2: IMPACT ASSESSMENT

PROPOSAL: "{proposal.rule_text}"
Proposed by: {proposer.name} (Rank #{proposer_rank}, {proposer.points} points)
Your Position: {player.name} (Rank #{my_rank}, {player.points} points)

Focus on understanding the strategic implications. You have 1500+ tokens for deep analysis.

Analyze how this rule affects:
- Your competitive position and victory chances
- Strategic opportunities and risks it creates
- Whether it could lead to auto-lose scenarios (Rule 117)
- Game balance and long-term strategic depth

No format required - think freely about the strategic implications."""

        else:  # turn_num == 2 and total_turns == 2
            # Second turn of two-turn analysis
            previous_analysis = previous_results.get("turn_1", "")
            return f""" VOTING ANALYSIS TURN 2/2: FINAL DECISION

PROPOSAL: "{proposal.rule_text}"

YOUR PREVIOUS ANALYSIS:
{previous_analysis}

Based on your analysis above, make your final voting decision.

REQUIRED ENDING FORMAT:
VOTE: AYE/NAY
REASONING: [Your strategic reasoning for this vote]"""

    def vote_on_proposal_with_deliberation(self, proposal):
        """Flexible voting with 1-2 turn deliberation for SOTA models"""
        votes = {}
        proposer = next(p for p in self.players if p.id == proposal.player_id)

        self.add_event(" Starting flexible voting phase...")

        for player in self.players:
            # CRITICAL: Proposer must always vote AYE for their own proposal
            if player.id == proposal.player_id:
                votes[player.id] = True
                self.add_event(f" {player.name}: Aye (own proposal)")
                continue

            # Handle human players differently
            if player.is_human:
                # For human players, skip the AI deliberation and wait for web input
                self.add_event(f" {player.name}, please vote on the proposal via the web interface.")
                # The vote will be set via the /human/submit-vote route
                # For now, we continue with other players and check back later
                continue

            # Check if this player has async deliberation stored
            has_async_analysis = hasattr(player, "async_deliberation") and proposal.id in player.async_deliberation

            if has_async_analysis:
                self.add_event(f" {player.name} using pre-computed analysis + current voting context...")
                async_analysis = player.async_deliberation[proposal.id]["initial_analysis"]
            else:
                self.add_event(f" {player.name} entering voting deliberation...")
                async_analysis = None

            # Get current standings for competitive context
            sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
            my_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == player.id)
            proposer_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == proposer.id)

            # Get votes cast so far for context
            votes_so_far = []
            for voted_player_id, vote_value in votes.items():
                voted_player = next(p for p in self.players if p.id == voted_player_id)
                votes_so_far.append(f"{voted_player.name}: {'AYE' if vote_value else 'NAY'}")

            # Calculate context pressure for voting deliberation
            context_pressure = 0.0
            if hasattr(self, 'context_manager') and self.context_manager:
                try:
                    player_model = getattr(player, 'model', 'unknown')
                    context_status = self.context_manager.get_context_usage(player_model)
                    context_pressure = context_status.get('usage_percentage', 0.0) / 100.0
                except Exception as e:
                    print(f"Error calculating voting context pressure for {player.name}: {e}")
                    context_pressure = 0.0

            # Let model choose voting deliberation approach (1 or 2 turns) with context pressure
            if hasattr(player, 'character') and player.character and hasattr(player.character, 'traits'):
                # Use character traits for voting turns with context pressure
                if context_pressure > 0.95:
                    chosen_turns = 1  # Emergency single-turn voting
                    pressure_note = " (emergency context reduction)"
                elif context_pressure > 0.85:
                    chosen_turns = 1  # High pressure single-turn voting
                    pressure_note = f" (context pressure: {context_pressure:.1%})"
                else:
                    # Normal character-based voting turns (1-2 range)
                    _, max_turns = player.character.traits.deliberation_range
                    chosen_turns = min(2, max(1, max_turns - 1))  # Voting is typically shorter
                    pressure_note = ""
                
                self.add_event(f" {player.name} chooses {chosen_turns}-turn voting analysis{pressure_note}")
            else:
                # Fallback to original choice method
                choice_prompt = self._generate_voting_choice_prompt(player, proposal, proposer, my_rank, proposer_rank)
                choice_response = self.unified_generate(player, choice_prompt, temperature=0.3, max_tokens=2000)
                chosen_turns = self._parse_voting_turns(choice_response)
                
                # Apply context pressure to fallback method
                if context_pressure > 0.95:
                    chosen_turns = 1
                    self.add_event(f" {player.name} chooses {chosen_turns}-turn voting analysis (emergency context reduction)")
                elif context_pressure > 0.85:
                    chosen_turns = min(1, chosen_turns)
                    self.add_event(f" {player.name} chooses {chosen_turns}-turn voting analysis (context pressure applied)")
                else:
                    self.add_event(f" {player.name} chooses {chosen_turns}-turn voting analysis")

            # Store deliberation results for building final vote
            voting_results = {}

            # Run flexible voting deliberation loop (1-2 turns)
            for voting_turn in range(1, chosen_turns + 1):
                self.add_event(f" {player.name} voting analysis turn {voting_turn}/{chosen_turns}")

                # Generate flexible voting turn prompt
                prompt = self._generate_voting_turn_prompt(
                    player,
                    proposal,
                    proposer,
                    my_rank,
                    proposer_rank,
                    voting_turn,
                    chosen_turns,
                    voting_results,
                    async_analysis,
                    votes_so_far,
                )

                # Get voting analysis with high token limit
                response = self.unified_generate(
                    player, prompt, 
                    temperature=0.6, max_tokens=2000,
                    task_type="voting_decision",
                    task_context={
                        "voting_turn": voting_turn,
                        "total_turns": chosen_turns,
                        "affects_player_directly": proposer.id == player.id,
                        "predicted_vote_margin": 2,  # Will enhance this with actual prediction
                        "proposal_type": "amendment" if "amend" in proposal.rule_text.lower() else "new_rule"
                    }
                )

                # Store free-form voting analysis (no rigid parsing)
                voting_results[f"turn_{voting_turn}"] = response

                # Optional: Extract any key insights
                if "INSIGHT:" in response:
                    insight = response.split("INSIGHT:")[1].split("\n")[0].strip()
                    self.add_event(f" {player.name} voting insight: {insight[:50]}...")

                # Small delay between voting turns
                time.sleep(0.3)

            # Final vote decision - simplified and more effective
            final_vote_prompt = f""" VOTING DECISION

PROPOSAL: "{proposal.rule_text}"
BY: {proposer.name} (Rank #{proposer_rank})
YOU: {player.name} (Rank #{my_rank})

QUICK DECISION:
1. Does this help YOU reach 100 points? 
2. Does this help others MORE than you?
3. Will this rule hurt your future chances?

VOTE STRATEGY:
- Vote AYE if you benefit or if it is neutral but helps underdogs
- Vote NAY if it primarily helps the proposer or hurts you
- Remember: You need others votes for YOUR proposals later

REQUIRED OUTPUT:
VOTE: [AYE or NAY]
REASONING: [One sentence why]

Keep it simple - overthinking leads to poor decisions."""

            final_response = self.unified_generate(player, final_vote_prompt, temperature=0.5, max_tokens=2000)

            # Parse vote decision - improved parsing with fallbacks
            vote = False  # Fallback only if parsing fails
            reasoning = "Failed to determine vote from response - defaulting to NAY for safety"

            # Try multiple parsing approaches
            vote_parsed = False
            
            # Primary parsing: Look for VOTE: format
            if "VOTE:" in final_response:
                vote_text = final_response.split("VOTE:")[1].split("REASONING:")[0].strip()
                if "AYE" in vote_text.upper():
                    vote = True
                    vote_parsed = True
                elif "NAY" in vote_text.upper():
                    vote = False
                    vote_parsed = True

                if "REASONING:" in final_response:
                    reasoning = final_response.split("REASONING:")[1].strip()
                    reasoning = reasoning[:300]  # Allow longer reasoning display
            
            # Fallback parsing: Look for standalone AYE/NAY anywhere in response
            if not vote_parsed:
                response_upper = final_response.upper()
                if "AYE" in response_upper and "NAY" not in response_upper:
                    vote = True
                    vote_parsed = True
                    reasoning = "Parsed AYE from response text"
                elif "NAY" in response_upper and "AYE" not in response_upper:
                    vote = False
                    vote_parsed = True
                    reasoning = "Parsed NAY from response text"
                elif "YES" in response_upper and "NO" not in response_upper:
                    vote = True
                    vote_parsed = True
                    reasoning = "Parsed YES from response text"
                elif "NO" in response_upper and "YES" not in response_upper:
                    vote = False
                    vote_parsed = True
                    reasoning = "Parsed NO from response text"
            
            # Log parsing issues for debugging
            if not vote_parsed:
                self.add_event(f" {player.name} vote parsing failed, response: {final_response[:100]}...")

            # Only set vote if player hasn't voted yet
            if player.id not in votes:
                votes[player.id] = vote
                # Trigger vote_cast effects (like Rule 301: gain points for voting)
                self.execute_rule_effects("vote_cast", voter=player, proposal=proposal)
            else:
                self.add_event(f" {player.name} attempted to vote twice - ignoring")

            # Track competitive voting patterns
            competitive_score = 0
            if not vote and proposer_rank < my_rank:  # Voted against someone ahead
                competitive_score += 2
            elif vote and my_rank == 1:  # Leader helping others (suspicious)
                competitive_score -= 1
            elif "negative" in reasoning.lower():
                competitive_score += 1

            # Update memory with competitive context and deliberation results
            self.agent_memory[player.id]["voting_history"].append(
                {
                    "rule": proposal.rule_text,
                    "vote": vote,
                    "proposer": proposer.name,
                    "turn": self.turn_number,
                    "my_rank": my_rank,
                    "proposer_rank": proposer_rank,
                    "competitive_score": competitive_score,
                    "reasoning": reasoning,
                    "deliberation_turns": chosen_turns,
                    "voting_analysis": voting_results,
                }
            )
            self.agent_memory[player.id]["voting_history"] = self.agent_memory[player.id]["voting_history"][-10:]

            # Log vote with competitive context
            vote_text = "Aye" if vote else "Nay"
            competitive_indicator = "" if competitive_score > 0 else "" if competitive_score == 0 else ""
            self.add_event(f" {player.name}: {vote_text} {competitive_indicator} ({chosen_turns}T) - {reasoning}")

            # Log vote in game logger
            self.game_logger.log_vote(player.id, vote, reasoning)

        # Analyze voting patterns for competitiveness
        ayes = sum(votes.values())
        total = len(votes)
        vote_breakdown = ", ".join([f"{p.name}: {'Aye' if votes.get(p.id, False) else 'Nay'}" for p in self.players])

        # Check for concerning unanimous patterns
        if ayes == total:
            self.add_event(" UNANIMOUS VOTE - Checking for lack of strategic competition...")
        elif ayes == 1:  # Only proposer voted yes
            self.add_event(" HIGHLY COMPETITIVE VOTE - Strong strategic opposition detected")

        self.add_event(f" VOTE RESULT: {ayes}/{total} Aye - {vote_breakdown}")

        # Store votes in the proposal
        proposal.votes = votes
        
        # Check if voting is complete (all players have voted) and continue if so
        self.check_voting_complete_and_continue()
        
        return votes

    def process_proposal_result(self, proposal):
        """Process voting results with proper Nomic mechanics"""
        # Prevent double processing
        if self.processing_proposal_result:
            return False
        self.processing_proposal_result = True
        
        try:
            proposer = next(p for p in self.players if p.id == proposal.player_id)
            ayes = sum(proposal.votes.values())
            total = len(proposal.votes)

            # Use dynamic voting threshold
            percentage = (ayes / total) * 100
            passes = percentage >= self.current_voting_threshold

            if passes:
                # Handle transmutation proposals first
                if proposal.is_transmutation and proposal.transmute_target_rule:
                    if self.execute_transmutation(proposal):
                        # Transmutation successful
                        proposer.points += 10
                        self.add_event(f" {proposer.name} gains 10 points  {proposer.points}")
                        
                        # Index the transmutation in RAG system
                        if hasattr(self, 'rag_system'):
                            self.rag_system.index_proposal(
                                proposal.id, f"Transmute rule {proposal.transmute_target_rule}", 
                                proposer.id, True, self.turn_number
                            )
                        
                        return True
                    else:
                        # Transmutation failed - treat as failed proposal
                        self.add_event(f" TRANSMUTATION FAILED - Invalid rule number {proposal.transmute_target_rule}")
                        # Execute vote_fail effects
                        self.execute_rule_effects("vote_fail", proposer=proposer)
                        return False
                
                # Handle amendment proposals second
                if proposal.is_amendment and proposal.amend_target_rule:
                    if self.execute_amendment(proposal):
                        # Amendment successful
                        proposer.points += 10
                        self.add_event(f" {proposer.name} gains 10 points  {proposer.points}")
                        
                        # Index the amendment in RAG system
                        if hasattr(self, 'rag_system'):
                            self.rag_system.index_proposal(
                                proposal.id, f"Amend rule {proposal.amend_target_rule}: {proposal.rule_text[:60]}...", 
                                proposer.id, True, self.turn_number
                            )
                        
                        return True
                    else:
                        # Amendment failed - treat as failed proposal
                        self.add_event(f" AMENDMENT FAILED - Rule {proposal.amend_target_rule} cannot be amended")
                        # Execute vote_fail effects
                        self.execute_rule_effects("vote_fail", proposer=proposer)
                        return False
                
                # Enhanced rule interpretation with advanced Game Master API
                if self.gm_api_enabled:
                    # Validate rule first
                    validation = self.advanced_gm.validate_rule(proposal.rule_text)
                    
                    if not validation['is_valid']:
                        self.add_event(f" Rule validation failed: {'; '.join(validation['errors'])}")
                        # Execute vote_fail effects and continue as failed proposal
                        self.execute_rule_effects("vote_fail", proposer=proposer)
                        return False
                    
                    # Warn about potential issues
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            self.add_event(f" GM Warning: {warning}")
                    
                    # Interpret rule into actions
                    self.add_event(f" Advanced GM interpreting rule (confidence: {validation['confidence']:.2f}, complexity: {validation['complexity']:.2f})")
                    
                    game_context = {
                        'rule_id': self.next_rule_number,
                        'turn_number': self.turn_number,
                        'proposer': proposer,
                        'players': self.players,
                        'current_rules': len(self.rules["mutable"]) + len(self.rules["immutable"])
                    }
                    
                    actions = self.advanced_gm.interpret_rule(proposal.rule_text, self.next_rule_number, game_context)
                    
                    if actions:
                        execution_result = self.advanced_gm.execute_actions(actions, game_context)
                        
                        if execution_result.success:
                            self.add_event(f" Advanced GM executed {len(execution_result.actions_taken)} actions successfully")
                            for effect in execution_result.effects_applied:
                                self.add_event(f" GM Effect: {effect}")
                        else:
                            self.add_event(f" GM execution partially failed: {'; '.join(execution_result.errors)}")
                            
                # Check if rule needs legacy GM interpretation
                rule_complexity = self.game_master._assess_complexity(proposal.rule_text)
                
                # If complex, use legacy GM system as backup
                if self.gm_enabled and rule_complexity > 0.3:
                    self.add_event(f" Legacy Game Master interpreting complex rule (complexity: {rule_complexity:.2f})")
                    
                    # Get game context for legacy GM
                    game_context = {
                        'turn_number': self.turn_number,
                        'num_players': self.num_players,
                        'victory_points': 100,
                        'current_rules': len(self.rules["mutable"]) + len(self.rules["immutable"])
                    }
                    
                    # Legacy GM interprets the rule
                    interpretation = self.game_master.interpret_rule(
                        proposal.rule_text,
                        self.next_rule_number,
                        game_context
                    )
                    
                    # Register with dynamic engine
                    if self.dynamic_rule_engine.register_rule(self.next_rule_number, interpretation):
                        self.add_event(f" Complex rule registered with Legacy Game Master")
                    else:
                        self.add_event(f" GM registration failed, using traditional parsing")
                
                # Create the new rule with parsed effects
                new_rule = ParsedRule(
                    id=self.next_rule_number,
                    text=proposal.rule_text,
                    mutable=True,
                    effects=proposal.parsed_effects if hasattr(proposal, "parsed_effects") else [],
                    author=proposal.player_id,
                    turn_added=self.turn_number,
                )
                self.rules["mutable"].append(new_rule)

                # Track successful rules
                self.agent_memory[proposal.player_id]["successful_rules"].append(proposal.rule_text)

                self.add_event(f" RULE {self.next_rule_number} ADOPTED ({ayes}/{total})")
                self.add_event(f" New rule: {proposal.rule_text}")

                # Execute any immediate effects
                self.execute_rule_effects("rule_adopted", proposer=proposer, rule=new_rule)

                # Standard proposal reward
                proposer.points += 10
                self.add_event(f" {proposer.name} gains 10 points  {proposer.points}")

                # Index successful proposal in RAG system
                if hasattr(self, 'rag_system'):
                    self.rag_system.index_proposal(
                        proposal.id, proposal.rule_text, proposer.id, True, self.turn_number
                    )
                    # Also index the new rule for future retrieval
                    self.rag_system.index_rule(
                        new_rule.id, new_rule.text, 
                        {"type": "mutable", "author": proposer.id, "turn_created": self.turn_number}
                    )

                # Increment rules created counter for session tracking
                if self.session_manager.current_session:
                    self.session_manager.current_session.rules_created += 1
                    self.session_manager.save_session_metadata()

                self.next_rule_number += 1

                # Execute vote_pass trigger for dissenting voters
                self.execute_rule_effects("vote_pass", proposer=proposer, votes=proposal.votes)
            else:
                # Track failed rules
                self.agent_memory[proposal.player_id]["failed_rules"].append(proposal.rule_text)

                self.add_event(f" PROPOSAL DEFEATED ({percentage:.0f}% < {self.current_voting_threshold}% required)")

                # Index failed proposal in RAG system for learning
                if hasattr(self, 'rag_system'):
                    self.rag_system.index_proposal(
                        proposal.id, proposal.rule_text, proposer.id, False, self.turn_number
                    )

                # Execute vote_fail effects
                self.execute_rule_effects("vote_fail", proposer=proposer)

            # Update performance metrics if we have the data
            if hasattr(proposal, "_performance_data"):
                proposal_data = proposal._performance_data
                proposal_data["passed"] = passes

                # Calculate points gained/lost for this proposal
                points_change = 10 if passes else -10

                # Update comprehensive performance tracking
                self.performance_manager.update_model_performance(
                    proposer.model,
                    proposal_data,
                    {"points_change": points_change, "game_turn": self.turn_number, "vote_percentage": percentage},
                )

                # Update model metrics directly for game stats
                if proposer.assigned_model_metrics:
                    if passes:
                        proposer.assigned_model_metrics.total_points_gained += 10
                    else:
                        proposer.assigned_model_metrics.total_points_gained -= 10

            # Record vote outcome and turn state
            self.record_vote_outcome(proposal, ayes, total, passes)
            self.record_turn_state(proposer, proposal, proposal.votes)

            # Log proposal outcome
            self.game_logger.log_proposal_outcome(proposal.player_id, proposal.rule_text, passes, ayes, total, percentage)

            return passes
        finally:
            self.processing_proposal_result = False

    def check_victory(self):
        """Check for victory conditions"""
        for player in self.players:
            if player.points >= 100:
                return player
        return None

    def play_turn(self):
        """Play one complete turn with comprehensive logging and idle turn processing"""
        if self.game_over:
            return

        # Prevent race conditions with turn lock
        with self.turn_lock:
            if self.turn_in_progress:
                self.add_event(f" Turn already in progress, skipping duplicate call")
                return
            self.turn_in_progress = True

        try:
            current_player = self.get_current_player()

            # Monitor game health and apply any necessary interventions
            if hasattr(self, 'failure_recovery'):
                health_report = self.failure_recovery.monitor_game_health({"game": self})
                if health_report["status"] != "healthy":
                    self.add_event(f"🏥 HEALTH CHECK: {health_report['status'].upper()}")
                    
                    # Log any interventions applied
                    for intervention in health_report["interventions_applied"]:
                        self.add_event(f"🚨 AUTO-RECOVERY: {intervention['intervention']}")
                        for action in intervention["actions"]:
                            self.add_event(f"   - {action}")
                    
                    # Log recommendations
                    for rec in health_report["recommendations"]:
                        self.add_event(f"⚠️  RECOMMENDATION: {rec['suggested_action']}")

            # Log turn start
            self.game_logger.log_turn_start(self.turn_number, current_player)
            self.add_event(f" Turn {self.turn_number} - {current_player.name}'s turn")

            # Process idle turns for other players (not for humans)
            if not current_player.is_human:
                self.process_idle_turns(current_player)

            # Handle human vs AI player turns differently
            if current_player.is_human:
                # For human players, wait for them to submit a proposal via web interface
                if not self.waiting_for_human:
                    self.add_event(
                        f" {current_player.name}, it's your turn! Please submit a proposal via the web interface."
                    )
                    self.waiting_for_human = True
                # The proposal will be set via the /human/propose route
                # So we just wait and return early - the game will continue when human submits
                return
            else:
                # Generate proposal using deliberation process with logging (AI players only)
                proposal_data = self.generate_proposal_with_deliberation(current_player)

                proposal = Proposal(
                    id=len(self.proposals) + 1,
                    player_id=current_player.id,
                    rule_text=proposal_data["rule_text"],
                    explanation=proposal_data["explanation"],
                    internal_thoughts=proposal_data["internal_thought"],
                    turn=self.turn_number,
                )

                # Log the proposal
                self.game_logger.log_proposal(current_player.id, proposal)

                self.proposals.append(proposal)
                self.current_proposal = proposal
                
                # Execute rule effects for proposal creation (e.g., rhyming bonus)
                self.execute_rule_effects("proposal_made", proposer=current_player, proposal=proposal)

            # Start async deliberation immediately for all other players
            self.start_async_deliberation(proposal)

            # Update player's internal state with the proposal
            self.state_tracker.add_planned_proposal(current_player.id, f"Proposed: {proposal.rule_text}")

            # Vote using competitive deliberation (this will handle turn advancement)
            self.vote_on_proposal_with_deliberation(proposal)
        
        finally:
            # Always release the turn lock
            with self.turn_lock:
                self.turn_in_progress = False

    def continue_turn_after_human_proposal(self):
        """Continue the turn flow after a human player has submitted a proposal"""
        if not self.current_proposal:
            return

        proposal = self.current_proposal
        current_player = self.get_current_player()

        # Vote using competitive deliberation (handles human voting via web interface)
        self.vote_on_proposal_with_deliberation(proposal)

        # Process results
        passes = self.process_proposal_result(proposal)

        # Update internal states based on outcome
        self.update_states_after_turn(current_player, proposal, passes)

        # Save current player states
        self.session_manager.save_player_states(self.state_tracker.get_all_states())

        # Finalize turn log
        self.game_logger.finalize_turn_log(self.turn_number)

        # Check victory
        winner = self.check_victory()
        if winner:
            self.game_over = True
            self.winner = winner
            self.add_event(f" {winner.name} WINS with {winner.points} points!")

            # End session
            final_scores = {p.id: p.points for p in self.players}
            self.session_manager.end_session(final_scores, winner.id, self.turn_number)

            # Update final game statistics for all players
            self.finalize_game_metrics()
            return

        # Check Rule 117 (zero points auto-lose)
        if self.check_zero_points_rule():
            # Game over - do not advance turn
            final_scores = {p.id: p.points for p in self.players}
            self.session_manager.end_session(final_scores, None, self.turn_number)
            self.finalize_game_metrics()
            return

        # Advance turn
        self.advance_turn()

    def check_voting_complete_and_continue(self):
        """Check if all players have voted and continue game if complete"""
        if not self.current_proposal:
            return

        # Check if all players have voted
        all_voted = all(player.id in self.current_proposal.votes for player in self.players)

        if all_voted:
            # All players have voted, proceed with the game
            proposal = self.current_proposal
            current_player = self.get_current_player()

            # Process results
            passes = self.process_proposal_result(proposal)

            # Update internal states based on outcome
            self.update_states_after_turn(current_player, proposal, passes)

            # Save current player states
            self.session_manager.save_player_states(self.state_tracker.get_all_states())

            # Finalize turn log
            self.game_logger.finalize_turn_log(self.turn_number)
            
            # Display cost tracking if using OpenRouter
            self.display_cost_tracking()

            # Check victory
            winner = self.check_victory()
            if winner:
                self.game_over = True
                self.winner = winner
                self.add_event(f" {winner.name} WINS with {winner.points} points!")

                # End session
                final_scores = {p.id: p.points for p in self.players}
                self.session_manager.end_session(final_scores, winner.id, self.turn_number)

                # Update final game statistics for all players
                self.finalize_game_metrics()
                return

            # Check Rule 117 (zero points auto-lose)
            if self.check_zero_points_rule():
                # Game over - do not advance turn
                final_scores = {p.id: p.points for p in self.players}
                self.session_manager.end_session(final_scores, None, self.turn_number)
                self.finalize_game_metrics()
                return

            # Advance turn
            self.advance_turn()

    def process_idle_turns(self, current_player: Player):
        """Process idle turns for players who are not currently active"""
        for player in self.players:
            if player.id == current_player.id:
                continue  # Skip current player

            player_state = self.state_tracker.get_player_state(player.id)
            if not player_state or not self.idle_processor.can_process_idle_turn(player, player_state):
                continue

            # Create game state for idle turn processing
            sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
            game_state = {
                "turn": self.turn_number,
                "players": [{"id": p.id, "name": p.name, "points": p.points} for p in sorted_players],
                "mutable_rules": [{"id": r.id, "text": r.text} for r in self.rules["mutable"]],
                "recent_proposals": (
                    [{"player_id": p.player_id, "rule_text": p.rule_text} for p in self.proposals[-3:]]
                    if self.proposals
                    else []
                ),
                "context_string": self.get_game_context(),
            }
            game_state["player_states"] = {
                pid: asdict(state) for pid, state in self.state_tracker.get_all_states().items()
            }

            try:
                # Process idle turn
                idle_turn = self.idle_processor.process_idle_turn(player, game_state, self.turn_number)

                # Update player's internal state
                player_state.idle_turn_count += 1

                # Update strategic insights from idle turn
                if idle_turn.strategic_analysis:
                    self.state_tracker.update_strategic_focus(player.id, idle_turn.strategic_analysis)
                if idle_turn.plan_refinement:
                    self.state_tracker.get_player_state(player.id).victory_path = idle_turn.plan_refinement
                if idle_turn.learned_insights:
                    self.state_tracker.add_learning_observation(player.id, idle_turn.learned_insights)

                # Log the idle turn
                self.game_logger.log_idle_turn(idle_turn)
                self.session_manager.save_idle_turn_log(idle_turn)

                self.add_event(f" {player.name} processes idle turn {player_state.idle_turn_count}/3")

            except Exception as e:
                self.add_event(f" Error processing idle turn for {player.name}: {str(e)}")

    def update_states_after_turn(self, current_player: Player, proposal: Proposal, passed: bool):
        """Update player internal states after a turn"""
        # Update proposer's state
        if passed:
            self.state_tracker.add_learning_observation(current_player.id, f"Successful proposal: {proposal.rule_text}")
        else:
            self.state_tracker.add_learning_observation(
                current_player.id, f"Failed proposal: {proposal.rule_text} - need better strategy"
            )

        # Update other players' threat assessments
        for player in self.players:
            if player.id == current_player.id:
                continue

            # Update threat assessment based on proposal success
            if passed and current_player.points > player.points:
                threat_level = "High threat - successful proposals and leading"
            elif passed:
                threat_level = "Moderate threat - successful proposals"
            elif current_player.points > player.points + 20:
                threat_level = "High threat - significant point lead"
            else:
                threat_level = "Low threat - struggling with proposals"

            self.state_tracker.update_threat_assessment(player.id, current_player.id, threat_level)

    def advance_turn(self):
        """Advance to next player"""
        # Double-check that game isn't over before advancing
        if self.game_over:
            self.add_event(" Cannot advance turn - game is over")
            return
            
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        if self.current_player_idx == 0:
            self.turn_number += 1
        self.current_proposal = None
        self.waiting_for_human = False  # Clear flag when advancing turns
        
        # Log the turn advancement for debugging
        next_player = self.get_current_player()
        self.add_event(f" Turn advanced to {next_player.name} (Turn {self.turn_number})")
        
        # If next player is AI, immediately trigger their turn
        if not next_player.is_human and not self.game_over:
            self.add_event(f" Immediately starting {next_player.name}'s AI turn...")
            # Use threading to avoid blocking
            def start_next_turn():
                time.sleep(1)  # Brief delay to ensure state is clean
                if not self.waiting_for_human and not self.game_over and not self.turn_in_progress:
                    self.turn_in_progress = True
                    try:
                        self.play_turn()
                    finally:
                        self.turn_in_progress = False
            
            threading.Thread(target=start_next_turn, daemon=True).start()
    
    def display_cost_tracking(self):
        """Display cost tracking information for OpenRouter usage"""
        if hasattr(self.llm_provider, 'get_total_session_cost'):
            total_cost = self.llm_provider.get_total_session_cost()
            budget = getattr(self.llm_provider, 'session_budget', 5.0)
            
            if total_cost > 0:
                usage_percent = (total_cost / budget) * 100
                print(f"\n💰 Session Cost: ${total_cost:.4f} / ${budget:.2f} ({usage_percent:.1f}%)")
                
                # Show per-model breakdown
                costs = self.llm_provider.get_session_costs()
                if costs:
                    print("   Model breakdown:")
                    for model, model_costs in costs.items():
                        model_name = model.split('/')[-1]  # Get just the model name
                        print(f"     {model_name}: ${model_costs['total_cost']:.4f} ({model_costs['input_tokens']:,} in, {model_costs['output_tokens']:,} out)")

    def start_async_deliberation(self, proposal):
        """Start background deliberation for all non-proposing players immediately when proposal is made"""
        proposer_id = proposal.player_id
        self.add_event(
            f" Starting async deliberation for all players on proposal by {next(p.name for p in self.players if p.id == proposer_id)}"
        )

        for player in self.players:
            # Skip the proposer and human players for now
            if player.id == proposer_id or player.is_human:
                continue

            self.add_event(f" {player.name} begins thinking about the proposal...")

            # Start first turn of deliberation immediately
            try:
                # Get initial analysis prompt
                sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
                my_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == player.id)
                proposer = next(p for p in self.players if p.id == proposer_id)
                proposer_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == proposer_id)

                initial_prompt = f"""ASYNC DELIBERATION - INITIAL ANALYSIS
                
PROPOSAL JUST MADE: "{proposal.rule_text}"
Explanation: {proposal.explanation}
Proposed by: {proposer.name} (Rank #{proposer_rank}, {proposer.points} points)
Your Position: {player.name} (Rank #{my_rank}, {player.points} points)

You have time to think before voting begins. Start your strategic analysis:

1. How does this rule affect YOUR path to 100 points?
2. What are the proposer likely motivations?
3. How might other players react to this?
4. What are the immediate vs long-term implications?

This is just your initial thinking - you will get more information before final vote.
No format required, just think strategically about the implications."""

                # Generate initial deliberation
                initial_response = self.unified_generate(player, initial_prompt, temperature=0.7, max_tokens=1500)

                # Store this initial analysis for later use in voting
                if not hasattr(player, "async_deliberation"):
                    player.async_deliberation = {}

                player.async_deliberation[proposal.id] = {
                    "initial_analysis": initial_response,
                    "timestamp": datetime.now(),
                }

                self.add_event(f" {player.name} completed initial analysis")

            except Exception as e:
                self.add_event(f" Error in async deliberation for {player.name}: {str(e)}")

    def finalize_game_metrics(self):
        """Update final game statistics for all players"""
        for player in self.players:
            if player.assigned_model_metrics:
                metrics = player.assigned_model_metrics
                metrics.total_games += 1

                # Track wins
                if self.winner and player.id == self.winner.id:
                    metrics.games_won += 1

                # Update final metrics
                metrics.last_updated = datetime.now().isoformat()

                # Add performance snapshot to history
                performance_snapshot = {
                    "game_date": datetime.now().isoformat(),
                    "final_points": player.points,
                    "won_game": self.winner and player.id == self.winner.id,
                    "proposals_made": len([p for p in self.proposals if p.player_id == player.id]),
                    "proposals_passed": len(
                        [
                            p
                            for p in self.proposals
                            if p.player_id == player.id and p.votes and sum(p.votes.values()) == len(p.votes)
                        ]
                    ),
                    "turn_count": self.turn_number,
                }
                metrics.performance_history.append(performance_snapshot)

                # Keep only last 10 games in history
                metrics.performance_history = metrics.performance_history[-10:]

        # Save all metrics
        self.performance_manager.save_metrics()
        
        # Update proto character statistics and get reflections
        self.update_character_stats_and_reflections()

        # Log game completion
        self.game_logger.log_game_end(
            self.winner.id if self.winner else None,
            self.turn_number,
            {player.id: player.points for player in self.players},
        )

        # Log final statistics
        self.add_event(" Final Model Performance Summary:")
        for player in self.players:
            if player.assigned_model_metrics:
                overall_score = player.assigned_model_metrics.calculate_overall_score()
                self.add_event(f"  {player.name} ({player.model}): Overall Score: {overall_score:.1f}")

        # Get model statistics for display
        model_stats = self.performance_manager.get_model_statistics()
        self.add_event(" Cross-Game Model Rankings:")
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["overall_score"], reverse=True)
        for i, (model, stats) in enumerate(sorted_models[:3], 1):
            self.add_event(f"  #{i} {model}: {stats['overall_score']:.1f} points ({stats['games_played']} games)")

    def update_character_stats_and_reflections(self):
        """Update character statistics and get reflections for next game"""
        self.add_event(" Updating character memories and gathering reflections...")
        
        for player in self.players:
            if not player.character or player.is_human:
                continue
                
            character = player.character
            won = self.winner and player.id == self.winner.id
            
            # Update character stats
            character.update_after_game(player.points, won, "")
            
            # Generate reflection for next game
            reflection_prompt = f""" END-GAME REFLECTION for {character.name}

GAME SUMMARY:
Final Result: {" WON" if won else " Lost"} 
- Final Points: {player.points}/100
- Your Strategy: {character.strategy}
- Games Played: {character.games_played} (Win Rate: {character.get_win_rate()} percent)

REFLECTION TASK:
What is ONE key insight or lesson you want to remember for your next Nomic game? 
This thought will be included in your context for future games.

Think about:
- What worked well or poorly in your strategy this game?
- What did you learn about other players or the game dynamics?
- What would you do differently next time?

Provide a single, concise thought (max 200 characters) for your future self:

PERSISTENT_THOUGHT: [Your key insight for next game]"""

            try:
                reflection_response = self.unified_generate(player, reflection_prompt, temperature=0.7, max_tokens=200)
                
                # Extract the persistent thought
                if "PERSISTENT_THOUGHT:" in reflection_response:
                    thought = reflection_response.split("PERSISTENT_THOUGHT:")[1].strip()
                    character.set_persistent_thought(thought)
                    self.add_event(f" {character.name}: \"{thought[:100]}{'...' if len(thought) > 100 else ''}\"")
                else:
                    # Fallback thought
                    fallback = f"Game {character.games_played}: {'Won' if won else 'Lost'} with {player.points} points"
                    character.set_persistent_thought(fallback)
                    
            except Exception as e:
                # Error generating reflection
                self.add_event(f"WARNING: Could not generate reflection for {character.name}: {e}")
                fallback = f"Game {character.games_played}: Experience gained"
                character.set_persistent_thought(fallback)
        
        # Save updated character data
        self.character_manager.save_characters()
        self.add_event(" Character memories saved for future games")

    def add_event(self, message):
        """Add game event"""
        self.events.append(
            {"id": len(self.events), "message": message, "timestamp": datetime.now().strftime("%H:%M:%S")}
        )
        print(f"[{self.events[-1]['timestamp']}] {message}")

    def start_game_thread(self):
        """Start game in background thread"""

        def run_game():
            while not self.game_over and self.turn_number < 15:
                # Only call play_turn if we're not waiting for human input
                if not self.waiting_for_human:
                    self.play_turn()
                    time.sleep(4)  # Normal pace for AI turns
                else:
                    time.sleep(5)  # Check every 5 seconds when waiting for human
                    
                # Additional check to ensure game continues after turn completion
                current_player = self.get_current_player()
                if not self.waiting_for_human and not current_player.is_human:
                    # If we're not waiting for human and current player is AI, ensure we continue
                    continue

        thread = threading.Thread(target=run_game, daemon=True)
        thread.start()


# Flask App Factory for Port-Specific Instances
def create_app():
    """Create Flask app instance for port-specific game isolation"""
    port_app = Flask(__name__)
    port_app.config['game'] = None  # Will be set when game is created
    
    # Import all route functions and register them with this app instance
    register_routes(port_app)
    
    return port_app

def register_routes(app_instance):
    """Register all routes with the given Flask app instance"""
    # Copy all routes from the global app to this app instance
    # This is done automatically by Flask when using the same function references
    # The key fix is that each route now uses current_app.config['game'] instead of global game
    pass

# Default app instance (will be replaced with port-specific instances)
app = create_app()


@app.route("/")
def index():
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Proper Nomic Game</title>
    <style>
        /* Light mode (default) */
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #f5f5f5; 
            color: #333;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .card { 
            background: white; 
            padding: 20px; 
            margin: 10px 0; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .triple { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .player { 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
            border: 2px solid #ddd;
            transition: border-color 0.3s ease, background-color 0.3s ease;
        }
        .player.current { border-color: #007bff; background: #e3f2fd; }
        .event { 
            padding: 5px 10px; 
            margin: 3px 0; 
            border-radius: 3px; 
            background: #f8f9fa; 
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        .rule { 
            padding: 8px; 
            margin: 3px 0; 
            border-radius: 4px; 
            font-size: 13px;
            transition: background-color 0.3s ease;
        }
        .immutable { background: #ffebee; border-left: 4px solid #f44336; }
        .mutable { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .new-rule { background: #fff3e0; border-left: 4px solid #ff9800; font-weight: bold; }
        button { 
            padding: 10px 20px; 
            background: #007bff; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px;
            transition: background-color 0.3s ease;
        }
        button:hover { background: #0056b3; }
        h1, h2, h3 { 
            color: #333;
            transition: color 0.3s ease;
        }
        .winner { background: #fff3cd; border: 2px solid #ffc107; padding: 15px; text-align: center; font-size: 18px; font-weight: bold; }
        .proposal-details { 
            background: #f0f7ff; 
            padding: 15px; 
            border-radius: 5px; 
            margin: 10px 0;
            transition: background-color 0.3s ease;
        }
        .internal-thought { 
            background: #f5f5f5; 
            padding: 10px; 
            border-radius: 3px; 
            font-style: italic; 
            margin: 5px 0;
            transition: background-color 0.3s ease;
        }
        .explanation { 
            background: #e8f5e9; 
            padding: 10px; 
            border-radius: 3px; 
            margin: 5px 0;
            transition: background-color 0.3s ease;
        }
        .port-info { 
            font-size: 12px; 
            color: #666;
            transition: color 0.3s ease;
        }
        .vote-aye { color: #4caf50; font-weight: bold; background: #e8f5e9; padding: 3px 8px; border-radius: 3px; }
        .vote-nay { color: #f44336; font-weight: bold; background: #ffebee; padding: 3px 8px; border-radius: 3px; }
        .event.important { background: #fff3e0; border-left: 4px solid #ff9800; font-weight: bold; }
        .event.proposal { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .event.vote-result { background: #f3e5f5; border-left: 4px solid #9c27b0; font-weight: bold; }

        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            body { 
                background: #1a1a1a; 
                color: #e0e0e0;
            }
            .card { 
                background: #2d2d2d; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .player { 
                border-color: #444;
                background: #2a2a2a;
            }
            .player.current { 
                border-color: #4a9eff; 
                background: #1e3a5f;
            }
            .event { 
                background: #333;
                color: #e0e0e0;
            }
            .immutable { 
                background: #4a1a1a; 
                border-left: 4px solid #f44336; 
            }
            .mutable { 
                background: #1a3d1a; 
                border-left: 4px solid #4caf50; 
            }
            .new-rule { 
                background: #3d2a1a; 
                border-left: 4px solid #ff9800; 
                font-weight: bold; 
            }
            button { 
                background: #4a9eff; 
            }
            button:hover { 
                background: #357abd; 
            }
            h1, h2, h3 { 
                color: #e0e0e0;
            }
            .winner { 
                background: #4a3d1a; 
                border: 2px solid #ffc107; 
                color: #ffc107;
            }
            .proposal-details { 
                background: #1a2a3d;
                color: #e0e0e0;
            }
            .internal-thought { 
                background: #333;
                color: #e0e0e0;
            }
            .explanation { 
                background: #1a3d1a;
                color: #e0e0e0;
            }
            .port-info { 
                color: #999;
            }
            .vote-aye { 
                background: #1a3d1a; 
            }
            .vote-nay { 
                background: #4a1a1a; 
            }
            .event.important { 
                background: #3d2a1a; 
                color: #ffcc80;
            }
            .event.proposal { 
                background: #1a2a3d; 
                color: #bbdefb;
            }
            .event.vote-result { 
                background: #2a1a3d; 
                color: #e1bee7;
            }
        }
    </style>
    <script>
        function startGame() {
            fetch('/start', {method: 'POST'})
                .then(() => { 
                    document.getElementById('status').textContent = 'Game Started!';
                    startPolling();
                });
        }
        
        let updateInterval;
        let userIsTyping = false;
        let typingTimeout;
        
        function startPolling() {
            updateInterval = setInterval(updateGameSmart, 10000); // Default 10 seconds
        }
        
        function updateGameSmart() {
            // Don't update if user is actively typing
            if (userIsTyping) {
                console.log("Skipping update - user is typing");
                return;
            }
            updateGame();
        }
        
        function trackUserTyping() {
            userIsTyping = true;
            clearTimeout(typingTimeout);
            
            // Reset typing flag after 5 seconds of inactivity
            typingTimeout = setTimeout(() => {
                userIsTyping = false;
                console.log("User stopped typing - updates will resume");
            }, 5000);
        }
        
        function updateGame() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.error) return;
                    
                    document.getElementById('turn').textContent = data.turn;
                    document.getElementById('current').textContent = data.current_player || '-';
                    
                    // Check if current player is human and show appropriate interface
                    const currentPlayer = data.players.find(p => p.current);
                    if (currentPlayer && currentPlayer.is_human) {
                        // Show human player interface
                        if (data.current_proposal && !data.current_proposal.votes.hasOwnProperty(currentPlayer.id)) {
                            // Human needs to vote
                            document.getElementById('human-action').innerHTML = 
                                `<div class="card" style="background: #fff3cd; border: 2px solid #ffc107;">
                                    <h3> Your Turn to Vote!</h3>
                                    <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                        <strong>Proposal:</strong> "${data.current_proposal.rule_text}"<br>
                                        <strong>Explanation:</strong> ${data.current_proposal.explanation}
                                    </div>
                                    <div style="text-align: center; margin: 20px 0;">
                                        <button onclick="submitHumanVote(true)" style="background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 4px; font-size: 16px; margin: 10px;">
                                             Vote AYE (Yes)
                                        </button>
                                        <button onclick="submitHumanVote(false)" style="background: #dc3545; color: white; padding: 12px 24px; border: none; border-radius: 4px; font-size: 16px; margin: 10px;">
                                             Vote NAY (No)
                                        </button>
                                    </div>
                                </div>`;
                        } else if (!data.current_proposal) {
                            // Save current form state before regenerating HTML
                            const existingDropdown = document.getElementById('human-rule-type');
                            const existingRuleText = document.getElementById('human-rule-text');
                            const existingTransmuteNumber = document.getElementById('human-transmute-number');
                            const existingExplanation = document.getElementById('human-explanation');
                            
                            // Save which element has focus and cursor position
                            const activeElement = document.activeElement;
                            const focusedElementId = activeElement && activeElement.id ? activeElement.id : null;
                            const cursorPosition = activeElement && activeElement.selectionStart ? activeElement.selectionStart : 0;
                            
                            const savedState = {
                                ruleType: existingDropdown ? existingDropdown.value : 'new',
                                ruleText: existingRuleText ? existingRuleText.value : '',
                                transmuteNumber: existingTransmuteNumber ? existingTransmuteNumber.value : '',
                                explanation: existingExplanation ? existingExplanation.value : '',
                                focusedElement: focusedElementId,
                                cursorPosition: cursorPosition
                            };
                            
                            // Human needs to propose
                            document.getElementById('human-action').innerHTML = 
                                `<div class="card" style="background: #d1ecf1; border: 2px solid #17a2b8;">
                                    <h3> Your Turn to Propose!</h3>
                                    <div style="margin: 15px 0;">
                                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Rule Type:</label>
                                        <select id="human-rule-type" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                                            <option value="new">New Rule</option>
                                            <option value="transmute">Transmute Existing Rule</option>
                                        </select>
                                    </div>
                                    
                                    <div id="human-new-rule-section">
                                        <div style="margin: 15px 0;">
                                            <label style="display: block; margin-bottom: 5px; font-weight: bold;">Rule Text:</label>
                                            <textarea id="human-rule-text" placeholder="Enter your new rule here..." style="width: 100%; height: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                                        </div>
                                    </div>
                                    
                                    <div id="human-transmute-section" style="display: none;">
                                        <div style="margin: 15px 0;">
                                            <label style="display: block; margin-bottom: 5px; font-weight: bold;">Rule Number to Transmute:</label>
                                            <input type="number" id="human-transmute-number" placeholder="e.g., 301" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                                            <small>Enter the number of an existing rule to change from immutable  mutable</small>
                                        </div>
                                    </div>

                                    <div style="margin: 15px 0;">
                                        <label style="display: block; margin-bottom: 5px; font-weight: bold;">Explanation:</label>
                                        <textarea id="human-explanation" placeholder="Explain why this rule is good for the game..." style="width: 100%; height: 60px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"></textarea>
                                    </div>

                                    <div style="text-align: center; margin: 20px 0;">
                                        <button onclick="submitHumanProposal()" style="background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; font-size: 16px;">
                                             Submit Proposal
                                        </button>
                                    </div>
                                </div>`;
                            
                            // Restore saved form state
                            const ruleTypeDropdown = document.getElementById('human-rule-type');
                            const ruleTextArea = document.getElementById('human-rule-text');
                            const transmuteNumberInput = document.getElementById('human-transmute-number');
                            const explanationArea = document.getElementById('human-explanation');
                            
                            if (ruleTypeDropdown) ruleTypeDropdown.value = savedState.ruleType;
                            if (ruleTextArea) ruleTextArea.value = savedState.ruleText;
                            if (transmuteNumberInput) transmuteNumberInput.value = savedState.transmuteNumber;
                            if (explanationArea) explanationArea.value = savedState.explanation;
                            
                            // Show/hide sections based on saved rule type
                            const newSection = document.getElementById('human-new-rule-section');
                            const transmuteSection = document.getElementById('human-transmute-section');
                            if (savedState.ruleType === 'transmute') {
                                newSection.style.display = 'none';
                                transmuteSection.style.display = 'block';
                            } else {
                                newSection.style.display = 'block';
                                transmuteSection.style.display = 'none';
                            }
                            
                            // Add event listener for rule type change after HTML is created
                            document.getElementById('human-rule-type').addEventListener('change', function() {
                                const ruleType = this.value;
                                const newSection = document.getElementById('human-new-rule-section');
                                const transmuteSection = document.getElementById('human-transmute-section');
                                
                                if (ruleType === 'transmute') {
                                    newSection.style.display = 'none';
                                    transmuteSection.style.display = 'block';
                                } else {
                                    newSection.style.display = 'block';
                                    transmuteSection.style.display = 'none';
                                }
                            });
                            
                            // Add typing detection to prevent interruptions
                            const textElements = ['human-rule-text', 'human-transmute-number', 'human-explanation'];
                            textElements.forEach(id => {
                                const element = document.getElementById(id);
                                if (element) {
                                    element.addEventListener('input', trackUserTyping);
                                    element.addEventListener('keydown', trackUserTyping);
                                    element.addEventListener('focus', trackUserTyping);
                                }
                            });
                            
                            // Restore focus and cursor position if user was typing
                            if (savedState.focusedElement) {
                                const focusElement = document.getElementById(savedState.focusedElement);
                                if (focusElement) {
                                    setTimeout(() => {
                                        focusElement.focus();
                                        if (focusElement.setSelectionRange && savedState.cursorPosition) {
                                            focusElement.setSelectionRange(savedState.cursorPosition, savedState.cursorPosition);
                                        }
                                    }, 10); // Small delay to ensure DOM is ready
                                }
                            }
                        } else {
                            document.getElementById('human-action').innerHTML = '';
                        }
                    } else {
                        document.getElementById('human-action').innerHTML = '';
                    }
                    
                    // Update players
                    const playersHtml = data.players.map(p => 
                        `<div class="player ${p.current ? 'current' : ''}">
                            <strong>${p.name}${p.is_human ? ' ' : ''}</strong> (${p.role.split(' - ')[0]})<br>
                            Points: ${p.points} | Model: ${p.model}
                            ${!p.is_human ? `<div class="port-info">Port: ${p.port}</div>` : ''}
                        </div>`
                    ).join('');
                    document.getElementById('players').innerHTML = playersHtml;
                    
                    // Update rules
                    const immutableHtml = data.rules.immutable.map(rule => 
                        `<div class="rule immutable">Rule ${rule.id}: ${rule.text}</div>`
                    ).join('');
                    const mutableHtml = data.rules.mutable.map((rule, i) => {
                        const isNew = i >= data.rules.mutable.length - 2;
                        let effectsHtml = '';
                        if (rule.effects && rule.effects.length > 0) {
                            effectsHtml = '<div style="margin-top: 5px; font-size: 12px; color: #666;">' + 
                                rule.effects.map(e => 
                                    `<div style="margin-left: 20px;"> ${e.description} (${e.trigger})</div>`
                                ).join('') + '</div>';
                        }
                        return `<div class="rule ${isNew ? 'new-rule' : 'mutable'}">Rule ${rule.id}: ${rule.text}${effectsHtml}</div>`;
                    }).join('');
                    document.getElementById('immutable-rules').innerHTML = immutableHtml;
                    document.getElementById('mutable-rules').innerHTML = mutableHtml;
                    
                    // Update failed proposals
                    const failedProposals = data.failed_proposals || [];
                    let failedHtml = '';
                    if (failedProposals.length === 0) {
                        failedHtml = '<em>No failed proposals yet</em>';
                    } else {
                        failedHtml = failedProposals.map(proposal => {
                            const proposer = data.players.find(p => p.id === proposal.player_id);
                            const proposerName = proposer ? proposer.name : `Player ${proposal.player_id}`;
                            return `<div class="failed-proposal" style="margin: 10px 0; padding: 10px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px;">
                                <strong>Turn ${proposal.turn}:</strong> ${proposerName}<br>
                                <em>"${proposal.rule_text}"</em><br>
                                <small>Votes: ${proposal.votes_summary || 'Not available'}</small>
                            </div>`;
                        }).join('');
                    }
                    document.getElementById('failed-proposals').innerHTML = failedHtml;
                    
                    // Update current proposal
                    if (data.current_proposal) {
                        const prop = data.current_proposal;
                        const voteEntries = Object.entries(prop.votes || {});
                        const ayes = voteEntries.filter(([id, vote]) => vote).length;
                        const total = voteEntries.length;
                        const voteHtml = voteEntries.map(([id, vote]) => {
                            const player = data.players.find(p => p.name.includes(id) || p.id == id);
                            const playerName = player ? player.name : `Player ${id}`;
                            const voteClass = vote ? 'vote-aye' : 'vote-nay';
                            return `<span class="${voteClass}">${playerName}: ${vote ? 'Aye' : 'Nay'}</span>`;
                        }).join(' | ');
                        
                        document.getElementById('current-proposal').innerHTML = `
                            <div class="proposal-details">
                                <h3 style="color: #2196F3;">RULE ${prop.next_number} PROPOSAL</h3>
                                <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ff9800;">
                                    <strong>Rule Text:</strong> "${prop.rule_text}"
                                </div>
                                <div class="explanation"><strong>Public Explanation:</strong> ${prop.explanation}</div>
                                <div style="margin-top: 15px;">
                                    <strong>Voting Status (${ayes}/${total} Aye):</strong><br>
                                    <div style="margin-top: 5px;">${voteHtml || 'Voting in progress...'}</div>
                                </div>
                            </div>
                        `;
                        document.getElementById('proposal-section').style.display = 'block';
                    } else {
                        document.getElementById('proposal-section').style.display = 'none';
                    }
                    
                    // Update events with proper styling
                    const eventsHtml = data.events.slice(-15).map(e => {
                        let eventClass = 'event';
                        if (e.message.includes('proposes Rule') || e.message.includes('')) {
                            eventClass += ' proposal';
                        } else if (e.message.includes('VOTE RESULT') || e.message.includes('')) {
                            eventClass += ' vote-result';
                        } else if (e.message.includes('ADOPTED') || e.message.includes('DEFEATED') || e.message.includes('') || e.message.includes('')) {
                            eventClass += ' important';
                        }
                        return `<div class="${eventClass}">[${e.timestamp}] ${e.message}</div>`;
                    }).join('');
                    document.getElementById('events').innerHTML = eventsHtml;
                    
                    // Check winner
                    if (data.winner) {
                        document.getElementById('winner').innerHTML = 
                            `<div class="winner"> ${data.winner} WINS! </div>`;
                    }
                });
        }

        function submitHumanProposal() {
            const ruleType = document.getElementById('human-rule-type').value;
            const ruleText = document.getElementById('human-rule-text').value;
            const transmuteNumber = document.getElementById('human-transmute-number').value;
            const explanation = document.getElementById('human-explanation').value;
            
            if (ruleType === 'new' && !ruleText.trim()) {
                alert('Please enter rule text');
                return;
            }
            if (ruleType === 'transmute' && !transmuteNumber) {
                alert('Please enter rule number to transmute');
                return;
            }
            if (!explanation.trim()) {
                alert('Please provide an explanation');
                return;
            }

            const data = {
                rule_type: ruleType,
                rule_text: ruleType === 'new' ? ruleText : `Transmute rule ${transmuteNumber}`,
                transmute_number: ruleType === 'transmute' ? parseInt(transmuteNumber) : null,
                explanation: explanation,
                effects: {} // Simplified - no checkbox system in inline version
            };

            fetch('/human/propose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    alert('Proposal submitted! Proceeding to voting...');
                    document.getElementById('human-action').innerHTML = '<div class="card"><h3> Proposal Submitted!</h3><p>Waiting for voting phase...</p></div>';
                } else {
                    alert('Error: ' + result.error);
                }
            });
        }

        function submitHumanVote(vote) {
            fetch('/human/submit-vote', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    vote: vote,
                    reasoning: vote ? "Human voted AYE" : "Human voted NAY"
                })
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    alert(vote ? 'Voted AYE!' : 'Voted NAY!');
                    document.getElementById('human-action').innerHTML = '<div class="card"><h3> Vote Submitted!</h3><p>Waiting for other players...</p></div>';
                } else {
                    alert('Error: ' + result.error);
                }
            });
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1> Proper Nomic Game</h1>
            <p><strong>Multi-Instance Ollama - Validated Rules - Structured Proposals</strong></p>
            <button onclick="startGame()">Start New Game</button>
            <div id="status">Ready to start</div>
            <p><strong>Turn:</strong> <span id="turn">0</span> | <strong>Current Player:</strong> <span id="current">-</span></p>
        </div>
        
        <div id="winner"></div>
        
        <div id="human-action"></div>
        
        <div id="proposal-section" style="display: none;" class="card">
            <h2> Current Proposal</h2>
            <div id="current-proposal"></div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2> Players</h2>
                <div id="players">No game started</div>
            </div>
            
            <div class="card">
                <h2> Game Events</h2>
                <div id="events" style="height: 400px; overflow-y: auto;">No events yet</div>
            </div>
        </div>
        
        <div class="triple">
            <div class="card">
                <h3> Immutable Rules</h3>
                <div id="immutable-rules" style="max-height: 500px; overflow-y: auto; font-size: 14px; line-height: 1.4;">No rules loaded</div>
            </div>
            <div class="card">
                <h3> Mutable Rules</h3>
                <div id="mutable-rules" style="max-height: 500px; overflow-y: auto; font-size: 14px; line-height: 1.4;">No rules loaded</div>
            </div>
            <div class="card">
                <h3> Failed Proposals</h3>
                <div id="failed-proposals" style="max-height: 500px; overflow-y: auto; font-size: 14px; line-height: 1.4;">No failed proposals yet</div>
            </div>
        </div>
        
        <div class="card">
            <h3> System Info</h3>
                <div style="font-size: 13px;">
                    <p><strong>Validator:</strong> llama3:8b on port 11435</p>
                    <p><strong>Player Instances:</strong> Ports 11436-11441</p>
                    <p><strong>Rule Numbering:</strong> System-controlled</p>
                    <p><strong>Response Parsing:</strong> Structured format</p>
                    <p><strong>Validation:</strong> Active rule checking</p>
                </div>
            </div>
        </div>
        
        <div class="card" style="text-align: center; margin-top: 20px;">
            <h3> Enhanced Analysis & Logs</h3>
            <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <a href="/stats" style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Model Performance Stats</a>
                <a href="/player-states" style="background: linear-gradient(45deg, #2c3e50, #34495e); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Player States & Scratchpads</a>
                <a href="/game-logs" style="background: linear-gradient(45deg, #1e3c72, #2a5298); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Detailed Game Logs</a>
                <a href="/sessions" style="background: linear-gradient(45deg, #8e44ad, #3498db); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Session History</a>
                <a href="/analytics" style="background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Deep Analytics</a>
                <a href="/api/enhanced-logs" style="background: linear-gradient(45deg, #00b894, #00a085); color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; display: inline-block;"> Enhanced Logs API</a>
            </div>
        </div>
    </div>
</body>
</html>
    """
    )


@app.route("/start", methods=["POST"])
def start_game():
    # The game instance is already created in main block with correct provider
    from flask import current_app
    game = current_app.config.get('game')
    if game:
        game.start_game_thread()
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "No game instance available"})


@app.route("/new-game", methods=["POST"])
def new_game():
    """Create a brand new game instance"""
    from flask import current_app
    try:
        # Get provider from current game if available
        provider = None
        current_game = current_app.config.get('game')
        if current_game and hasattr(current_game, 'llm_provider'):
            provider = current_game.llm_provider
        
        # Create new game instance
        new_game_instance = ProperNomicGame(
            num_players=4,  # Default to 4 players
            provider=provider, 
            include_human=False
        )
        
        # Store in app context instead of global
        current_app.config['game'] = new_game_instance
        
        return jsonify({
            "status": "new_game_created",
            "session_id": new_game_instance.session_id,
            "message": "New game created successfully! You can now start it."
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create new game: {str(e)}"})


@app.route("/status")
def get_status():
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    current_proposal_data = None
    if game.current_proposal:
        current_proposal_data = {
            "rule_text": game.current_proposal.rule_text,
            "explanation": game.current_proposal.explanation,
            "votes": game.current_proposal.votes,
            "next_number": game.next_rule_number,
        }

    return jsonify(
        {
            "turn": game.turn_number,
            "current_player": game.get_current_player().name if not game.game_over else None,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role,
                    "points": p.points,
                    "model": p.model,
                    "port": p.port,
                    "current": p.id == game.get_current_player().id,
                    "is_human": p.is_human,
                }
                for p in game.players
            ],
            "rules": {
                "immutable": [{"id": r.id, "text": r.text} for r in game.rules["immutable"]],
                "mutable": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "effects": (
                            [
                                {
                                    "description": e.description,
                                    "trigger": e.trigger,
                                    "action": e.action,
                                    "value": e.value,
                                }
                                for e in r.effects
                            ]
                            if hasattr(r, "effects")
                            else []
                        ),
                        "author": r.author if hasattr(r, "author") else None,
                        "turn_added": r.turn_added if hasattr(r, "turn_added") else None,
                    }
                    for r in game.rules["mutable"]
                ],
            },
            "current_proposal": current_proposal_data,
            "events": game.events[-30:],
            "vote_history": game.vote_history[-10:],
            "turn_history": game.turn_history[-5:],
            "proposals_count": len(game.proposals),
            "voting_threshold": game.current_voting_threshold,
            "point_affecting_rules": [
                {
                    "rule_id": r.id,
                    "rule_text": r.text,
                    "effects": [
                        {"description": e.description, "trigger": e.trigger, "value": e.value}
                        for e in r.effects
                        if "point" in e.action
                    ],
                }
                for r in game.rules["mutable"]
                if hasattr(r, "effects") and any("point" in e.action for e in r.effects)
            ],
            "failed_proposals": [
                {
                    "rule_text": p.rule_text,
                    "player_id": p.player_id,
                    "turn": getattr(p, 'turn_number', getattr(p, 'turn', 'Unknown')),
                    "votes_summary": f"{sum(p.votes.values())}/{len(p.votes)} Aye" if hasattr(p, 'votes') and p.votes else "No votes recorded"
                }
                for p in game.proposals
                if hasattr(p, 'votes') and p.votes and sum(p.votes.values()) / len(p.votes) < (game.current_voting_threshold / 100)
            ][-10:],  # Show last 10 failed proposals
            "winner": game.winner.name if game.winner else None,
        }
    )


@app.route("/stats")
def stats_page():
    """Beautiful HTML statistics page"""
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Model Performance Statistics</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
            margin: 0;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-bottom: 20px;
        }
        .triple-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card { 
            background: rgba(255,255,255,0.95); 
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .card h2 { 
            color: #2c3e50; 
            margin: 0 0 20px 0; 
            font-size: 1.4em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .card h3 { 
            color: #34495e; 
            margin: 0 0 15px 0; 
            font-size: 1.2em;
        }
        .model-card {
            background: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #3498db;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .model-card.elite { border-left-color: #f1c40f; background: linear-gradient(90deg, #fff9e6, #ffffff); }
        .model-card.competent { border-left-color: #2ecc71; background: linear-gradient(90deg, #e8f8f5, #ffffff); }
        .model-card.problematic { border-left-color: #e67e22; background: linear-gradient(90deg, #fdf2e9, #ffffff); }
        .model-card.elimination { border-left-color: #e74c3c; background: linear-gradient(90deg, #fdedec, #ffffff); }
        .current-player { border-left-color: #9b59b6; background: linear-gradient(90deg, #f4ecf7, #ffffff); }
        .model-name {
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            padding: 3px 0;
            border-bottom: 1px solid #ecf0f1;
            font-size: 0.9em;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #7f8c8d; }
        .metric-value { font-weight: 600; color: #2c3e50; }
        .tier-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .tier-elite { background: #f1c40f; color: #8b6508; }
        .tier-competent { background: #2ecc71; color: #ffffff; }
        .tier-problematic { background: #e67e22; color: #ffffff; }
        .tier-elimination { background: #e74c3c; color: #ffffff; }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .summary-item {
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #3498db;
        }
        .summary-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }
        .summary-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0 0 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            transform: scale(1.05);
        }
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 30px;
            font-size: 0.9em;
        }
        @media (max-width: 1200px) {
            .stats-grid, .triple-grid { grid-template-columns: 1fr; }
        }
    </style>
    <script>
        function refreshStats() {
            location.reload();
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('main-content').innerHTML = 
                            '<div class="card"><h2> No Game Running</h2><p>Start a game to view statistics.</p></div>';
                        return;
                    }
                    
                    updateSummaryStats(data.summary);
                    updateCurrentGame(data.current_game);
                    updateCrossGameRankings(data.cross_game_rankings);
                    updateModelTiers(data.model_tiers);
                });
        }
        
        function updateSummaryStats(summary) {
            document.getElementById('total-models').textContent = summary.total_models_tracked;
            document.getElementById('games-completed').textContent = summary.games_completed;
            document.getElementById('elite-count').textContent = summary.elite_count;
            document.getElementById('elimination-count').textContent = summary.elimination_candidates;
            document.getElementById('best-model').textContent = summary.best_model || 'None';
            document.getElementById('worst-model').textContent = summary.worst_model || 'None';
        }
        
        function updateCurrentGame(current_game) {
            const html = current_game.map(player => `
                <div class="model-card current-player">
                    <div class="model-name">${player.player_name} (${player.model})</div>
                    <div class="metric-row">
                        <span class="metric-label">Current Points:</span>
                        <span class="metric-value">${player.current_points}/100</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${player.current_points}%"></div>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Overall Score:</span>
                        <span class="metric-value">${player.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate:</span>
                        <span class="metric-value">${player.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Coherence:</span>
                        <span class="metric-value">${player.coherence.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Games:</span>
                        <span class="metric-value">${player.total_games}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById('current-game').innerHTML = html;
        }
        
        function updateCrossGameRankings(rankings) {
            const html = rankings.slice(0, 10).map((model, index) => `
                <div class="model-card">
                    <div class="model-name">#${index + 1} ${model.model}</div>
                    <div class="metric-row">
                        <span class="metric-label">Overall Score:</span>
                        <span class="metric-value">${model.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Games Played:</span>
                        <span class="metric-value">${model.games_played}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate:</span>
                        <span class="metric-value">${model.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Coherence:</span>
                        <span class="metric-value">${model.coherence.toFixed(1)}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById('cross-game-rankings').innerHTML = html;
        }
        
        function updateModelTiers(tiers) {
            updateTier('elite', tiers.elite, 'tier-elite');
            updateTier('competent', tiers.competent, 'tier-competent');
            updateTier('problematic', tiers.problematic, 'tier-problematic');
            updateTier('elimination', tiers.elimination_candidates, 'tier-elimination');
        }
        
        function updateTier(tierId, models, badgeClass) {
            if (models.length === 0) {
                document.getElementById(tierId).innerHTML = '<p style="color: #7f8c8d; font-style: italic;">No models in this tier</p>';
                return;
            }
            
            const html = models.map(model => `
                <div class="model-card ${tierId}">
                    <div class="model-name">
                        ${model.model}
                        <span class="tier-badge ${badgeClass}">${tierId}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Score:</span>
                        <span class="metric-value">${model.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Games:</span>
                        <span class="metric-value">${model.games_played}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Errors:</span>
                        <span class="metric-value">${model.error_count}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById(tierId).innerHTML = html;
        }
        
        // Auto-refresh every 10 seconds
        setInterval(loadStats, 10000);
        
        // Load stats when page loads
        window.onload = loadStats;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Nomic Model Performance Analytics</h1>
            <p>Comprehensive AI model evaluation and competitive analysis</p>
            <button class="refresh-btn" onclick="refreshStats()"> Refresh Data</button>
        </div>
        
        <div id="main-content">
            <div class="summary-stats">
                <div class="summary-item">
                    <div class="summary-number" id="total-models">-</div>
                    <div class="summary-label">Models Tracked</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="games-completed">-</div>
                    <div class="summary-label">Games Completed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="elite-count">-</div>
                    <div class="summary-label">Elite Models</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="elimination-count">-</div>
                    <div class="summary-label">At Risk</div>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="card">
                    <h2> Current Game Players</h2>
                    <div id="current-game">Loading...</div>
                </div>
                
                <div class="card">
                    <h2> Cross-Game Rankings</h2>
                    <div id="cross-game-rankings">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h2> Performance Leaders</h2>
                <div class="stats-grid">
                    <div>
                        <strong> Best Model:</strong> <span id="best-model">-</span>
                    </div>
                    <div>
                        <strong> Worst Model:</strong> <span id="worst-model">-</span>
                    </div>
                </div>
            </div>
            
            <div class="triple-grid">
                <div class="card">
                    <h3> Elite Tier (75+ Score)</h3>
                    <div id="elite">Loading...</div>
                </div>
                
                <div class="card">
                    <h3> Competent Tier (50-74 Score)</h3>
                    <div id="competent">Loading...</div>
                </div>
                
                <div class="card">
                    <h3> Problematic Tier (25-49 Score)</h3>
                    <div id="problematic">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h3> Elimination Candidates (<25 Score)</h3>
                <div id="elimination">Loading...</div>
            </div>
        </div>
        
        <div class="footer">
            <p> Automated model performance tracking and competitive analysis for Nomic AI gameplay</p>
        </div>
    </div>
</body>
</html>
    """
    )


@app.route("/api/stats")
def get_statistics_api():
    """API endpoint for statistics data"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    # Get current game model assignments and performance
    current_game_stats = []
    for player in game.players:
        if player.assigned_model_metrics:
            metrics = player.assigned_model_metrics
            current_game_stats.append(
                {
                    "player_name": player.name,
                    "model": player.model,
                    "current_points": player.points,
                    "overall_score": metrics.calculate_overall_score(),
                    "success_rate": metrics.calculate_success_rate(),
                    "coherence": metrics.calculate_average_coherence(),
                    "engagement": metrics.calculate_strategic_engagement(),
                    "tie_obsession": metrics.tie_related_proposals,
                    "total_games": metrics.total_games,
                    "games_won": metrics.games_won,
                }
            )

    # Get cross-game model statistics
    model_stats = game.performance_manager.get_model_statistics()

    # Sort models by overall score
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["overall_score"], reverse=True)

    # Create tier classifications
    model_tiers = {"elite": [], "competent": [], "problematic": [], "elimination_candidates": []}

    for model, stats in sorted_models:
        score = stats["overall_score"]
        if score >= 75:
            model_tiers["elite"].append({"model": model, **stats})
        elif score >= 50:
            model_tiers["competent"].append({"model": model, **stats})
        elif score >= 25:
            model_tiers["problematic"].append({"model": model, **stats})
        else:
            model_tiers["elimination_candidates"].append({"model": model, **stats})

    return jsonify(
        {
            "current_game": current_game_stats,
            "cross_game_rankings": [{"model": model, **stats} for model, stats in sorted_models],
            "model_tiers": model_tiers,
            "summary": {
                "total_models_tracked": len(model_stats),
                "games_completed": max([stats["games_played"] for stats in model_stats.values()] + [0]),
                "best_model": sorted_models[0][0] if sorted_models else None,
                "worst_model": sorted_models[-1][0] if sorted_models else None,
                "elite_count": len(model_tiers["elite"]),
                "elimination_candidates": len(model_tiers["elimination_candidates"]),
            },
        }
    )


@app.route("/player-states")
def player_states_page():
    """Beautiful HTML page showing player internal states"""
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Player Internal States</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
        }
        .header {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .players-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); 
            gap: 20px; 
        }
        .player-card { 
            background: rgba(255,255,255,0.1); 
            padding: 25px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .player-card:hover {
            transform: translateY(-5px);
        }
        .player-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
        }
        .player-name {
            font-size: 1.4em;
            font-weight: bold;
        }
        .player-role {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .player-points {
            font-size: 1.2em;
            font-weight: bold;
            color: #f39c12;
        }
        .state-section {
            margin: 15px 0;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }
        .state-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #3498db;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .state-content {
            line-height: 1.6;
        }
        .threat-list, .alliance-list {
            list-style: none;
            padding: 0;
        }
        .threat-item, .alliance-item {
            background: rgba(255,255,255,0.1);
            margin: 5px 0;
            padding: 8px 12px;
            border-radius: 5px;
            border-left: 3px solid #e74c3c;
        }
        .alliance-item {
            border-left-color: #2ecc71;
        }
        .planned-proposals {
            max-height: 150px;
            overflow-y: auto;
        }
        .proposal-item {
            background: rgba(255,255,255,0.1);
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #f39c12;
        }
        .learning-observations {
            max-height: 120px;
            overflow-y: auto;
        }
        .observation-item {
            background: rgba(255,255,255,0.1);
            margin: 5px 0;
            padding: 8px 10px;
            border-radius: 5px;
            border-left: 3px solid #9b59b6;
            font-size: 0.9em;
        }
        .scratchpad-notes {
            max-height: 180px;
            overflow-y: auto;
        }
        .scratchpad-item {
            background: rgba(255,255,255,0.1);
            margin: 5px 0;
            padding: 8px 10px;
            border-radius: 5px;
            border-left: 3px solid #1abc9c;
            font-size: 0.9em;
            font-family: 'Courier New', monospace;
        }
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .refresh-btn:hover {
            transform: scale(1.05);
        }
        .idle-turns {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(155, 89, 182, 0.2);
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .last-updated {
            font-size: 0.8em;
            opacity: 0.7;
            text-align: right;
            margin-top: 15px;
        }
        @media (max-width: 768px) {
            .players-grid { grid-template-columns: 1fr; }
        }
    </style>
    <script>
        function refreshStates() {
            location.reload();
        }
        
        function loadPlayerStates() {
            fetch('/api/player-states')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('main-content').innerHTML = 
                            '<div class="player-card"><h2> No Game Running</h2><p>Start a game to view player states.</p></div>';
                        return;
                    }
                    
                    updatePlayerStates(data.player_states);
                    document.getElementById('game-info').innerHTML = `
                        <strong>Session:</strong> ${data.session_id} | 
                        <strong>Turn:</strong> ${data.current_turn} | 
                        <strong>Current Player:</strong> ${data.current_player}
                    `;
                });
        }
        
        function updatePlayerStates(playerStates) {
            const html = Object.values(playerStates).map(player => `
                <div class="player-card">
                    <div class="player-header">
                        <div>
                            <div class="player-name">${player.name} (${player.model})</div>
                            <div class="player-role">${player.role}</div>
                        </div>
                        <div class="player-points">${player.points}/100 pts</div>
                    </div>
                    
                    <div class="idle-turns">
                        <span>Idle Turns Used: ${player.state.idle_turn_count}/3</span>
                        <span style="color: ${player.state.idle_turn_count < 3 ? '#2ecc71' : '#e74c3c'}">
                            ${player.state.idle_turn_count < 3 ? 'Available' : 'Exhausted'}
                        </span>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Strategic Focus</div>
                        <div class="state-content">${player.state.strategic_focus || 'Not set'}</div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Victory Path</div>
                        <div class="state-content">${player.state.victory_path || 'Not defined'}</div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Threat Assessments</div>
                        <div class="state-content">
                            ${Object.entries(player.state.threat_assessments).length > 0 ? 
                                '<ul class="threat-list">' +
                                Object.entries(player.state.threat_assessments).map(([pid, threat]) => 
                                    `<li class="threat-item">Player ${pid}: ${threat}</li>`
                                ).join('') + '</ul>' :
                                '<em>No threats identified</em>'
                            }
                        </div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Alliance Considerations</div>
                        <div class="state-content">
                            ${Object.entries(player.state.alliance_considerations).length > 0 ? 
                                '<ul class="alliance-list">' +
                                Object.entries(player.state.alliance_considerations).map(([pid, alliance]) => 
                                    `<li class="alliance-item">Player ${pid}: ${alliance}</li>`
                                ).join('') + '</ul>' :
                                '<em>No alliances considered</em>'
                            }
                        </div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Planned Proposals</div>
                        <div class="planned-proposals">
                            ${player.state.planned_proposals.length > 0 ? 
                                player.state.planned_proposals.map(proposal => 
                                    `<div class="proposal-item">${proposal}</div>`
                                ).join('') :
                                '<em>No proposals planned</em>'
                            }
                        </div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Learning Observations</div>
                        <div class="learning-observations">
                            ${player.state.learning_observations.length > 0 ? 
                                player.state.learning_observations.slice(-5).map(obs => 
                                    `<div class="observation-item">${obs}</div>`
                                ).join('') :
                                '<em>No observations recorded</em>'
                            }
                        </div>
                    </div>
                    
                    <div class="state-section">
                        <div class="state-title"> Scratchpad Notes</div>
                        <div class="scratchpad-notes">
                            ${player.state.scratchpad && player.state.scratchpad.length > 0 ? 
                                player.state.scratchpad.slice(-8).map(note => 
                                    `<div class="scratchpad-item">${note}</div>`
                                ).join('') :
                                '<em>No scratchpad notes</em>'
                            }
                        </div>
                    </div>
                    
                    <div class="last-updated">
                        Last updated: ${new Date(player.state.last_updated).toLocaleString()}
                    </div>
                </div>
            `).join('');
            document.getElementById('players-container').innerHTML = html;
        }
        
        // Auto-refresh every 5 seconds
        setInterval(loadPlayerStates, 5000);
        
        // Load states when page loads
        window.onload = loadPlayerStates;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Nomic Player Internal States</h1>
            <p>Real-time strategic thinking and planning of AI players</p>
            <div id="game-info" style="margin-top: 10px; font-size: 0.9em;">Loading...</div>
            <button class="refresh-btn" onclick="refreshStates()"> Refresh States</button>
        </div>
        
        <div id="main-content">
            <div class="players-grid" id="players-container">
                Loading player states...
            </div>
        </div>
    </div>
</body>
</html>
    """
    )


@app.route("/api/player-states")
def get_player_states_api():
    """API endpoint for player internal states"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    # Get player states from state tracker
    player_states = {}
    for player in game.players:
        state = game.state_tracker.get_player_state(player.id)
        if state:
            player_states[player.id] = {
                "id": player.id,
                "name": player.name,
                "role": player.role,
                "model": player.model,
                "points": player.points,
                "state": asdict(state),
            }

    return jsonify(
        {
            "session_id": game.session_id,
            "current_turn": game.turn_number,
            "current_player": game.get_current_player().name if not game.game_over else None,
            "player_states": player_states,
        }
    )


@app.route("/game-logs")
def game_logs_page():
    """Game logs viewer with filtering and search"""
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Game Logs</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .filters {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
        }
        .filter-group label {
            margin-bottom: 5px;
            font-weight: bold;
        }
        .filter-group select, .filter-group input {
            padding: 8px;
            border-radius: 5px;
            border: none;
            background: rgba(255,255,255,0.9);
            color: #333;
        }
        .logs-container {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        .log-entry {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .log-entry.deliberation { border-left-color: #9b59b6; }
        .log-entry.proposal { border-left-color: #e67e22; }
        .log-entry.vote { border-left-color: #2ecc71; }
        .log-entry.idle_turn { border-left-color: #f39c12; }
        .log-entry.scratchpad_update { border-left-color: #1abc9c; }
        .log-entry.failed_proposal { border-left-color: #e74c3c; }
        .log-entry.deliberation_choice { border-left-color: #34495e; }
        .log-entry.insight { border-left-color: #f1c40f; }
        .log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .log-type {
            background: rgba(0,0,0,0.3);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            text-transform: uppercase;
        }
        .log-timestamp {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .log-content {
            line-height: 1.6;
        }
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            margin: 0 10px;
        }
        .export-btn {
            background: linear-gradient(45deg, #e67e22, #f39c12);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
        }
    </style>
    <script>
        let allLogs = [];
        
        function loadLogs() {
            fetch('/api/game-logs')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('logs-container').innerHTML = 
                            '<div class="log-entry"><h3> No Game Running</h3></div>';
                        return;
                    }
                    allLogs = data.logs;
                    displayLogs(allLogs);
                });
        }
        
        function displayLogs(logs) {
            const html = logs.map(turn => {
                return turn.events.map(event => {
                    let content = '';
                    switch(event.type) {
                        case 'turn_start':
                            content = `<strong>${event.player_name}</strong> starts turn ${event.turn_number} (${event.player_model})`;
                            break;
                        case 'deliberation':
                            content = `Deliberation step ${event.deliberation_step} - Response: ${event.response.substring(0, 200)}...`;
                            break;
                        case 'proposal':
                            content = `
                                <strong>Proposal:</strong> "${event.rule_text}"<br>
                                <strong>Explanation:</strong> ${event.explanation}<br>
                                <strong>Internal Thoughts:</strong> ${event.internal_thoughts}
                            `;
                            break;
                        case 'vote':
                            content = `Voted ${event.vote ? 'AYE' : 'NAY'} - ${event.reasoning}`;
                            break;
                        case 'idle_turn':
                            content = `
                                <strong>Idle Turn Analysis:</strong><br>
                                Strategic: ${event.strategic_analysis.substring(0, 100)}...<br>
                                Insights: ${event.insights.substring(0, 100)}...
                            `;
                            break;
                        case 'scratchpad_update':
                            content = `
                                <strong>Scratchpad Update:</strong><br>
                                Added ${event.new_notes.length} notes (Total: ${event.total_notes})<br>
                                <em>${event.new_notes.join(', ')}</em>
                            `;
                            break;
                        case 'failed_proposal':
                            content = `
                                <strong>Failed Proposal:</strong><br>
                                Reason: ${event.error_reason}<br>
                                Retry Count: ${event.retry_count}
                            `;
                            break;
                        case 'deliberation_choice':
                            content = `
                                <strong>Deliberation Choice:</strong><br>
                                Chose ${event.chosen_turns} turns<br>
                                Reasoning: ${event.reasoning.substring(0, 150)}...
                            `;
                            break;
                        case 'insight':
                            content = `
                                <strong>Player Insight:</strong><br>
                                ${event.insight}
                            `;
                            break;
                        default:
                            content = JSON.stringify(event);
                    }
                    
                    return `
                        <div class="log-entry ${event.type}">
                            <div class="log-header">
                                <span class="log-type">${event.type}</span>
                                <span class="log-timestamp">${new Date(event.timestamp).toLocaleString()}</span>
                            </div>
                            <div class="log-content">${content}</div>
                        </div>
                    `;
                }).join('');
            }).join('');
            
            document.getElementById('logs-container').innerHTML = html;
        }
        
        function filterLogs() {
            const turnFilter = document.getElementById('turn-filter').value;
            const typeFilter = document.getElementById('type-filter').value;
            const playerFilter = document.getElementById('player-filter').value;
            
            let filtered = allLogs;
            
            if (turnFilter) {
                filtered = filtered.filter(turn => turn.turn_number == turnFilter);
            }
            
            if (typeFilter || playerFilter) {
                filtered = filtered.map(turn => ({
                    ...turn,
                    events: turn.events.filter(event => {
                        let matches = true;
                        if (typeFilter && event.type !== typeFilter) matches = false;
                        if (playerFilter && event.player_id != playerFilter) matches = false;
                        return matches;
                    })
                })).filter(turn => turn.events.length > 0);
            }
            
            displayLogs(filtered);
        }
        
        function exportLogs() {
            const dataStr = JSON.stringify(allLogs, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `nomic_logs_${new Date().toISOString().split('T')[0]}.json`;
            link.click();
        }
        
        setInterval(loadLogs, 10000);
        window.onload = loadLogs;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Nomic Game Logs</h1>
            <p>Comprehensive turn-by-turn game event logging</p>
            <button class="refresh-btn" onclick="loadLogs()"> Refresh</button>
            <button class="export-btn" onclick="exportLogs()"> Export JSON</button>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>Turn Number:</label>
                <select id="turn-filter" onchange="filterLogs()">
                    <option value="">All Turns</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Event Type:</label>
                <select id="type-filter" onchange="filterLogs()">
                    <option value="">All Types</option>
                    <option value="turn_start">Turn Start</option>
                    <option value="deliberation">Deliberation</option>
                    <option value="proposal">Proposal</option>
                    <option value="vote">Vote</option>
                    <option value="idle_turn">Idle Turn</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Player:</label>
                <select id="player-filter" onchange="filterLogs()">
                    <option value="">All Players</option>
                </select>
            </div>
        </div>
        
        <div class="logs-container" id="logs-container">
            Loading logs...
        </div>
    </div>
</body>
</html>
    """
    )


@app.route("/api/game-logs")
def get_game_logs_api():
    """API endpoint for game logs"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    # Load logs from session manager
    try:
        session_path = os.path.join(game.session_manager.sessions_dir, game.session_id)
        logs_path = os.path.join(session_path, "turn_logs.json")

        if os.path.exists(logs_path):
            with open(logs_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        return jsonify({"session_id": game.session_id, "logs": logs})
    except Exception as e:
        return jsonify({"error": f"Could not load logs: {str(e)}"})


@app.route("/sessions")
def sessions_page():
    """Session history page"""
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Session History</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .sessions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }
        .session-card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .session-card:hover {
            transform: translateY(-5px);
        }
        .session-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
        }
        .session-id {
            font-weight: bold;
            font-size: 1.1em;
        }
        .session-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            background: #2ecc71;
        }
        .session-status.in-progress {
            background: #f39c12;
        }
        .session-info {
            margin: 10px 0;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
        }
        .info-label {
            font-weight: bold;
            opacity: 0.8;
        }
        .players-summary {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .winner-highlight {
            background: linear-gradient(45deg, #f1c40f, #f39c12);
            color: #000;
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
    </style>
    <script>
        function loadSessions() {
            fetch('/api/sessions')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('sessions-container').innerHTML = 
                            '<div class="session-card"><h3> No Sessions Found</h3></div>';
                        return;
                    }
                    displaySessions(data.sessions);
                });
        }
        
        function displaySessions(sessions) {
            const html = sessions.map(session => {
                const startTime = new Date(session.start_time);
                const endTime = session.end_time ? new Date(session.end_time) : null;
                const duration = endTime ? 
                    Math.round((endTime - startTime) / (1000 * 60)) + ' minutes' : 
                    'In progress';
                
                const winner = session.winner_id ? 
                    session.players.find(p => p.id === session.winner_id) : null;
                
                return `
                    <div class="session-card">
                        <div class="session-header">
                            <div class="session-id">${session.session_id}</div>
                            <div class="session-status ${!session.end_time ? 'in-progress' : ''}">
                                ${session.end_time ? 'Completed' : 'In Progress'}
                            </div>
                        </div>
                        
                        <div class="session-info">
                            <div class="info-row">
                                <span class="info-label">Started:</span>
                                <span>${startTime.toLocaleString()}</span>
                            </div>
                            ${endTime ? `
                                <div class="info-row">
                                    <span class="info-label">Ended:</span>
                                    <span>${endTime.toLocaleString()}</span>
                                </div>
                            ` : ''}
                            <div class="info-row">
                                <span class="info-label">Duration:</span>
                                <span>${duration}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Total Turns:</span>
                                <span>${session.total_turns}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Rules Created:</span>
                                <span>${session.rules_created}</span>
                            </div>
                        </div>
                        
                        ${winner ? `
                            <div class="winner-highlight">
                                 Winner: ${winner.name} with ${session.final_scores[winner.id]} points
                            </div>
                        ` : ''}
                        
                        <div class="players-summary">
                            <strong>Players:</strong><br>
                            ${session.players.map(p => {
                                const finalScore = session.final_scores[p.id] || p.starting_points;
                                return `${p.name} (${p.model}): ${finalScore} points`;
                            }).join('<br>')}
                        </div>
                    </div>
                `;
            }).join('');
            
            document.getElementById('sessions-container').innerHTML = html;
        }
        
        window.onload = loadSessions;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Nomic Session History</h1>
            <p>Complete archive of all game sessions and outcomes</p>
        </div>
        
        <div class="sessions-grid" id="sessions-container">
            Loading session history...
        </div>
    </div>
</body>
</html>
    """
    )


@app.route("/api/sessions")
def get_sessions_api():
    """API endpoint for session history"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    sessions = game.session_manager.get_session_history()
    return jsonify({"sessions": [asdict(session) for session in sessions]})


@app.route("/costs")
def costs_dashboard():
    """OpenRouter cost tracking dashboard"""
    if not game:
        return "No game running"

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Game - Cost Tracking</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .cost-item { display: flex; justify-content: space-between; padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 5px; }
        .total-cost { font-size: 24px; font-weight: bold; color: #e74c3c; text-align: center; padding: 20px; background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; }
        .token-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat { text-align: center; padding: 15px; background: #e3f2fd; border-radius: 8px; }
        .provider-info { background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        h1, h2 { color: #333; }
        .refresh-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px 0; }
        .refresh-btn:hover { background: #0056b3; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1> Cost Tracking Dashboard</h1>
            <button class="refresh-btn" onclick="updateCosts()"> Refresh Costs</button>
            <a href="/" style="margin-left: 20px;"> Back to Game</a>
        </div>
        
        <div class="card">
            <h2> Provider Information</h2>
            <div class="provider-info" id="provider-info">Loading...</div>
        </div>
        
        <div class="card">
            <h2> Total Session Cost</h2>
            <div class="total-cost" id="total-cost">$0.0000</div>
        </div>
        
        <div class="card">
            <h2> Token Usage</h2>
            <div class="token-stats">
                <div class="stat">
                    <h3>Input Tokens</h3>
                    <div id="input-tokens">0</div>
                </div>
                <div class="stat">
                    <h3>Output Tokens</h3>
                    <div id="output-tokens">0</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2> Cost by Model</h2>
            <div id="model-costs">No data available</div>
        </div>
        
        <div class="card">
            <h2> Cost Breakdown Chart</h2>
            <canvas id="costChart" width="400" height="200"></canvas>
        </div>
    </div>
    
    <script>
        let costChart;
        
        function updateCosts() {
            fetch('/api/costs')
                .then(r => r.json())
                .then(data => {
                    // Update provider info
                    const providerTypes = {
                        'openrouter': ' OpenRouter API',
                        'ollama_new': ' Local Ollama (New)',
                        'ollama_legacy': ' Local Ollama (Legacy)'
                    };
                    document.getElementById('provider-info').innerHTML = 
                        providerTypes[data.provider_type] || data.provider_type;
                    
                    // Update total cost
                    document.getElementById('total-cost').textContent = `$${data.total_cost}`;
                    
                    // Update token counts
                    document.getElementById('input-tokens').textContent = data.total_tokens.input.toLocaleString();
                    document.getElementById('output-tokens').textContent = data.total_tokens.output.toLocaleString();
                    
                    // Update model costs
                    let modelCostsHtml = '';
                    if (Object.keys(data.costs).length === 0) {
                        modelCostsHtml = '<div style="text-align: center; color: #666;">No costs tracked yet</div>';
                    } else {
                        for (const [model, costs] of Object.entries(data.costs)) {
                            modelCostsHtml += `
                                <div class="cost-item">
                                    <span><strong>${model}</strong></span>
                                    <span>$${costs.total_cost.toFixed(4)} (${costs.input_tokens + costs.output_tokens} tokens)</span>
                                </div>
                            `;
                        }
                    }
                    document.getElementById('model-costs').innerHTML = modelCostsHtml;
                    
                    // Update chart
                    updateCostChart(data.costs);
                });
        }
        
        function updateCostChart(costsData) {
            const ctx = document.getElementById('costChart').getContext('2d');
            
            if (costChart) {
                costChart.destroy();
            }
            
            const models = Object.keys(costsData);
            const costs = Object.values(costsData).map(c => c.total_cost);
            
            if (models.length === 0) {
                return;
            }
            
            costChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: models,
                    datasets: [{
                        data: costs,
                        backgroundColor: ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        },
                        title: {
                            display: true,
                            text: 'Cost Distribution by Model'
                        }
                    }
                }
            });
        }
        
        // Update costs every 10 seconds
        updateCosts();
        setInterval(updateCosts, 10000);
    </script>
</body>
</html>
    """
    )


@app.route("/analytics")
def analytics():
    """Comprehensive model performance analytics dashboard"""
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Model Performance Analytics</title>
    <style>
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .analytics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }
        .analytics-card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .chart-container {
            width: 100%;
            height: 300px;
            margin: 15px 0;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .metric-label {
            font-weight: bold;
            color: #555;
        }
        .metric-value {
            color: #2c3e50;
            font-weight: 600;
        }
        .model-comparison {
            grid-column: 1 / -1;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .comparison-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        .comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .performance-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge-excellent { background: #2ecc71; color: white; }
        .badge-good { background: #f39c12; color: white; }
        .badge-average { background: #95a5a6; color: white; }
        .badge-poor { background: #e74c3c; color: white; }
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin: 20px auto;
            display: block;
            transition: transform 0.2s;
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
        }
        .strategy-insights {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            color: white;
        }
        .insight-item {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #fdcb6e;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Model Performance Analytics</h1>
            <p>Comprehensive analysis of AI model behavior in Nomic gameplay</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshAnalytics()"> Refresh Analytics</button>
        
        <div class="analytics-grid">
            <div class="analytics-card">
                <h2> Overall Performance Metrics</h2>
                <div id="overall-metrics">Loading...</div>
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
            
            <div class="analytics-card">
                <h2> Strategic Behavior Analysis</h2>
                <div id="strategic-analysis">Loading...</div>
                <div class="chart-container">
                    <canvas id="strategy-chart"></canvas>
                </div>
            </div>
            
            <div class="analytics-card model-comparison">
                <h2> Model Comparison Matrix</h2>
                <div id="model-comparison">Loading...</div>
            </div>
            
            <div class="analytics-card strategy-insights">
                <h2> Strategic Insights & Patterns</h2>
                <div id="strategy-insights">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let performanceChart, strategyChart;
        
        function initCharts() {
            // Performance Chart
            const perfCtx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(perfCtx, {
                type: 'radar',
                data: {
                    labels: ['Success Rate', 'Coherence', 'Strategic Depth', 'Adaptability', 'Consistency'],
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Strategy Chart
            const stratCtx = document.getElementById('strategy-chart').getContext('2d');
            strategyChart = new Chart(stratCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Aggressive', 'Defensive', 'Cooperative', 'Analytical'],
                    datasets: [{
                        data: [25, 25, 25, 25],
                        backgroundColor: ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        function refreshAnalytics() {
            fetch('/api/analytics')
                .then(r => r.json())
                .then(data => {
                    updateOverallMetrics(data.overall);
                    updateStrategicAnalysis(data.strategic);
                    updateModelComparison(data.comparison);
                    updateStrategyInsights(data.insights);
                    updateCharts(data);
                })
                .catch(e => console.error('Analytics error:', e));
        }
        
        function updateOverallMetrics(data) {
            const html = `
                <div class="metric-row">
                    <span class="metric-label">Total Games Played:</span>
                    <span class="metric-value">${data.total_games}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Average Game Length:</span>
                    <span class="metric-value">${data.avg_game_length} turns</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Most Successful Model:</span>
                    <span class="metric-value">${data.top_model}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Average Proposal Success:</span>
                    <span class="metric-value">${data.avg_success_rate}%</span>
                </div>
            `;
            document.getElementById('overall-metrics').innerHTML = html;
        }
        
        function updateStrategicAnalysis(data) {
            const html = `
                <div class="metric-row">
                    <span class="metric-label">Most Common Strategy:</span>
                    <span class="metric-value">${data.dominant_strategy}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Strategic Depth:</span>
                    <span class="metric-value">${data.avg_strategic_depth}/100</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Coalition Formation:</span>
                    <span class="metric-value">${data.coalition_rate}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Innovation Index:</span>
                    <span class="metric-value">${data.innovation_score}/100</span>
                </div>
            `;
            document.getElementById('strategic-analysis').innerHTML = html;
        }
        
        function updateModelComparison(models) {
            let html = `
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Games</th>
                            <th>Win Rate</th>
                            <th>Success Rate</th>
                            <th>Coherence</th>
                            <th>Strategic Score</th>
                            <th>Classification</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            models.forEach(model => {
                const badgeClass = model.score >= 80 ? 'excellent' : 
                                 model.score >= 60 ? 'good' : 
                                 model.score >= 40 ? 'average' : 'poor';
                
                html += `
                    <tr>
                        <td><strong>${model.name}</strong></td>
                        <td>${model.games}</td>
                        <td>${model.win_rate}%</td>
                        <td>${model.success_rate}%</td>
                        <td>${model.coherence}/100</td>
                        <td>${model.strategic_score}/100</td>
                        <td><span class="performance-badge badge-${badgeClass}">${model.classification}</span></td>
                    </tr>
                `;
            });
            
            html += '</tbody></table>';
            document.getElementById('model-comparison').innerHTML = html;
        }
        
        function updateStrategyInsights(insights) {
            let html = '';
            insights.forEach(insight => {
                html += `<div class="insight-item">${insight}</div>`;
            });
            document.getElementById('strategy-insights').innerHTML = html;
        }
        
        function updateCharts(data) {
            // Update performance radar chart
            performanceChart.data.datasets = data.chart_data.performance;
            performanceChart.update();
            
            // Update strategy distribution chart
            strategyChart.data.datasets[0].data = data.chart_data.strategy_distribution;
            strategyChart.update();
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            refreshAnalytics();
            setInterval(refreshAnalytics, 30000); // Refresh every 30 seconds
        });
    </script>
</body>
</html>
    """
    )


@app.route("/gm-rules")
def gm_rules_page():
    """Display Game Master interpreted rules"""
    if not game:
        return "No game running"
    
    # Get GM interpretations and execution stats
    gm_interpretations = []
    if hasattr(game, 'game_master') and game.game_master.interpretations:
        for rule_id, interpretation in game.game_master.interpretations.items():
            stats = game.dynamic_rule_engine.get_statistics().get(rule_id, {})
            gm_interpretations.append({
                'rule_id': rule_id,
                'rule_text': interpretation.rule_text,
                'complexity': interpretation.complexity_score,
                'requires_linguistic': interpretation.requires_linguistic,
                'triggers': interpretation.triggers,
                'execution_count': stats.get('execution_count', 0),
                'error_count': stats.get('error_count', 0)
            })
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Game Master Rules - Nomic</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        h1 { color: #333; text-align: center; }
        .rule-card {
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .rule-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .rule-id { font-weight: bold; color: #007bff; }
        .complexity {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .complexity-low { background: #d4edda; color: #155724; }
        .complexity-medium { background: #fff3cd; color: #856404; }
        .complexity-high { background: #f8d7da; color: #721c24; }
        .rule-text { font-style: italic; margin: 10px 0; }
        .triggers { margin: 10px 0; }
        .trigger-badge {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            margin-right: 5px;
            font-size: 0.85em;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.9em;
        }
        .stat-item { color: #666; }
        .linguistic-flag {
            background: #6f42c1;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }
        .no-rules {
            text-align: center;
            color: #666;
            padding: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Game Master Rules</h1>
        
        {% if gm_interpretations %}
            {% for rule in gm_interpretations %}
            <div class="rule-card">
                <div class="rule-header">
                    <span class="rule-id">Rule {{ rule.rule_id }}</span>
                    <div>
                        {% if rule.requires_linguistic %}
                        <span class="linguistic-flag">Linguistic Analysis</span>
                        {% endif %}
                        <span class="complexity 
                            {% if rule.complexity < 0.3 %}complexity-low{% elif rule.complexity < 0.6 %}complexity-medium{% else %}complexity-high{% endif %}">
                            Complexity: {{ "%.2f"|format(rule.complexity) }}
                        </span>
                    </div>
                </div>
                
                <div class="rule-text">"{{ rule.rule_text }}"</div>
                
                <div class="triggers">
                    <strong>Triggers:</strong>
                    {% for trigger in rule.triggers %}
                    <span class="trigger-badge">{{ trigger }}</span>
                    {% endfor %}
                </div>
                
                <div class="stats">
                    <span class="stat-item"> Executed: {{ rule.execution_count }} times</span>
                    <span class="stat-item"> Errors: {{ rule.error_count }}</span>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <div class="no-rules">
                <h2>No GM-interpreted rules yet</h2>
                <p>Complex rules requiring Game Master interpretation will appear here.</p>
            </div>
        {% endif %}
        
        <div style="margin-top: 30px; text-align: center;">
            <a href="/" style="text-decoration: none; color: #007bff;"> Back to Game</a>
        </div>
    </div>
</body>
</html>
    """, gm_interpretations=gm_interpretations)


@app.route("/api/analytics")
def get_analytics_api():
    """API endpoint for comprehensive analytics data"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game"})

    # Get performance manager data
    performance_manager = game.performance_manager
    model_stats = performance_manager.get_model_statistics()

    # Calculate overall metrics
    total_games = sum(stats["games_played"] for stats in model_stats.values())
    avg_success_rate = (
        sum(stats["success_rate"] for stats in model_stats.values()) / len(model_stats) if model_stats else 0
    )

    # Find top performing model
    top_model = max(model_stats.items(), key=lambda x: x[1]["overall_score"])[0] if model_stats else "N/A"

    # Strategic analysis - with defensive programming for missing keys
    strategic_scores = [stats.get("strategic_engagement", 50) for stats in model_stats.values()]
    avg_strategic_depth = sum(strategic_scores) / len(strategic_scores) if strategic_scores else 0

    # Model comparison data
    comparison_data = []
    for model_name, stats in model_stats.items():
        comparison_data.append(
            {
                "name": model_name,
                "games": stats.get("games_played", 0),
                "win_rate": round(stats.get("win_rate", 0), 1),
                "success_rate": round(stats.get("success_rate", 0), 1),
                "coherence": round(stats.get("coherence", 0), 1),
                "strategic_score": round(stats.get("strategic_engagement", 50), 1),
                "score": round(stats.get("overall_score", 0), 1),
                "classification": classify_model_performance(stats.get("overall_score", 0)),
            }
        )

    # Sort by overall score
    comparison_data.sort(key=lambda x: x["score"], reverse=True)

    # Generate insights
    insights = generate_strategic_insights(model_stats, game)

    # Get real analytics data from session files
    sessions_data = aggregate_session_data()
    real_metrics = calculate_real_analytics_metrics(sessions_data)

    # Chart data
    chart_data = {
        "performance": generate_performance_chart_data(model_stats),
        "strategy_distribution": real_metrics["strategy_distribution"],
    }

    return jsonify(
        {
            "overall": {
                "total_games": total_games,
                "avg_game_length": real_metrics["avg_game_length"],
                "top_model": top_model,
                "avg_success_rate": round(avg_success_rate, 1),
            },
            "strategic": {
                "dominant_strategy": real_metrics["dominant_strategy"],
                "avg_strategic_depth": round(avg_strategic_depth, 1),
                "coalition_rate": real_metrics["coalition_rate"],
                "innovation_score": real_metrics["innovation_score"],
            },
            "comparison": comparison_data,
            "insights": insights,
            "chart_data": chart_data,
        }
    )


@app.route("/api/enhanced-logs")
def get_enhanced_logs_api():
    """API endpoint for enhanced logging data including scratchpad and deliberation details"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game running"})
    
    try:
        session_path = os.path.join(game.session_manager.sessions_dir, game.session_id)
        
        # Load turn logs with enhanced event types
        turn_logs_path = os.path.join(session_path, "turn_logs.json")
        turn_logs = []
        if os.path.exists(turn_logs_path):
            with open(turn_logs_path, "r") as f:
                turn_logs = json.load(f)
        
        # Load player states with scratchpad data
        player_states_path = os.path.join(session_path, "player_states.json") 
        player_states = {}
        if os.path.exists(player_states_path):
            with open(player_states_path, "r") as f:
                player_states_data = json.load(f)
                player_states = player_states_data.get("states", {})
        
        # Load idle turns
        idle_turns_path = os.path.join(session_path, "idle_turns.json")
        idle_turns = []
        if os.path.exists(idle_turns_path):
            with open(idle_turns_path, "r") as f:
                idle_turns = json.load(f)
        
        # Extract enhanced event statistics
        event_stats = {
            "total_events": 0,
            "scratchpad_updates": 0,
            "failed_proposals": 0,
            "deliberation_choices": 0,
            "insights": 0,
            "deliberations": 0,
            "proposals": 0,
            "votes": 0,
            "idle_turns": len(idle_turns)
        }
        
        # Count events by type
        for turn_log in turn_logs:
            for event in turn_log.get("events", []):
                event_stats["total_events"] += 1
                event_type = event.get("type", "")
                if event_type in event_stats:
                    event_stats[event_type] += 1
        
        # Extract scratchpad summary
        scratchpad_summary = {}
        for player_id, state in player_states.items():
            scratchpad_notes = state.get("scratchpad", [])
            scratchpad_summary[player_id] = {
                "total_notes": len(scratchpad_notes),
                "recent_notes": scratchpad_notes[-5:] if scratchpad_notes else []  # Last 5 notes
            }
        
        return jsonify({
            "session_id": game.session_id,
            "current_turn": game.turn_number,
            "enhanced_features": {
                "temporal_context": True,
                "scratchpad_memory": True,
                "flexible_deliberation": True,
                "no_strategic_fallbacks": True
            },
            "event_statistics": event_stats,
            "scratchpad_summary": scratchpad_summary,
            "detailed_logs": {
                "turn_logs": turn_logs,
                "player_states": player_states,
                "idle_turns": idle_turns
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Could not load enhanced logs: {str(e)}"})


@app.route("/api/costs")
def get_costs_api():
    """API endpoint for OpenRouter cost tracking"""
    if not game:
        return jsonify({"error": "No game running"})

    cost_data = {
        "provider_type": game.provider_type,
        "costs": {},
        "total_cost": 0,
        "total_tokens": {"input": 0, "output": 0},
    }

    if hasattr(game.llm_provider, "get_session_costs"):
        session_costs = game.llm_provider.get_session_costs()
        cost_data["costs"] = session_costs

        # Calculate totals
        total_cost = 0
        total_input = 0
        total_output = 0

        for model_costs in session_costs.values():
            total_cost += model_costs.get("total_cost", 0)
            total_input += model_costs.get("input_tokens", 0)
            total_output += model_costs.get("output_tokens", 0)

        cost_data["total_cost"] = round(total_cost, 4)
        cost_data["total_tokens"]["input"] = total_input
        cost_data["total_tokens"]["output"] = total_output

    return jsonify(cost_data)


@app.route("/api/context")
def get_context_analytics():
    """API endpoint for context window analytics"""
    if not game:
        return jsonify({"error": "No game running"})

    analytics = game.context_manager.get_usage_analytics()

    # Add current real-time status for each model
    for model in analytics["models"]:
        current_status = game.context_manager.get_context_usage(model)
        analytics["models"][model]["real_time_status"] = current_status

    return jsonify(analytics)


@app.route("/api/hybrid-stats")
def get_hybrid_stats():
    """API endpoint for hybrid model selection statistics"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game running"})
    
    if not game.hybrid_selector:
        return jsonify({"error": "Hybrid mode not enabled"})
    
    stats = game.hybrid_selector.get_escalation_stats()
    
    # Add cost savings calculation
    total_decisions = stats.get("total_decisions", 0)
    escalations = stats.get("escalations", 0)
    small_model_usage = stats.get("small_model_usage", 0)
    
    estimated_savings = 0
    if total_decisions > 0:
        # Rough calculation: assume 15x average cost difference
        normal_cost = total_decisions * 1.0  # Baseline cost unit
        hybrid_cost = (escalations * 1.0) + (small_model_usage * 0.067)  # ~15x difference
        estimated_savings = ((normal_cost - hybrid_cost) / normal_cost) * 100 if normal_cost > 0 else 0
    
    return jsonify({
        "hybrid_enabled": True,
        "escalation_stats": stats,
        "estimated_cost_savings_percent": round(estimated_savings, 1),
        "model_families": game.llm_provider.model_families if hasattr(game.llm_provider, 'model_families') else {}
    })


@app.route("/api/context-degradation")
def get_context_degradation_status():
    """API endpoint for context degradation tracking and intervention history"""
    from flask import current_app
    game = current_app.config.get('game')
    if not game:
        return jsonify({"error": "No game running"})
    
    # Get degradation summary from context manager
    degradation_summary = game.context_manager.get_degradation_summary()
    
    # Get intervention statistics from context restoration manager
    intervention_stats = {}
    if hasattr(game, 'context_restoration_manager'):
        intervention_stats = game.context_restoration_manager.get_intervention_statistics()
    
    # Add per-player detailed status
    detailed_status = {}
    for player in game.players:
        if hasattr(game.context_manager, 'player_degradation_status') and player.id in game.context_manager.player_degradation_status:
            player_status = game.context_manager.player_degradation_status[player.id]
            detailed_status[player.id] = {
                "player_name": getattr(player, 'name', f'Player {player.id}'),
                "model": getattr(player, 'model', 'unknown'),
                "consecutive_high_usage": player_status.get("consecutive_high_usage", 0),
                "total_degradation_events": player_status.get("total_degradation_events", 0),
                "last_intervention": player_status.get("last_intervention"),
                "recent_usage_history": [
                    {
                        "timestamp": h.get("timestamp", "").isoformat() if hasattr(h.get("timestamp", ""), "isoformat") else str(h.get("timestamp", "")),
                        "usage_percentage": h.get("usage_percentage", 0),
                        "is_degraded": h.get("is_degraded", False)
                    }
                    for h in player_status.get("degradation_history", [])[-5:]  # Last 5 events
                ],
                "interventions_this_game": game.context_manager.emergency_resets_per_game.get(player.id, 0)
            }
    
    # Get current context status for all active models
    current_context_status = {}
    for player in game.players:
        if hasattr(player, 'model'):
            current_context_status[player.model] = game.context_manager.get_context_usage(player.model)
    
    response_data = {
        "degradation_summary": degradation_summary,
        "intervention_statistics": intervention_stats,
        "detailed_player_status": detailed_status,
        "current_context_status": current_context_status,
        "degradation_settings": {
            "threshold_percentage": game.context_manager.degradation_threshold * 100,
            "trigger_window_size": game.context_manager.degradation_window,
            "intervention_cooldown_minutes": 30,
            "total_players_tracked": len(game.context_manager.player_degradation_status)
        },
        "system_health": {
            "total_interventions": intervention_stats.get("total_interventions", 0),
            "players_needing_help": len(degradation_summary.get("players_needing_intervention", [])),
            "players_at_risk": len(degradation_summary.get("players_at_risk", [])),
            "system_status": "healthy" if len(degradation_summary.get("players_needing_intervention", [])) == 0 else "monitoring"
        }
    }
    
    return jsonify(response_data)


# Analytics Data Aggregation Functions
# Cache for analytics data to avoid re-reading files constantly
_analytics_cache = {"data": None, "timestamp": 0, "cache_duration": 300}  # 5 minute cache

def aggregate_session_data():
    """Aggregate data from all completed game sessions"""
    global _analytics_cache
    
    # Check cache first
    current_time = time.time()
    if (_analytics_cache["data"] is not None and 
        current_time - _analytics_cache["timestamp"] < _analytics_cache["cache_duration"]):
        return _analytics_cache["data"]
    
    sessions_dir = "game_sessions"
    
    if not os.path.exists(sessions_dir):
        return {"sessions": [], "error": "No sessions directory found"}
    
    sessions_data = []
    
    # Get all session directories
    for session_name in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_name)
        
        if not os.path.isdir(session_path):
            continue
        
        # Load session metadata
        metadata_path = os.path.join(session_path, 'game_metadata.json')
        if not os.path.exists(metadata_path):
            continue
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Only include completed sessions
            if not metadata.get('end_time'):
                continue
                
            # Load turn logs for detailed analysis
            turn_logs_path = os.path.join(session_path, 'turn_logs.json')
            turn_logs = []
            if os.path.exists(turn_logs_path):
                with open(turn_logs_path, 'r') as f:
                    turn_logs = json.load(f)
            
            session_data = {
                "session_id": metadata.get("session_id"),
                "players": metadata.get("players", []),
                "final_scores": metadata.get("final_scores", {}),
                "winner_id": metadata.get("winner_id"),
                "total_turns": metadata.get("total_turns", 0),
                "rules_created": metadata.get("rules_created", 0),
                "turn_logs": turn_logs
            }
            
            sessions_data.append(session_data)
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading session {session_name}: {e}")
            continue
    
    # Cache the result
    result = {"sessions": sessions_data, "total_sessions": len(sessions_data)}
    _analytics_cache["data"] = result
    _analytics_cache["timestamp"] = current_time
    
    return result


def calculate_real_analytics_metrics(sessions_data):
    """Calculate real analytics metrics from session data"""
    if not sessions_data["sessions"]:
        return {
            "avg_game_length": 0,
            "dominant_strategy": "No data",
            "coalition_rate": 0,
            "innovation_score": 0,
            "strategy_distribution": [0, 0, 0, 0]
        }
    
    sessions = sessions_data["sessions"]
    
    # Calculate average game length
    total_turns = sum(session.get("total_turns", 0) for session in sessions)
    avg_game_length = round(total_turns / len(sessions), 1) if sessions else 0
    
    # Analyze character roles and their success rates
    role_stats = {}
    role_wins = {}
    
    for session in sessions:
        players = session.get("players", [])
        winner_id = session.get("winner_id")
        
        for player in players:
            role = player.get("role", "Unknown")
            # Extract role type from full description (e.g. "Bold Risk-Taker" from longer text)
            role_type = role.split(" - ")[0] if " - " in role else role
            
            if role_type not in role_stats:
                role_stats[role_type] = 0
                role_wins[role_type] = 0
            
            role_stats[role_type] += 1
            
            if winner_id and int(player.get("id", 0)) == winner_id:
                role_wins[role_type] += 1
    
    # Find dominant strategy (role with highest win rate)
    dominant_strategy = "Balanced"
    best_win_rate = 0
    
    for role, total_games in role_stats.items():
        if total_games > 0:
            win_rate = role_wins.get(role, 0) / total_games
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                dominant_strategy = role
    
    # Calculate coalition rate (analyze voting patterns from turn logs)
    total_votes = 0
    unanimous_votes = 0
    
    for session in sessions:
        turn_logs = session.get("turn_logs", [])
        
        for event in turn_logs:
            if event.get("type") == "rule_execution" and event.get("trigger") == "rule_adopted":
                total_votes += 1
                # In Nomic, rules require unanimous voting, so all adopted rules are unanimous
                unanimous_votes += 1
    
    coalition_rate = round((unanimous_votes / total_votes * 100), 1) if total_votes > 0 else 100
    
    # Calculate innovation score (based on rules created and variety)
    total_rules = sum(session.get("rules_created", 0) for session in sessions)
    total_possible_rules = len(sessions) * 3  # Assume ~3 rules per game is baseline
    innovation_score = min(round((total_rules / total_possible_rules * 100), 1), 100) if total_possible_rules > 0 else 50
    
    # Calculate strategy distribution (percentage of each major role type)
    total_players = sum(len(session.get("players", [])) for session in sessions)
    
    major_roles = ["Bold Risk-Taker", "System Builder", "Creative Disruptor", "Unpredictable Maverick"]
    strategy_distribution = []
    
    for role in major_roles:
        role_count = role_stats.get(role, 0)
        percentage = round((role_count / total_players * 100), 1) if total_players > 0 else 25
        strategy_distribution.append(percentage)
    
    # Normalize to ensure total is close to 100%
    total_percentage = sum(strategy_distribution)
    if total_percentage > 0:
        strategy_distribution = [round(p / total_percentage * 100, 1) for p in strategy_distribution]
    else:
        strategy_distribution = [25, 25, 25, 25]  # Default equal distribution
    
    return {
        "avg_game_length": avg_game_length,
        "dominant_strategy": dominant_strategy,
        "coalition_rate": coalition_rate,
        "innovation_score": innovation_score,
        "strategy_distribution": strategy_distribution
    }


def classify_model_performance(score):
    """Classify model performance based on overall score"""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Needs Improvement"


def generate_strategic_insights(model_stats, game_instance):
    """Generate strategic insights from model performance data"""
    insights = []

    if not model_stats:
        return ["No data available for analysis."]

    # Find best and worst performers
    best_model = max(model_stats.items(), key=lambda x: x[1]["overall_score"])
    worst_model = min(model_stats.items(), key=lambda x: x[1]["overall_score"])

    insights.append(f" Top performer: {best_model[0]} with {best_model[1]['overall_score']:.1f}/100 overall score")
    insights.append(
        f" Best success rate: {max(model_stats.values(), key=lambda x: x['success_rate'])['success_rate']:.1f}% proposal acceptance"
    )

    # Coherence analysis
    coherence_scores = [stats["coherence"] for stats in model_stats.values()]
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    insights.append(f" Average coherence score: {avg_coherence:.1f}/100 across all models")

    # Strategic depth analysis
    strategic_scores = [stats.get("strategic_engagement", 0) for stats in model_stats.values()]
    if strategic_scores:
        max_strategic = max(strategic_scores)
        insights.append(f" Highest strategic engagement: {max_strategic:.1f}/100 shows sophisticated gameplay")

    # Performance spread
    score_range = best_model[1]["overall_score"] - worst_model[1]["overall_score"]
    if score_range > 30:
        insights.append(
            f" High performance variance: {score_range:.1f} point spread indicates diverse model capabilities"
        )
    else:
        insights.append(f" Consistent performance: {score_range:.1f} point spread shows similar model capabilities")

    return insights


def generate_performance_chart_data(model_stats):
    """Generate chart data for performance radar chart"""
    datasets = []
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (model_name, stats) in enumerate(model_stats.items()):
        color = colors[i % len(colors)]
        datasets.append(
            {
                "label": model_name,
                "data": [
                    stats["success_rate"],
                    stats["coherence"],
                    stats.get("strategic_engagement", 0),
                    min(stats.get("adaptation_score", 50), 100),  # Cap at 100
                    min(stats.get("consistency_score", 75), 100),  # Placeholder consistency
                ],
                "backgroundColor": color + "20",
                "borderColor": color,
                "borderWidth": 2,
            }
        )

    return datasets


# Human Player Routes
@app.route("/human/turn")
def human_turn():
    """Human player interface for their turn"""
    if not game or not hasattr(game, "players"):
        return "Game not initialized", 400

    current_player = game.players[game.current_player_idx]
    if not current_player.is_human:
        return "Not human player's turn", 400

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Your Turn - Nomic Game</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        textarea { height: 120px; }
        button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        .checkbox-group { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }
        .checkbox-item { padding: 8px; background: #f8f9fa; border-radius: 4px; }
        .current-state { background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1> Your Turn - Turn {{ game.turn_number }}</h1>
        
        <div class="current-state">
            <h3> Current Game State</h3>
            <p><strong>Your Points:</strong> {{ current_player.points }}/100</p>
            <p><strong>Current Standings:</strong></p>
            <ul>
            {% for player in sorted_players %}
                <li>{{ player.name }}: {{ player.points }} points {% if player.id == current_player.id %}(YOU){% endif %}</li>
            {% endfor %}
            </ul>
        </div>

        <div class="card">
            <h3> Propose a New Rule</h3>
            <form id="proposal-form">
                <div class="form-group">
                    <label>Rule Type:</label>
                    <select id="rule-type" onchange="toggleRuleType()">
                        <option value="new">New Rule</option>
                        <option value="amendment">Amend Existing Mutable Rule</option>
                        <option value="transmute">Transmute Existing Rule</option>
                    </select>
                </div>
                
                <div id="new-rule-section">
                    <div class="form-group">
                        <label>Rule Text:</label>
                        <textarea id="rule-text" placeholder="Enter your new rule here..."></textarea>
                    </div>
                </div>
                
                <div id="amendment-section" style="display: none;">
                    <div class="form-group">
                        <label>Rule Number to Amend:</label>
                        <input type="number" id="amend-number" placeholder="e.g., 301" onchange="loadRuleText()" onblur="loadRuleText()">
                        <button type="button" onclick="loadRuleText()" style="margin-left: 10px;">Load Rule Text</button>
                        <small>Enter the number of an existing MUTABLE rule to modify</small>
                    </div>
                    <div class="form-group">
                        <label>Current Rule Text:</label>
                        <div id="current-rule-display" style="background: #f5f5f5; padding: 10px; border: 1px solid #ddd; margin-bottom: 10px; min-height: 40px;">
                            <em>Enter a rule number above to load its current text</em>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>New Rule Text:</label>
                        <textarea id="amended-text" placeholder="Edit the rule text above, then copy your changes here..."></textarea>
                        <small>Note: This will completely replace the existing rule text</small>
                    </div>
                </div>
                
                <div id="transmute-section" style="display: none;">
                    <div class="form-group">
                        <label>Rule Number to Transmute:</label>
                        <input type="number" id="transmute-number" placeholder="e.g., 301">
                        <small>Enter the number of an existing rule to change from immutable to mutable</small>
                    </div>
                </div>

                <div class="form-group">
                    <label>Explanation:</label>
                    <textarea id="explanation" placeholder="Explain why this rule is good for the game..."></textarea>
                </div>

                <h4> Rule Effects Checklist (Help the engine understand your rule)</h4>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <strong>Points System:</strong><br>
                        <input type="checkbox" id="add-points"> Add points (how many? <input type="number" id="add-points-value" style="width:60px"> to whom? <input type="text" id="add-points-target" style="width:80px">)<br>
                        <input type="checkbox" id="subtract-points"> Subtract points<br>
                        <input type="checkbox" id="steal-points"> Steal points
                    </div>
                    <div class="checkbox-item">
                        <strong>Voting Mechanics:</strong><br>
                        <input type="checkbox" id="change-threshold"> Change voting threshold<br>
                        <input type="checkbox" id="unanimous"> Require unanimous consent<br>
                        <input type="checkbox" id="voting-restrictions"> Add voting restrictions
                    </div>
                    <div class="checkbox-item">
                        <strong>Turn Mechanics:</strong><br>
                        <input type="checkbox" id="skip-turns"> Skip turns<br>
                        <input type="checkbox" id="extra-turns"> Extra turns<br>
                        <input type="checkbox" id="change-order"> Change turn order
                    </div>
                    <div class="checkbox-item">
                        <strong>Special:</strong><br>
                        <input type="checkbox" id="dice-random"> Add dice/random elements<br>
                        <input type="checkbox" id="conditional"> Add conditional triggers<br>
                        <input type="checkbox" id="win-conditions"> Modify win conditions
                    </div>
                </div>

                <button type="button" onclick="submitProposal()">Submit Proposal</button>
            </form>
        </div>
    </div>

    <script>
        // Parse URL parameters on page load
        function parseURLParams() {
            const urlParams = new URLSearchParams(window.location.search);
            
            // Check for amend parameter
            if (urlParams.has('amend')) {
                const ruleNumber = urlParams.get('amend');
                document.getElementById('rule-type').value = 'amendment';
                document.getElementById('amend-number').value = ruleNumber;
                toggleRuleType(); // Show amendment section
                loadRuleText(); // Load the rule text automatically
            }
            
            // Check for transmute parameter
            if (urlParams.has('transmute')) {
                const ruleNumber = urlParams.get('transmute');
                document.getElementById('rule-type').value = 'transmute';
                document.getElementById('transmute-number').value = ruleNumber;
                toggleRuleType(); // Show transmute section
            }
        }

        function toggleRuleType() {
            const ruleType = document.getElementById('rule-type').value;
            const newSection = document.getElementById('new-rule-section');
            const amendmentSection = document.getElementById('amendment-section');
            const transmuteSection = document.getElementById('transmute-section');
            
            // Hide all sections first
            newSection.style.display = 'none';
            amendmentSection.style.display = 'none';
            transmuteSection.style.display = 'none';
            
            // Show the appropriate section
            if (ruleType === 'new') {
                newSection.style.display = 'block';
            } else if (ruleType === 'amendment') {
                amendmentSection.style.display = 'block';
            } else if (ruleType === 'transmute') {
                transmuteSection.style.display = 'block';
            }
        }

        // Initialize page on load
        window.addEventListener('load', function() {
            parseURLParams();
        });

        function loadRuleText() {
            const ruleNumber = document.getElementById('amend-number').value;
            const displayDiv = document.getElementById('current-rule-display');
            const amendedTextArea = document.getElementById('amended-text');
            
            if (!ruleNumber) {
                displayDiv.innerHTML = '<em>Enter a rule number above to load its current text</em>';
                amendedTextArea.value = '';
                return;
            }
            
            displayDiv.innerHTML = '<em>Loading rule text...</em>';
            
            fetch(`/api/rule/${ruleNumber}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        if (data.mutable) {
                            displayDiv.innerHTML = `<strong>Rule ${ruleNumber} (MUTABLE):</strong><br>${data.text}`;
                            amendedTextArea.value = data.text; // Pre-populate with current text
                        } else {
                            displayDiv.innerHTML = `<strong style="color: red;">Rule ${ruleNumber} is IMMUTABLE and cannot be amended!</strong><br>${data.text}<br><em>Use "Transmute" to make it mutable first.</em>`;
                            amendedTextArea.value = '';
                        }
                    } else {
                        displayDiv.innerHTML = `<em style="color: red;">Rule ${ruleNumber} not found</em>`;
                        amendedTextArea.value = '';
                    }
                })
                .catch(error => {
                    displayDiv.innerHTML = '<em style="color: red;">Error loading rule text</em>';
                    amendedTextArea.value = '';
                });
        }

        function submitProposal() {
            const ruleType = document.getElementById('rule-type').value;
            const ruleText = document.getElementById('rule-text').value;
            const amendNumber = document.getElementById('amend-number').value;
            const amendedText = document.getElementById('amended-text').value;
            const transmuteNumber = document.getElementById('transmute-number').value;
            const explanation = document.getElementById('explanation').value;
            
            if (ruleType === 'new' && !ruleText.trim()) {
                alert('Please enter rule text');
                return;
            }
            if (ruleType === 'amendment' && (!amendNumber || !amendedText.trim())) {
                alert('Please enter both the rule number to amend and the new rule text');
                return;
            }
            if (ruleType === 'transmute' && !transmuteNumber) {
                alert('Please enter rule number to transmute');
                return;
            }
            if (!explanation.trim()) {
                alert('Please provide an explanation');
                return;
            }

            // Collect checkbox effects
            const effects = {
                add_points: document.getElementById('add-points').checked,
                add_points_value: document.getElementById('add-points-value').value,
                add_points_target: document.getElementById('add-points-target').value,
                subtract_points: document.getElementById('subtract-points').checked,
                steal_points: document.getElementById('steal-points').checked,
                change_threshold: document.getElementById('change-threshold').checked,
                unanimous: document.getElementById('unanimous').checked,
                voting_restrictions: document.getElementById('voting-restrictions').checked,
                skip_turns: document.getElementById('skip-turns').checked,
                extra_turns: document.getElementById('extra-turns').checked,
                change_order: document.getElementById('change-order').checked,
                dice_random: document.getElementById('dice-random').checked,
                conditional: document.getElementById('conditional').checked,
                win_conditions: document.getElementById('win-conditions').checked
            };

            // Handle different rule types
            let finalRuleText = ruleText;
            if (ruleType === 'transmute') {
                finalRuleText = `Transmute rule ${transmuteNumber}`;
            } else if (ruleType === 'amendment') {
                const amendNumber = document.getElementById('amend-number').value;
                const amendedText = document.getElementById('amended-text').value;
                finalRuleText = amendedText; // The amended text becomes the new rule text
            }

            const data = {
                rule_type: ruleType,
                rule_text: finalRuleText,
                transmute_number: ruleType === 'transmute' ? parseInt(transmuteNumber) : null,
                amend_number: ruleType === 'amendment' ? parseInt(document.getElementById('amend-number').value) : null,
                original_rule_text: ruleType === 'amendment' ? document.getElementById('current-rule-display').textContent : null,
                explanation: explanation,
                effects: effects
            };

            fetch('/human/propose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    alert('Proposal submitted! Proceeding to voting...');
                    window.location.href = '/';
                } else {
                    alert('Error: ' + result.error);
                }
            });
        }
    </script>
</body>
</html>
    """,
        game=game,
        current_player=current_player,
        sorted_players=sorted(game.players, key=lambda p: p.points, reverse=True),
    )


@app.route("/human/propose", methods=["POST"])
def human_propose():
    """Handle human player proposal submission"""
    if not game or not hasattr(game, "players"):
        return jsonify({"success": False, "error": "Game not initialized"})

    current_player = game.players[game.current_player_idx]
    if not current_player.is_human:
        return jsonify({"success": False, "error": "Not human player's turn"})

    data = request.json

    # Sanitize and validate human input
    rule_sanitization = game.input_sanitizer.sanitize_rule_text(data.get("rule_text", ""))
    if not rule_sanitization["success"]:
        return jsonify({"success": False, "error": f"Rule text error: {rule_sanitization['error']}"})

    explanation_sanitization = game.input_sanitizer.sanitize_explanation(data.get("explanation", ""))
    if not explanation_sanitization["success"]:
        return jsonify({"success": False, "error": f"Explanation error: {explanation_sanitization['error']}"})

    # Validate total input size
    total_validation = game.input_sanitizer.validate_total_input_size(
        rule_sanitization["sanitized"], explanation_sanitization["sanitized"]
    )
    if not total_validation["valid"]:
        return jsonify({"success": False, "error": total_validation["error"]})

    # Handle different rule types
    rule_type = data.get("rule_type", "new")
    is_transmutation = rule_type == "transmute"
    is_amendment = rule_type == "amendment"
    transmute_target = data.get("transmute_number") if is_transmutation else None
    amend_target = data.get("amend_number") if is_amendment else None
    original_rule_text = data.get("original_rule_text") if is_amendment else None
    
    # Create proposal from sanitized human input
    proposal = Proposal(
        id=len(game.proposals) + 1,
        player_id=current_player.id,
        rule_text=rule_sanitization["sanitized"],
        explanation=explanation_sanitization["sanitized"],
        internal_thoughts=f"Human player proposal - {rule_type} rule (sanitized: {rule_sanitization['final_length']}/{rule_sanitization['original_length']} chars)",
        turn=game.turn_number,
        is_transmutation=is_transmutation,
        transmute_target_rule=transmute_target,
        is_amendment=is_amendment,
        amend_target_rule=amend_target,
        original_rule_text=original_rule_text,
    )

    # Parse effects with enhanced parsing using sanitized text
    parsed_effects = game.parse_rule_effects_with_checkboxes(rule_sanitization["sanitized"])
    proposal.parsed_effects = parsed_effects

    # Add to game
    game.proposals.append(proposal)
    game.current_proposal = proposal

    # Log the proposal
    game.game_logger.log_proposal(current_player.id, proposal)
    game.add_event(f" {current_player.name} proposes Rule {game.next_rule_number}: {proposal.rule_text}")
    game.add_event(f" Explanation: {proposal.explanation}")

    # Start async deliberation immediately for all AI players
    game.start_async_deliberation(proposal)

    # Update player's internal state with the proposal
    game.state_tracker.add_planned_proposal(current_player.id, f"Proposed: {proposal.rule_text}")

    # Clear the waiting flag and continue the game flow
    game.waiting_for_human = False
    game.continue_turn_after_human_proposal()

    return jsonify({"success": True})


@app.route("/api/rule/<int:rule_id>")
def get_rule_details(rule_id):
    """Get details of a specific rule for amendment purposes"""
    if not game:
        return jsonify({"success": False, "error": "No game running"})
    
    # Search in both mutable and immutable rules
    for rule in game.rules["mutable"]:
        if rule.id == rule_id:
            return jsonify({
                "success": True,
                "rule_id": rule.id,
                "text": rule.text,
                "mutable": True,
                "can_amend": True
            })
    
    for rule in game.rules["immutable"]:
        if rule.id == rule_id:
            return jsonify({
                "success": True,
                "rule_id": rule.id,
                "text": rule.text,
                "mutable": False,
                "can_amend": False
            })
    
    return jsonify({"success": False, "error": f"Rule {rule_id} not found"})


@app.route("/api/rules")
def get_all_rules():
    """Get all current rules for browsing"""
    if not game:
        return jsonify({"success": False, "error": "No game running"})
    
    rules_data = {
        "success": True,
        "mutable_rules": [],
        "immutable_rules": []
    }
    
    # Add mutable rules
    for rule in game.rules["mutable"]:
        rules_data["mutable_rules"].append({
            "id": rule.id,
            "text": rule.text,
            "author": getattr(rule, 'author', None),
            "turn_added": getattr(rule, 'turn_added', None)
        })
    
    # Add immutable rules
    for rule in game.rules["immutable"]:
        rules_data["immutable_rules"].append({
            "id": rule.id,
            "text": rule.text,
            "author": getattr(rule, 'author', None),
            "turn_added": getattr(rule, 'turn_added', None)
        })
    
    return jsonify(rules_data)


@app.route("/human/rules")
def human_rules_browser():
    """Rule browsing interface for human players"""
    if not game:
        return "No game running", 400
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Rule Browser - Nomic Game</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; }
        .nav { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .nav a { margin: 0 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
        .nav a:hover { background: #0056b3; }
        .rules-section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .rule-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 4px; }
        .rule-mutable { border-left: 4px solid #28a745; background: #f8fff9; }
        .rule-immutable { border-left: 4px solid #dc3545; background: #fff8f8; }
        .rule-header { font-weight: bold; margin-bottom: 10px; }
        .rule-text { line-height: 1.5; }
        .rule-meta { font-size: 0.9em; color: #666; margin-top: 10px; }
        .search-box { width: 100%; padding: 10px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 20px; }
        .quick-actions { background: #e9ecef; padding: 15px; border-radius: 4px; margin: 10px 0; }
        .quick-actions button { margin: 5px; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-amend { background: #ffc107; color: black; }
        .btn-transmute { background: #17a2b8; color: white; }
        .btn-amend:hover { background: #e0a800; }
        .btn-transmute:hover { background: #138496; }
        .loading { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">← Back to Game</a>
            <a href="/human/propose">Create New Rule</a>
            <a href="#" onclick="location.reload()">🔄 Refresh Rules</a>
        </div>
        
        <h1>📜 Rule Browser</h1>
        
        <input type="text" id="search-box" class="search-box" placeholder="🔍 Search rules by text, number, or author..." onkeyup="filterRules()">
        
        <div id="rules-container" class="loading">
            Loading rules...
        </div>
    </div>

    <script>
        let allRules = [];
        
        function loadRules() {
            fetch('/api/rules')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allRules = [...data.mutable_rules.map(r => ({...r, mutable: true})), 
                                   ...data.immutable_rules.map(r => ({...r, mutable: false}))];
                        displayRules(allRules);
                    } else {
                        document.getElementById('rules-container').innerHTML = '<p>Error loading rules: ' + data.error + '</p>';
                    }
                })
                .catch(error => {
                    document.getElementById('rules-container').innerHTML = '<p>Error loading rules</p>';
                });
        }
        
        function displayRules(rules) {
            const container = document.getElementById('rules-container');
            
            if (rules.length === 0) {
                container.innerHTML = '<p>No rules found.</p>';
                return;
            }
            
            // Sort rules by ID
            rules.sort((a, b) => a.id - b.id);
            
            // Group by mutable/immutable
            const mutableRules = rules.filter(r => r.mutable);
            const immutableRules = rules.filter(r => !r.mutable);
            
            let html = '';
            
            if (mutableRules.length > 0) {
                html += '<div class="rules-section"><h2>📝 Mutable Rules (Can be amended)</h2>';
                mutableRules.forEach(rule => {
                    html += generateRuleHTML(rule);
                });
                html += '</div>';
            }
            
            if (immutableRules.length > 0) {
                html += '<div class="rules-section"><h2>🔒 Immutable Rules (Permanent unless transmuted)</h2>';
                immutableRules.forEach(rule => {
                    html += generateRuleHTML(rule);
                });
                html += '</div>';
            }
            
            container.innerHTML = html;
        }
        
        function generateRuleHTML(rule) {
            const ruleClass = rule.mutable ? 'rule-mutable' : 'rule-immutable';
            const statusBadge = rule.mutable ? '<span style="background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">MUTABLE</span>' 
                                            : '<span style="background: #dc3545; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">IMMUTABLE</span>';
            
            let actions = '';
            if (rule.mutable) {
                actions = `<div class="quick-actions">
                    <button class="btn-amend" onclick="amendRule(${rule.id})">✏️ Amend This Rule</button>
                    <button class="btn-transmute" onclick="transmuteRule(${rule.id})">🔄 Make Immutable</button>
                </div>`;
            } else {
                actions = `<div class="quick-actions">
                    <button class="btn-transmute" onclick="transmuteRule(${rule.id})">🔄 Make Mutable (then amend)</button>
                </div>`;
            }
            
            return `
                <div class="rule-item ${ruleClass}" data-rule-id="${rule.id}" data-rule-text="${rule.text.toLowerCase()}">
                    <div class="rule-header">
                        Rule ${rule.id} ${statusBadge}
                    </div>
                    <div class="rule-text">${rule.text}</div>
                    <div class="rule-meta">
                        ${rule.author ? `Added by Player ${rule.author}` : 'System rule'} 
                        ${rule.turn_added ? ` • Turn ${rule.turn_added}` : ''}
                    </div>
                    ${actions}
                </div>
            `;
        }
        
        function filterRules() {
            const searchTerm = document.getElementById('search-box').value.toLowerCase();
            if (!searchTerm) {
                displayRules(allRules);
                return;
            }
            
            const filteredRules = allRules.filter(rule => 
                rule.text.toLowerCase().includes(searchTerm) ||
                rule.id.toString().includes(searchTerm) ||
                (rule.author && rule.author.toString().includes(searchTerm))
            );
            
            displayRules(filteredRules);
        }
        
        function amendRule(ruleId) {
            window.location.href = `/human/propose?amend=${ruleId}`;
        }
        
        function transmuteRule(ruleId) {
            window.location.href = `/human/propose?transmute=${ruleId}`;
        }
        
        // Load rules on page load
        loadRules();
    </script>
</body>
</html>
    """)


@app.route("/human/vote")
def human_vote():
    """Human player voting interface"""
    if not game or not game.current_proposal:
        return "No active proposal", 400

    current_player = game.players[game.current_player_idx]
    if not current_player.is_human:
        return "Not human player's turn to vote", 400

    proposal = game.current_proposal
    proposer = next(p for p in game.players if p.id == proposal.player_id)

    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Vote - Nomic Game</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .proposal { background: #fff3e0; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; border-radius: 4px; }
        button { padding: 12px 24px; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 10px; font-size: 16px; }
        .vote-aye { background: #4caf50; }
        .vote-nay { background: #f44336; }
        button:hover { opacity: 0.9; }
        .current-state { background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1> Vote on Proposal</h1>
        
        <div class="current-state">
            <h3> Your Position</h3>
            <p><strong>Your Points:</strong> {{ current_player.points }}/100</p>
            <p><strong>Your Ranking:</strong> {{ ranking }} out of {{ game.players|length }}</p>
        </div>

        <div class="proposal">
            <h3> Proposal by {{ proposer.name }}</h3>
            <p><strong>Rule:</strong> {{ proposal.rule_text }}</p>
            <p><strong>Explanation:</strong> {{ proposal.explanation }}</p>
        </div>

        <div class="card">
            <h3> Consider This Proposal</h3>
            <p>Think about:</p>
            <ul>
                <li>How does this rule help or hurt YOUR chances of winning?</li>
                <li>Does this rule benefit the proposer more than you?</li>
                <li>Will this rule help you catch up or maintain your lead?</li>
                <li>Is this rule fair and good for the game overall?</li>
            </ul>
            
            <div style="text-align: center; margin-top: 30px;">
                <button class="vote-aye" onclick="submitVote(true)"> Vote AYE (Yes)</button>
                <button class="vote-nay" onclick="submitVote(false)"> Vote NAY (No)</button>
            </div>
        </div>
    </div>

    <script>
        function submitVote(vote) {
            fetch('/human/submit-vote', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    vote: vote,
                    reasoning: vote ? "Human voted AYE" : "Human voted NAY"
                })
            })
            .then(r => r.json())
            .then(result => {
                if (result.success) {
                    alert(vote ? 'Voted AYE!' : 'Voted NAY!');
                    window.location.href = '/';
                } else {
                    alert('Error: ' + result.error);
                }
            });
        }
    </script>
</body>
</html>
    """,
        game=game,
        proposal=proposal,
        proposer=proposer,
        current_player=current_player,
        ranking=sorted(game.players, key=lambda p: p.points, reverse=True).index(current_player) + 1,
    )


@app.route("/human/submit-vote", methods=["POST"])
def human_submit_vote():
    """Handle human player vote submission"""
    if not game or not game.current_proposal:
        return jsonify({"success": False, "error": "No active proposal"})

    current_player = game.players[game.current_player_idx]
    if not current_player.is_human:
        return jsonify({"success": False, "error": "Not human player's turn to vote"})

    data = request.json
    vote = data["vote"]
    raw_reasoning = data["reasoning"]

    # Check if player has already voted
    if current_player.id in game.current_proposal.votes:
        return jsonify({"error": "You have already voted on this proposal"})

    # Sanitize voting reasoning (basic safety)
    reasoning = game.input_sanitizer.sanitize_text(raw_reasoning, max_length=100)
    if not reasoning:
        reasoning = f"Human voted {'AYE' if vote else 'NAY'}"

    # Record the vote
    game.current_proposal.votes[current_player.id] = vote

    # Log the vote
    game.game_logger.log_vote(current_player.id, vote, reasoning)
    vote_text = "Aye" if vote else "Nay"
    game.add_event(f" {current_player.name}: {vote_text} - {reasoning}")

    # Check if all votes are now complete and continue game if so
    game.check_voting_complete_and_continue()

    return jsonify({"success": True})


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Strategic Nomic Game with LLM Provider Options")
    parser.add_argument(
        "--openrouter", action="store_true", help="Use OpenRouter API instead of local Ollama for SOTA model testing"
    )
    parser.add_argument(
        "--openrouter-key", type=str, help="OpenRouter API key (can also use OPENROUTER_API_KEY environment variable)"
    )
    parser.add_argument(
        "--big", action="store_true", help="Use premium SOTA models (Sonnet 3.5, GPT-4o, Grok 3, Gemini 2.5 Pro) - requires OpenRouter"
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Use hybrid big/small model system for cost optimization (70-75% savings) - requires OpenRouter"
    )
    parser.add_argument(
        "--budget", type=float, default=5.0, help="Session budget limit in USD (default: $5.00)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Flask web interface port (default: 8080)"
    )
    parser.add_argument(
        "--auto-port", action="store_true", help="Automatically find available port if default is busy"
    )
    parser.add_argument(
        "--ollama-base-port", type=int, default=11435, help="Base port for Ollama validator (default: 11435)"
    )
    parser.add_argument("--players", type=int, default=6, help="Number of players (default: 6)")
    parser.add_argument("--human", action="store_true", help="Include a human player in the game")
    parser.add_argument("--resume", action="store_true", help="Resume the most recent in-progress game session")

    args = parser.parse_args()

    # Port conflict detection and auto-assignment
    def check_port_available(port):
        """Check if a port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False

    def find_available_port_range(start_port, num_ports):
        """Find a range of available ports starting from start_port"""
        for base in range(start_port, start_port + 100):  # Try 100 different starting points
            if all(check_port_available(base + i) for i in range(num_ports)):
                return base
        return None

    # Check and adjust Flask port
    original_port = args.port
    if not check_port_available(args.port):
        if args.auto_port:
            new_port = find_available_port_range(args.port + 1, 1)
            if new_port:
                print(f"🔄 Port {args.port} busy, using available port {new_port}")
                args.port = new_port
            else:
                print(f"❌ Could not find available port near {args.port}")
                exit(1)
        else:
            print(f"❌ Port {args.port} is already in use!")
            print(f"   Current process using port {args.port}:")
            import subprocess
            try:
                result = subprocess.run(['lsof', '-i', f':{args.port}'], capture_output=True, text=True)
                print(f"   {result.stdout}")
            except:
                pass
            print("   Use --auto-port to automatically find available port")
            exit(1)

    # Check and adjust Ollama ports (only if using Ollama)
    if not (args.openrouter or args.big):
        # Calculate required Ollama ports: validator + players
        ollama_ports_needed = args.players + 1  # validator + player ports
        
        if not check_port_available(args.ollama_base_port):
            print(f"⚠️  Ollama validator port {args.ollama_base_port} is busy")
            if args.auto_port:
                new_base = find_available_port_range(args.ollama_base_port + 1, ollama_ports_needed)
                if new_base:
                    print(f"🔄 Using Ollama port range {new_base}-{new_base + ollama_ports_needed - 1}")
                    args.ollama_base_port = new_base
                else:
                    print(f"❌ Could not find {ollama_ports_needed} consecutive Ollama ports")
                    exit(1)
            else:
                print("   Use --auto-port to automatically find available Ollama ports")
                exit(1)
        else:
            # Check if we have enough consecutive ports
            for i in range(ollama_ports_needed):
                if not check_port_available(args.ollama_base_port + i):
                    print(f"⚠️  Ollama port {args.ollama_base_port + i} is busy")
                    if args.auto_port:
                        new_base = find_available_port_range(args.ollama_base_port + i + 1, ollama_ports_needed)
                        if new_base:
                            print(f"🔄 Using Ollama port range {new_base}-{new_base + ollama_ports_needed - 1}")
                            args.ollama_base_port = new_base
                            break
                        else:
                            print(f"❌ Could not find {ollama_ports_needed} consecutive Ollama ports")
                            exit(1)
                    else:
                        print("   Use --auto-port to automatically find available Ollama ports")
                        exit(1)

    # Initialize LLM provider based on arguments
    llm_provider = None
    provider_info = ""

    if args.openrouter or args.big:
        # Big models require OpenRouter
        if args.big and not args.openrouter:
            args.openrouter = True
            print("🚀 --big flag detected: automatically enabling OpenRouter for premium models")
        
        # Get API key from argument or environment variable
        api_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            print("❌ Error: OpenRouter API key required!")
            print("   Use --openrouter-key YOUR_KEY or set OPENROUTER_API_KEY environment variable")
            print("   Get your API key from: https://openrouter.ai/")
            exit(1)

        try:
            # Initialize with big models and/or hybrid mode if requested
            llm_provider = OpenRouterClient(api_key, use_big_models=args.big, use_hybrid=args.hybrid)
            
            # Set session budget
            llm_provider.session_budget = args.budget
            
            if args.big:
                print(f"🎯 Big Models Mode: Using premium SOTA models")
                print(f"💰 Session Budget: ${args.budget:.2f}")
                print("⚡ Models: Claude 3.5 Sonnet, GPT-4o, Grok 3, Gemini 2.5 Pro")
                
                # Verify model availability
                verification_results = llm_provider.verify_api_models()
                available_models = [model for model, result in verification_results.items() if result["available"]]
                
                if len(available_models) == 0:
                    print("❌ No big models available! Check API key and credits.")
                    exit(1)
                elif len(available_models) < len(llm_provider.big_models):
                    print(f"⚠️  Only {len(available_models)}/{len(llm_provider.big_models)} big models available")
                    llm_provider.available_models = available_models
                
                provider_info = f"🚀 OpenRouter Big Models (Available: {', '.join(available_models)})"
            else:
                provider_info = f"🌐 OpenRouter API (Models: {', '.join(llm_provider.available_models)})"
                
            print("✅ OpenRouter client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing OpenRouter client: {e}")
            exit(1)
    else:
        # Use Ollama with configurable ports
        llm_provider = OllamaClient(base_port=args.ollama_base_port)
        provider_info = f"🔧 Local Ollama (Validator: {args.ollama_base_port}, Players: {args.ollama_base_port + 1}+)"

    print(" Starting Advanced Strategic Nomic Game with Comprehensive Logging & State Tracking")
    print("=" * 80)
    print(f" LLM Provider: {provider_info}")
    print(f" Web Interfaces (Port {args.port}):")
    print(f"   - Main Game: http://127.0.0.1:{args.port}")
    print(f"   - Model Statistics: http://127.0.0.1:{args.port}/stats")
    print(f"   - Model Performance Analytics: http://127.0.0.1:{args.port}/analytics")
    print(f"   - Player Internal States: http://127.0.0.1:{args.port}/player-states")
    print(f"   - Game Logs Viewer: http://127.0.0.1:{args.port}/game-logs")
    print(f"   - Session History: http://127.0.0.1:{args.port}/sessions")
    if args.openrouter or args.big:
        print(f"   - Cost Tracking Dashboard: http://127.0.0.1:{args.port}/costs")
    print()
    print(" Enhanced Features:")
    print("   - 5-turn proposal deliberation loops for strategic thinking")
    print("   - 2-turn competitive voting deliberation with impact analysis")
    print("   - 1-3 idle turns for background strategy refinement")
    print("   - Comprehensive player internal state tracking")
    print("   - Persistent game session storage and logging")
    print("   - 8 distinct proposal categories with diversity enforcement")
    print("   - Multi-instance Ollama with performance-based model assignment")
    print("   - Cross-game model performance metrics and quality classification")
    print("   - Real-time strategic thinking and threat assessment visualization")
    print("   - Complete turn-by-turn logging with export capabilities")
    print()
    print(" New Systems:")
    print("   - GameSessionManager: Persistent storage of all game data")
    print("   - InternalStateTracker: Comprehensive player mental state tracking")
    print("   - IdleTurnProcessor: Background strategic thinking during other turns")
    print("   - GameLogger: Structured logging of all events and decisions")
    print()
    print(" Player Internal State Tracking:")
    print("   - Strategic focus and victory path planning")
    print("   - Threat assessments and alliance considerations")
    print("   - Planned proposals and voting strategies")
    print("   - Learning observations and rule effectiveness notes")
    print("   - Idle turn strategic analysis and refinement")
    print()
    print(" Data Persistence:")
    print("   - Complete game session archives in port-specific directories")
    print("   - Turn-by-turn logs with player reasoning and decisions")
    print("   - Player internal state snapshots")
    print("   - Idle turn deliberation logs")
    print("   - Cross-game model performance tracking")

    if args.openrouter or args.big or args.hybrid:
        print()
        if args.hybrid:
            print("⚡ Hybrid Cost Optimization Mode Features:")
            print("   - Intelligent big/small model pairing per family")
            print("   - 70-75% cost reduction with preserved strategic quality")
            print("   - Automatic escalation for critical decisions")
            print("   - Budget-aware threshold adjustment")
            print("   - Real-time complexity detection and model selection")
            print(f"   - Session budget: ${args.budget:.2f}")
        elif args.big:
            print("🚀 Big Models Mode Features:")
            print("   - Premium SOTA models: Claude 3.5 Sonnet, GPT-4o, Grok 3, Gemini 2.5 Pro")
            print("   - Enhanced context windows (up to 1.05M tokens)")
            print("   - Superior reasoning and strategic depth")
            print("   - Real-time cost tracking with budget controls")
            print(f"   - Session budget: ${args.budget:.2f}")
        else:
            print(" OpenRouter Features:")
            print("   - Standard models: Gemini 2.0 Flash, GPT-4o Mini, Claude 3.5 Haiku")
            print("   - Real-time cost tracking and usage analytics")
            print("   - No local hardware limitations")
            print("   - Access to models not available locally")

    # Create port-specific session directory for isolation
    session_dir = f"game_sessions_port{args.port}"
    
    # Show port configuration summary
    if not (args.openrouter or args.big):
        print()
        print(f"🔧 Port Configuration:")
        print(f"   - Flask Web Interface: {args.port}")
        print(f"   - Ollama Validator: {args.ollama_base_port}")
        print(f"   - Ollama Players: {args.ollama_base_port + 1}-{args.ollama_base_port + args.players}")
        print(f"   - Session Storage: {session_dir}/")
    else:
        print()
        print(f"🌐 Configuration:")
        print(f"   - Flask Web Interface: {args.port}")
        print(f"   - Session Storage: {session_dir}/")
        print(f"   - Models: Cloud-based via OpenRouter API")
    
    print()
    print("  Press Ctrl+C to stop")
    
    # Create game instance with the selected provider
    if args.resume:
        print()
        print("🔄 Attempting to resume most recent in-progress game...")
        game_instance = ProperNomicGame.resume_last_game(provider=llm_provider, session_dir=session_dir)
        if game_instance is None:
            print("❌ Resume failed. Starting new game instead...")
            game_instance = ProperNomicGame(num_players=args.players, provider=llm_provider, include_human=args.human, session_dir=session_dir)
    else:
        game_instance = ProperNomicGame(num_players=args.players, provider=llm_provider, include_human=args.human, session_dir=session_dir)

    # Create a NEW Flask app instance for this specific port
    # This completely isolates the game instance from other ports
    port_app = Flask(__name__)
    port_app.config['game'] = game_instance
    
    # Copy all route functions to the new app instance
    # This is the key fix: each port gets its own app with its own game instance
    for rule in app.url_map.iter_rules():
        port_app.add_url_rule(
            rule.rule,
            endpoint=rule.endpoint, 
            view_func=app.view_functions[rule.endpoint],
            methods=rule.methods
        )
    
    # Add logging validation
    print(f"✅ Game instance created for port {args.port}")
    print(f"✅ Session ID: {game_instance.session_id}")
    print(f"✅ Logging pipeline: {'Active' if game_instance.game_logger else 'Inactive'}")
    print(f"✅ Session directory: {game_instance.session_manager.sessions_dir}")

    port_app.run(host="127.0.0.1", port=args.port, debug=False)
