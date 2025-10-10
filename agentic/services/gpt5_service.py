#!/usr/bin/env python3
"""
GPT-5 API Service with fallback to local LLM
Provides high-quality text generation for resource enrichment
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json

import aiohttp
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GPT5Service:
    """Service for GPT-5 API calls with caching and fallback"""
    
    def __init__(self, use_cache: bool = True):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-5-mini"
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.use_cache = use_cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(days=30)
        
        # Fallback to local LLM
        self.fallback_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
        self.fallback_model = os.getenv('LLM_MODEL', 'llama3.1:8b')
        
        # Usage tracking
        self.stats = {
            'gpt5_calls': 0,
            'gpt5_cached': 0,
            'fallback_calls': 0,
            'errors': 0
        }
        
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found - will use local LLM fallback only")
    
    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate cache key from prompt and parameters"""
        import hashlib
        key_str = f"{prompt}_{temperature}_{self.model}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_entry: Dict) -> bool:
        """Check if cached entry is still valid"""
        if not cached_entry:
            return False
        cached_time = datetime.fromisoformat(cached_entry.get('timestamp', '2000-01-01'))
        return datetime.now() - cached_time < self.cache_ttl
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        system_message: Optional[str] = None
    ) -> str:
        """Generate text using GPT-5 mini with fallback"""
        
        # Check cache
        cache_key = self._get_cache_key(prompt, temperature)
        if self.use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            if self._is_cache_valid(cached):
                self.stats['gpt5_cached'] += 1
                logger.debug(f"📦 Cache hit for prompt: {prompt[:50]}...")
                return cached['response']
        
        # Try GPT-5 API first
        if self.api_key:
            try:
                response = await self._call_gpt5_api(
                    prompt, temperature, max_tokens, system_message
                )
                
                # Cache the result
                if self.use_cache:
                    self.cache[cache_key] = {
                        'response': response,
                        'timestamp': datetime.now().isoformat()
                    }
                
                self.stats['gpt5_calls'] += 1
                return response
                
            except Exception as e:
                logger.warning(f"⚠️ GPT-5 API error: {e}, falling back to local LLM")
                self.stats['errors'] += 1
        
        # Fallback to local LLM
        try:
            response = await self._call_local_llm(prompt, temperature, max_tokens)
            self.stats['fallback_calls'] += 1
            return response
        except Exception as e:
            logger.error(f"❌ Both GPT-5 and local LLM failed: {e}")
            self.stats['errors'] += 1
            return ""
    
    async def _call_gpt5_api(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str]
    ) -> str:
        """Call OpenAI GPT-5 API"""
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API returned {response.status}: {error_text}")
                
                data = await response.json()
                return data['choices'][0]['message']['content'].strip()
    
    async def _call_local_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Fallback to local Ollama LLM"""
        
        payload = {
            "model": self.fallback_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.fallback_endpoint,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Local LLM returned {response.status}")
                
                data = await response.json()
                return data.get('response', '').strip()
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.stats = {
            'gpt5_calls': 0,
            'gpt5_cached': 0,
            'fallback_calls': 0,
            'errors': 0
        }

