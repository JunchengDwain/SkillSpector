# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared constructors for provider-backed LangChain chat models."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def is_deepseek_endpoint(base_url: str | None) -> bool:
    """Detect DeepSeek's OpenAI-compatible endpoint from the base URL.

    DeepSeek does not implement OpenAI's ``response_format={"type":"json_schema"}``
    (returns 400 ``This response_format type is unavailable now``).  It only
    accepts ``json_object``, and that mode additionally requires the prompt to
    contain the substring ``json``.  We tag DeepSeek endpoints so callers can
    switch to a prompt-driven JSON path with ``json_object`` instead of
    Pydantic-driven json_schema.
    """
    if not base_url:
        return False
    normalized = base_url.lower().rstrip("/")
    return "api.deepseek.com" in normalized


def create_openai_compatible_chat_model(
    *,
    model: str,
    credentials: tuple[str, str | None] | None,
    max_tokens: int,
    timeout: float | None = 120,
    default_headers: dict[str, str] | None = None,
) -> BaseChatModel | None:
    """Create ``ChatOpenAI`` for providers serving OpenAI-compatible endpoints."""
    if credentials is None:
        return None

    api_key, base_url = credentials

    model_kwargs: dict[str, object] = {}
    if is_deepseek_endpoint(base_url):
        model_kwargs["response_format"] = {"type": "json_object"}
        # DeepSeek's reasoning-capable models (e.g. deepseek-v4-flash) burn
        # ``max_completion_tokens`` on internal chain-of-thought, so the
        # visible ``content`` is often empty even when the call succeeded.
        # Bump the budget by 4x so the JSON payload fits after reasoning.
        max_tokens = max(max_tokens * 4, 16384)

    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        max_completion_tokens=max_tokens,
        timeout=timeout,
        default_headers=default_headers,
        model_kwargs=model_kwargs or None,
    )
    # Explicit tag so downstream code (e.g. llm_analyzer_base) can route
    # DeepSeek to the prompt-driven JSON path without poking into private
    # LangChain/OpenAI client state.
    if is_deepseek_endpoint(base_url):
        setattr(chat, "_skillspector_deepseek", True)
    return chat
