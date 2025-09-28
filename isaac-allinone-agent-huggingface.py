"""Example demonstrating the ComputerAgent capabilities with the Omni provider."""
from __future__ import annotations

import asyncio
import logging
import traceback
import signal

from computer import Computer, VMProviderType

# Import the unified agent class and types
from agent import ComputerAgent

# Import utility functions
from utils import load_dotenv_files, handle_sigint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_attention_forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
    """Replacement forward method that uses regular attention instead of flash attention."""
    batch_size, seq_len, _ = hidden_states.size()

    # For variable-length attention, we need to reshape to (total_tokens, embed_dim)
    if batch_size != 1:
        raise ValueError("Variable-length attention expects batch_size=1 for packed sequences")
    hidden_states = hidden_states.squeeze(0)  # Remove batch dimension: (seq_len, embed_dim)

    # Store original dtype
    orig_dtype = hidden_states.dtype

    # 1. Linear projections
    Q = self.q_proj(hidden_states)  # (seq_len, embed_dim)
    K = self.k_proj(hidden_states)  # (seq_len, embed_dim)
    V = self.v_proj(hidden_states)  # (seq_len, embed_dim)

    # 2. Reshape for multi-head attention: (seq_len, n_heads, head_dim)
    Q = Q.view(-1, self.num_heads, self.embed_dim // self.num_heads)
    K = K.view(-1, self.num_heads, self.embed_dim // self.num_heads)
    V = V.view(-1, self.num_heads, self.embed_dim // self.num_heads)

    # 3. Apply regular scaled dot-product attention
    if cu_seqlens is not None and len(cu_seqlens) > 2:
        # Handle variable-length sequences with masking
        attn_outputs = []
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            
            q_seq = Q[start_idx:end_idx]  # (seq_i_len, num_heads, head_dim)
            k_seq = K[start_idx:end_idx]
            v_seq = V[start_idx:end_idx]
            
            # Transpose for attention: (num_heads, seq_len, head_dim)
            q_seq = q_seq.transpose(0, 1)
            k_seq = k_seq.transpose(0, 1)
            v_seq = v_seq.transpose(0, 1)
            
            # Compute attention scores
            attn_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * self.scale
            attn_probs = torch.softmax(attn_scores, dim=-1)
            
            if self.training and self.dropout > 0:
                attn_probs = torch.dropout(attn_probs, self.dropout, train=True)
            
            # Apply attention to values
            seq_output = torch.matmul(attn_probs, v_seq)
            # Transpose back: (seq_len, num_heads, head_dim)
            seq_output = seq_output.transpose(0, 1)
            
            attn_outputs.append(seq_output)
        
        attn_output = torch.cat(attn_outputs, dim=0)
    else:
        # Regular attention for single sequence
        # Transpose for attention: (num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        if self.training and self.dropout > 0:
            attn_probs = torch.dropout(attn_probs, self.dropout, train=True)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, V)
        # Transpose back: (seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(0, 1)

    # 4. Reshape attention output from (seq_len, n_heads, head_dim) to (seq_len, embed_dim)
    attn_output = attn_output.reshape(seq_len, self.embed_dim)

    # 5. Convert back to original dtype if needed
    if attn_output.dtype != orig_dtype:
        attn_output = attn_output.to(orig_dtype)

    # 6. Project output
    attn_output = self.out_proj(attn_output)  # (seq_len, embed_dim)

    # 7. Add back batch dimension for compatibility
    attn_output = attn_output.unsqueeze(0)  # (1, seq_len, embed_dim)

    return attn_output, None

def apply_attention_patch(model):
    """Apply the attention patch to all Siglip2VariableLengthAttention instances in the model."""
    patched_count = 0
    
    def patch_module(module):
        nonlocal patched_count
        for name, child in module.named_children():
            if child.__class__.__name__ == 'Siglip2VariableLengthAttention':
                logger.info(f"Patching {child.__class__.__name__} at {name}")
                child.forward = patch_attention_forward.__get__(child, child.__class__)
                patched_count += 1
            else:
                patch_module(child)
    
    patch_module(model)
    logger.info(f"Successfully patched {patched_count} functions")
    return patched_count > 0

import io
import re
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from agent import register_agent
from agent.loops.base import AsyncAgentConfig
from agent.responses import (
    make_output_text_item,
    make_click_item,
    make_double_click_item,
    make_drag_item,
    make_keypress_item,
    make_scroll_item,
    make_move_item,
    make_type_item,
    make_wait_item,
    make_screenshot_item,
    convert_responses_items_to_completion_messages,
)

@register_agent(models=r"(?i)^perceptron/.*$")
class PerceptronIsaacToolCallingConfig(AsyncAgentConfig):
    """
    Perceptron (Isaac-0.1) loop that enforces OpenAI-style tool call output (JSON-only)
    and maps it to Responses computer_call items. One tool call per step.

    Use: model="perceptron/isaac-0.1"
    """

    HF_PATH_MAP: Dict[str, str] = {"isaac-0.1": "PerceptronAI/Isaac-0.1"}

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._config = None
        self._device = None
        self._eos_token_id = None
        self._pad_token_id = None
        self._vision_token = "<image>"

        # Strict action space prompt (OpenAI-style tool call)
        # Require exactly one JSON object with either:
        #   {"name":"computer","arguments":{...}}
        # or:
        #   {"tool_calls":[{"type":"function","function":{"name":"computer","arguments":{...}}}]}
        self._action_prompt = (
            "You are an autonomous computer-using agent.\n"
            "- Output exactly ONE tool call in JSON only (no prose, no code fences, no XML/HTML tags, no <think> tags).\n"
            "- The tool name is always \"computer\". The arguments object must include an \"action\" key.\n"
            "- All JSON keys and string values MUST use double quotes.\n"
            "- Do NOT include any text before or after the JSON object.\n\n"
            "Allowed actions and required fields:\n"
            "- click: {\"action\":\"click\", \"x\":int, \"y\":int, \"button\":\"left\"|\"right\"|\"wheel\" (optional)}\n"
            "- double_click: {\"action\":\"double_click\", \"x\":int, \"y\":int}\n"
            "- move: {\"action\":\"move\", \"x\":int, \"y\":int}\n"
            "- scroll: {\"action\":\"scroll\", \"x\":int, \"y\":int, \"scroll_x\":int, \"scroll_y\":int}\n"
            "- type: {\"action\":\"type\", \"text\":string}\n"
            "- keypress: {\"action\":\"keypress\", \"keys\": string or [string,...]}\n"
            "- drag: {\"action\":\"drag\", \"path\":[{\"x\":int,\"y\":int}, ...]}  (or start_x,start_y,end_x,end_y)\n"
            "- wait: {\"action\":\"wait\"}\n"
            "- screenshot: {\"action\":\"screenshot\"}\n\n"
            "Return one of these JSON formats (choose one):\n"
            "{\"name\":\"computer\",\"arguments\":{...}}\n"
            "or\n"
            "{\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"computer\",\"arguments\":{...}}}]}\n"
        )

    # ------------------------------ Helpers ------------------------------

    def _resolve_hf_path(self, model: str) -> str:
        try:
            slug = model.split("/", 1)[1].strip()
        except Exception:
            slug = "isaac-0.1"
        return self.HF_PATH_MAP.get(slug.lower(), "PerceptronAI/Isaac-0.1")

    def _ensure_loaded(self, hf_path: str) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self._config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        self._processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

        # Apply the attention patch
        logger.info("Attempting to monkeypatch Siglip2VariableLengthAttention")
        if not apply_attention_patch(model):
            logger.warning("No monkeypatching occurred!")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = model.to(device=device, dtype=dtype).eval()

        self._model = model
        self._device = device

        tok = getattr(self._processor, "tokenizer", None)
        if tok is not None:
            self._eos_token_id = tok.eos_token_id
            self._pad_token_id = tok.eos_token_id
        self._vision_token = getattr(self._config, "vision_token", "<image>")

    def _load_pil(self, url: str) -> Optional[Image.Image]:
        try:
            if url.startswith("data:image/"):
                _, b64data = url.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
            if url.startswith("http://") or url.startswith("https://"):
                import requests

                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            return Image.open(url).convert("RGB")
        except Exception:
            return None

    def _messages_have_image(self, messages: List[Dict[str, Any]]) -> bool:
        # Look for user content with image_url items or computer_call_output image
        for m in reversed(messages):
            if m.get("type") == "computer_call_output":
                out = m.get("output", {})
                if isinstance(out, dict) and out.get("type") == "input_image":
                    return True
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                for part in m["content"]:
                    if isinstance(part, dict) and part.get("type") == "input_image":
                        return True
        return False

    def _to_perceptron_inputs(
        self, responses_messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], List[Image.Image]]:
        """
        Convert Responses-format items into Perceptron chat-template inputs:
        - messages: list of {"role": "user"|"assistant", "content": "..."}
        - images: list of PIL.Image aligned with vision tokens in messages
        """
        cmessages = convert_responses_items_to_completion_messages(
            responses_messages, allow_images_in_tool_results=False
        )
        messages: List[Dict[str, str]] = []
        images: List[Image.Image] = []

        def norm_role(role: str) -> str:
            return "assistant" if role == "assistant" else "user"

        for msg in cmessages:
            role = norm_role(str(msg.get("role", "user")))
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    messages.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                text_chunks: List[str] = []
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        t = part.get("text", "")
                        if t:
                            text_chunks.append(t)
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if not url:
                            continue
                        pil_img = self._load_pil(url)
                        if pil_img is not None:
                            images.append(pil_img)
                            messages.append({"role": role, "content": self._vision_token})
                if text_chunks:
                    messages.append({"role": role, "content": "\n".join(text_chunks)})

        if not messages:
            messages = [{"role": "user", "content": "Hello"}]
        return messages, images

    def _strip_code_fences(self, s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # remove starting ```... (with optional language)
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _strip_think_tags(self, s: str) -> str:
        # Remove any <think>...</think> blocks
        return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

    def _trim_to_json_object(self, s: str) -> str:
        # Keep only the substring from the first { to the last }
        start = s.find("{")
        end = s.rfind("}")
        return s[start:end + 1] if start != -1 and end != -1 and end > start else s

    def _json5_to_json(self, s: str) -> str:
        # 1) quote unquoted keys: key: value -> "key": value (avoid already-quoted keys)
        s = re.sub(r'(?<!")\b([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'"\1":', s)
        # 2) replace single-quoted string values with double quotes
        s = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r': "\1"', s)
        # 3) remove trailing commas before } or ]
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    def _extract_first_json(self, raw: str) -> Optional[dict]:
        """
        Tries to parse a JSON object from a model output that may contain <think> tags,
        code fences, or JSON5-like syntax (unquoted keys, single quotes, trailing commas).
        """
        if not raw:
            return None

        # Remove <think> content and code fences
        text = self._strip_think_tags(raw)
        text = self._strip_code_fences(text).strip()

        # Heuristic: trim to the first JSON object block
        text = self._trim_to_json_object(text)

        # Try strict JSON first
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to normalize JSON5-ish to strict JSON
        cleaned = self._json5_to_json(text)
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # As a last resort, scan for a balanced {...} block and normalize it
        start = raw.find("{")
        while start != -1:
            depth = 0
            for end in range(start, len(raw)):
                if raw[end] == "{":
                    depth += 1
                elif raw[end] == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = raw[start:end + 1]
                        chunk = self._strip_think_tags(self._strip_code_fences(chunk))
                        chunk = self._json5_to_json(self._trim_to_json_object(chunk))
                        try:
                            return json.loads(chunk)
                        except Exception:
                            break
            start = raw.find("{", start + 1)

        return None

    def _parse_tool_call(self, raw_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Accepts either:
         - {"name":"computer","arguments":{...}}
         - {"tool_calls":[{"type":"function","function":{"name":"computer","arguments":{...}}}]}
        Returns (name, arguments) or None.
        """
        data = self._extract_first_json(raw_text or "")
        if not isinstance(data, dict):
            return None

        # Direct form
        if data.get("name") == "computer" and isinstance(data.get("arguments"), dict):
            return "computer", data["arguments"]

        # tool_calls form
        if "tool_calls" in data and isinstance(data["tool_calls"], list) and data["tool_calls"]:
            tc = data["tool_calls"][0]
            if (
                isinstance(tc, dict)
                and tc.get("type") == "function"
                and isinstance(tc.get("function"), dict)
                and tc["function"].get("name") == "computer"
                and isinstance(tc["function"].get("arguments"), dict)
            ):
                return "computer", tc["function"]["arguments"]

        # Loose fallback: if top-level looks like arguments only with action, treat as arguments
        if "action" in data:
            return "computer", data
        return None

    def _normalize_keys(self, d: Dict[str, Any]) -> Dict[str, Any]:
        # Minor normalizations: allow hyphen or plus separators in keys string
        dd = dict(d)
        if "keys" in dd and isinstance(dd["keys"], str):
            s = dd["keys"].replace("-", "+")
            dd["keys"] = [k for k in s.split("+") if k]
        return dd

    def _arguments_to_computer_call(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Map arguments to a single computer_call item.
        """
        args = self._normalize_keys(args)
        action = (args.get("action") or "").strip().lower()

        if action == "click":
            x, y = int(args.get("x", 0)), int(args.get("y", 0))
            button = str(args.get("button", "left"))
            return make_click_item(x, y, button)  # left/right/wheel supported

        if action == "double_click":
            x, y = int(args.get("x", 0)), int(args.get("y", 0))
            return make_double_click_item(x, y)

        if action == "move":
            x, y = int(args.get("x", 0)), int(args.get("y", 0))
            return make_move_item(x, y)

        if action == "scroll":
            x, y = int(args.get("x", 0)), int(args.get("y", 0))
            sx, sy = int(args.get("scroll_x", 0)), int(args.get("scroll_y", 0))
            return make_scroll_item(x, y, sx, sy)

        if action == "type":
            text = str(args.get("text", ""))
            return make_type_item(text)

        if action == "keypress":
            keys = args.get("keys", [])
            if isinstance(keys, str):
                keys = [k for k in keys.replace("-", "+").split("+") if k]
            return make_keypress_item(keys)

        if action == "drag":
            path = args.get("path")
            if isinstance(path, list) and len(path) >= 2 and all(isinstance(p, dict) for p in path):
                return make_drag_item([{"x": int(p.get("x", 0)), "y": int(p.get("y", 0))} for p in path])
            # fallback: start/end fields
            sx, sy = args.get("start_x"), args.get("start_y")
            ex, ey = args.get("end_x"), args.get("end_y")
            if sx is not None and sy is not None and ex is not None and ey is not None:
                return make_drag_item([{"x": int(sx), "y": int(sy)}, {"x": int(ex), "y": int(ey)}])

        if action == "wait":
            return make_wait_item()

        if action == "screenshot":
            return make_screenshot_item()

        return None

    # ------------------------------ Loop API ------------------------------

    async def predict_step(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        use_prompt_caching: Optional[bool] = False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        hf_path = self._resolve_hf_path(model)
        self._ensure_loaded(hf_path)

        # Ensure a screenshot is present for grounding
        have_image = self._messages_have_image(messages)
        augmented_messages = list(messages)
        if not have_image and computer_handler is not None:
            try:
                screenshot_b64 = await computer_handler.screenshot()
                if _on_screenshot:
                    await _on_screenshot(screenshot_b64, "screenshot_before")
                augmented_messages.append(
                    {
                        "role": "user",
                        "type": "message",
                        "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}],
                    }
                )
            except Exception:
                pass

        # Convert to Perceptron chat+images
        p_msgs, images = self._to_perceptron_inputs(augmented_messages)

        # Prepend the action-space instruction (as a user message to keep it in-context)
        p_msgs = [{"role": "user", "content": self._action_prompt}] + p_msgs

        # Build chat text
        text = self._processor.apply_chat_template(p_msgs, tokenize=False, add_generation_prompt=True)

        # Prepare Isaac inputs
        inputs = self._processor(text=text, images=images, return_tensors="pt")
        tensor_stream = inputs["tensor_stream"].to(self._device)
        input_ids = inputs["input_ids"].to(self._device)

        if _on_api_start:
            await _on_api_start(
                {"provider": "perceptron-hf", "model": hf_path, "max_new_tokens": kwargs.get("max_new_tokens", 512)}
            )

        # Generate JSON-only tool call
        with torch.no_grad():
            generated_ids = self._model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                do_sample=False,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

        if _on_api_end:
            await _on_api_end(
                {"provider": "perceptron-hf", "model": hf_path},
                {"generated_len": int(generated_ids.shape[1])},
            )

        tok = getattr(self._processor, "tokenizer", None)
        if tok is not None and generated_ids.shape[1] > input_ids.shape[1]:
            new_tokens = generated_ids[0, input_ids.shape[1] :]
            raw_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            raw_text = tok.decode(generated_ids[0], skip_special_tokens=True).strip() if tok else ""

        # Parse tool call
        tool = self._parse_tool_call(raw_text)
        output_items: List[Dict[str, Any]] = []

        if tool is not None:
            name, args = tool
            if name == "computer":
                computer_call = self._arguments_to_computer_call(args or {})
                if computer_call is not None:
                    output_items.append(computer_call)

        # Fallback to plain text if no valid tool call
        if not output_items:
            output_items.append(make_output_text_item(raw_text))

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "response_cost": 0.0}
        if _on_usage:
            await _on_usage(usage)

        return {"output": output_items, "usage": usage}

    async def predict_click(
        self, model: str, image_b64: str, instruction: str, **kwargs: Any
    ) -> Optional[Tuple[int, int]]:
        # Not implementing click-only mode yet
        return None

    def get_capabilities(self) -> List[str]:
        return ["step"]

async def run_agent_example():
    """Run example of using the ComputerAgent with different models."""
    print("\n=== Example: ComputerAgent with different models ===")

    try:
        # Create a local macOS computer
        computer = Computer(
            use_host_computer_server=True,
            host="gtzs.local",
            os_type="linux",
            verbosity=logging.DEBUG,
        )

        # Create a remote Linux computer with Cua
        # computer = Computer(
        #     os_type="linux",
        #     api_key=os.getenv("CUA_API_KEY"),
        #     name=os.getenv("CUA_CONTAINER_NAME"),
        #     provider_type=VMProviderType.CLOUD,
        # )

        # Create ComputerAgent with new API
        agent = ComputerAgent(
            # Supported models:
            
            # == OpenAI CUA (computer-use-preview) ==
            # model="openai/computer-use-preview",

            # == Anthropic CUA (Claude > 3.5) ==
            # model="anthropic/claude-opus-4-20250514", 
            # model="anthropic/claude-sonnet-4-20250514",
            # model="anthropic/claude-3-7-sonnet-20250219",
            # model="anthropic/claude-3-5-sonnet-20241022",

            # == UI-TARS ==
            # model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B",
            # model="mlx/mlx-community/UI-TARS-1.5-7B-6bit",
            # model="ollama_chat/0000/ui-tars-1.5-7b",

            # == Omniparser + Any LLM ==
            # model="omniparser+anthropic/claude-opus-4-20250514",
            # model="omniparser+ollama_chat/gemma3:12b-it-q4_K_M",

            model="perceptron/isaac-0.1",

            tools=[computer],
            only_n_most_recent_images=3,
            verbosity=logging.DEBUG,
            trajectory_dir="trajectories",
            use_prompt_caching=True,
            max_trajectory_budget=1.0,
        )

        # Example tasks to demonstrate the agent
        tasks = [
            #"Look for a repository named trycua/cua on GitHub.",
            #"Check the open issues, open the most recent one and read it.",
            #"Clone the repository in users/lume/projects if it doesn't exist yet.",
            #"Open the repository with an app named Cursor (on the dock, black background and white cube icon).",
            #"From Cursor, open Composer if not already open.",
            #"Focus on the Composer text area, then write and submit a task to help resolve the GitHub issue.",
            'Identify the center of the Firefox icon and move the cursor to those coordinates',
            'Double-click on the Firefox icon',
            'Click on the Applications menu'
        ]

        # Use message-based conversation history
        history = []
        
        for i, task in enumerate(tasks):
            print(f"\nExecuting task {i+1}/{len(tasks)}: {task}")
            
            # Add user message to history
            history.append({"role": "user", "content": task})
            
            # Run agent with conversation history
            async for result in agent.run(history, stream=False):
                # Add agent outputs to history
                history += result.get("output", [])
                
                # Print output for debugging
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for content_part in content:
                            if content_part.get("text"):
                                print(f"Agent: {content_part.get('text')}")
                    elif item.get("type") == "computer_call":
                        action = item.get("action", {})
                        action_type = action.get("type", "")
                        print(f"Computer Action: {action_type}({action})")
                    elif item.get("type") == "computer_call_output":
                        print("Computer Output: [Screenshot/Result]")
                        
            print(f"✅ Task {i+1}/{len(tasks)} completed: {task}")

    except Exception as e:
        logger.error(f"Error in run_agent_example: {e}")
        traceback.print_exc()
        raise

def main():
    """Run the Anthropic agent example."""
    try:
        load_dotenv_files()

        # Register signal handler for graceful exit
        signal.signal(signal.SIGINT, handle_sigint)

        asyncio.run(run_agent_example())
    except Exception as e:
        print(f"Error running example: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()