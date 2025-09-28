"""Example demonstrating the ComputerAgent capabilities with the Omni provider."""

import asyncio
import logging
import traceback
import signal

from computer import Computer, VMProviderType

# Import the unified agent class and types
from agent import ComputerAgent

# Import utility functions
from utils import load_dotenv_files, handle_sigint

# Host computer server monkeypatch-related
from computer.interface.factory import InterfaceFactory as _IF

# TrajectorySaver-related
from typing import Any, Dict, List, Optional, Union, override
from agent.callbacks.trajectory_saver import TrajectorySaverCallback

# Perceptron-related
import os
import io
import base64
import argparse
from PIL import Image, ImageDraw
try:
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
    from perceptron.pointing.parser import extract_points
    HF_AVAILABLE = True
except Exception as _e:
    HF_AVAILABLE = False

# Ollama-related
import re
try:
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single, static prompts to use for the entire run (can override via env)
REDACTION_IMAGE_PROMPT = os.getenv(
    "REDACTION_IMAGE_PROMPT",
    "<hint>BBOX</hint> Identify and outline the various icons visible on the desktop."
)
REDACTION_TEXT_PROMPT = os.getenv(
    "REDACTION_TEXT_PROMPT",
    "What follows is text returned from an LLM.  Replace any instances of the word 'GitHub' or any URLs with '[REDACTED]'. Make no other changes."
)

# Hugging Face model repo (override via env if needed)
HF_ISAAC_PATH = os.getenv("PERC_HF_PATH", "PerceptronAI/Isaac-0.1")

# Choose the Ollama model:tag once (override via env OLLAMA_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")  # example: "gemma3:8b", "llama3:latest"

# Keys that should not be altered in metadata (model naming, loop identifier)
_TEXT_REDACT_SKIP_KEYS = {"model", "agent_loop", "role", "type", "finish_reason"}

# Global state
_ollama_client = None
_isaac_model = None
_isaac_processor = None
_isaac_config = None
_isaac_loaded = False
_no_flash_attention = False
_force_host_address = False
_force_host = None

# Host computer server monkeypatch
_original_create_interface_for_os = _IF.create_interface_for_os

def _patch_host_address(os, ip_address, api_key=None, vm_name=None):
    ip_address_to_use = _force_host or ip_address
    return _original_create_interface_for_os(os, ip_address_to_use, api_key=api_key, vm_name=vm_name)

_IF.create_interface_for_os = staticmethod(_patch_host_address)

def get_ollama_client(host: Optional[str] = None):
    """Get or create Ollama client."""
    global _ollama_client
    if _ollama_client is None:
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama Python library not available. Install with `pip install ollama`.")
        _ollama_client = OllamaClient(host=host) if host else OllamaClient()
    return _ollama_client

def redact_text_with_ollama(text: str, model: str = OLLAMA_MODEL, prompt: str = REDACTION_TEXT_PROMPT, host: Optional[str] = None) -> str:
    """Redact text using Ollama."""
    if not text:
        return text
    
    client = get_ollama_client(host)
    req = {
        "model": model,
        "prompt": f"{prompt}\n\n{text}",
        "stream": False,
    }
    try:
        resp = client.generate(**req)
        out = resp.get("response") or ""
        return out.strip()
    except Exception as e:
        logger.warning(f"Ollama redaction failed (returning original text): {e}")
        return text

def _is_probably_image_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:image/")

def _is_probably_base64(s: str) -> bool:
    """Check if string is likely base64 encoded data."""
    if not isinstance(s, str) or len(s) < 100:  # Too short to be meaningful base64
        return False
    
    # Base64 should only contain these characters
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
        return False
    
    # Base64 length should be divisible by 4 (after padding)
    if len(s) % 4 != 0:
        return False
    
    # If it's very long and matches base64 pattern, probably is base64
    return len(s) > 1000

def _should_skip_string(s: str) -> bool:
    return _is_probably_image_data_url(s) or _is_probably_base64(s)

async def redact_text_tree(
    obj: Any,
    model: str = OLLAMA_MODEL,
    prompt: str = REDACTION_TEXT_PROMPT,
    host: Optional[str] = None,
    skip_keys: set[str] = _TEXT_REDACT_SKIP_KEYS,
    _parent_key: Optional[str] = None,
) -> Any:
    """
    Walks a nested Python object (dict/list/str/etc.), redacting any strings via Ollama.
    Skips strings that look like base64 images or are overly large.
    """
    loop = asyncio.get_event_loop()

    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            # avoid redacting values of specific structural keys (like model)
            if isinstance(k, str) and k in skip_keys and isinstance(v, str):
                new_d[k] = v
                continue
            new_d[k] = await redact_text_tree(v, model, prompt, host, skip_keys, _parent_key=k)
        return new_d

    if isinstance(obj, list):
        return [await redact_text_tree(v, model, prompt, host, skip_keys, _parent_key) for v in obj]

    if isinstance(obj, str):
        if _should_skip_string(obj):
            return obj
        try:
            # Run Ollama redaction in a thread to avoid blocking the event loop
            return await loop.run_in_executor(None, redact_text_with_ollama, obj, model, prompt, host)
        except Exception as e:
            logger.warning(f"Text redaction failed on string (leaving original): {e}")
            return obj

    # Primitives / other types unchanged
    return obj

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

def ensure_isaac_loaded(hf_path: str = HF_ISAAC_PATH, device: Optional[str] = None):
    """Ensure Isaac model is loaded globally."""
    global _isaac_model, _isaac_processor, _isaac_config, _isaac_loaded, _no_flash_attention
    
    if _isaac_loaded:
        return
    
    if not HF_AVAILABLE:
        raise RuntimeError("HuggingFace/Perceptron imports unavailable; cannot run local redaction.")
    
    logger.info(f"Loading Isaac model and processor from '{hf_path}'")
    _isaac_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    _isaac_processor = AutoProcessor.from_pretrained(hf_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)

    # Apply the attention patch
    if _no_flash_attention:
        logger.info("Attempting to monkeypatch Siglip2VariableLengthAttention")
        if not apply_attention_patch(model):
            logger.warning("No monkeypatching occurred!")

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if (device_obj.type == "cuda") else torch.float32

    _isaac_model = model.to(device=device_obj, dtype=dtype)
    _isaac_model.eval()
    logger.info(f"Isaac loaded on {device_obj} (dtype={dtype})")
    _isaac_loaded = True

def generate_isaac_text(image: Image.Image, prompt: str) -> str:
    """Generate text using Isaac model for given image and prompt."""
    ensure_isaac_loaded()
    
    vision_token = getattr(_isaac_config, "vision_token", "<image>")
    messages = [
        {"role": "user", "content": vision_token},
        {"role": "user", "content": prompt},
    ]
    chat_text = _isaac_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _isaac_processor(text=chat_text, images=[image], return_tensors="pt")
    tensor_stream = inputs["tensor_stream"].to(next(_isaac_model.parameters()).device)

    with torch.no_grad():
        generated_ids = _isaac_model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=_isaac_processor.tokenizer.eos_token_id,
            eos_token_id=_isaac_processor.tokenizer.eos_token_id,
        )

    # Decode full output
    return _isaac_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

def scale_norm_to_px(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    """Convert Isaac's 0-1000 normalized coords to pixel space."""
    X = int(round((x / 1000.0) * w))
    Y = int(round((y / 1000.0) * h))
    return max(0, min(w - 1, X)), max(0, min(h - 1, Y))

def blacken_regions(image: Image.Image, tags_text: str) -> Image.Image:
    """Parse all canonical tags and draw filled black overlays."""
    w, h = image.size
    draw = ImageDraw.Draw(image)

    # Flatten everything returned (points, boxes, polygons, and within collections)
    regions = extract_points(tags_text, expected=None)

    for obj in regions:
        cls_name = obj.__class__.__name__
        if cls_name == "SinglePoint":
            x, y = scale_norm_to_px(obj.x, obj.y, w, h)
            # small black square around point
            r = 10  # radius; adjust as needed
            draw.rectangle([x - r, y - r, x + r, y + r], fill="black")
        elif cls_name == "BoundingBox":
            x1, y1 = scale_norm_to_px(obj.top_left.x, obj.top_left.y, w, h)
            x2, y2 = scale_norm_to_px(obj.bottom_right.x, obj.bottom_right.y, w, h)
            draw.rectangle([x1, y1, x2, y2], fill="black")
        elif cls_name == "Polygon":
            pts = [scale_norm_to_px(p.x, p.y, w, h) for p in obj.hull]
            if len(pts) >= 3:
                draw.polygon(pts, fill="black")
        # Collections are already flattened by extract_points → ignored here explicitly

    return image

def redact_image_with_isaac(image_bytes: bytes, prompt: str = REDACTION_IMAGE_PROMPT) -> bytes:
    """
    Redact image using Isaac model.
    - Bytes → PIL → generate → parse → blacken → return PNG bytes
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # If decode fails, just return the original bytes
        return image_bytes

    generated_text = generate_isaac_text(pil, prompt)
    logger.info(f"\nFull generated output:\n{generated_text}")
    redacted = blacken_regions(pil.copy(), generated_text)

    out = io.BytesIO()
    redacted.save(out, format="PNG")
    return out.getvalue()

# -----------------------------------------------------------------------------
# Combined callback: image redaction (Perceptron) + text redaction (Ollama)
# -----------------------------------------------------------------------------
class CustomTrajectorySaverCallback(TrajectorySaverCallback):
    """
    Trajectory saver that redacts screenshots locally via Perceptron Isaac
    and redacts text locally via Ollama before saving.
    """

    def __init__(
        self,
        trajectory_dir: str,
        *,
        reset_on_run: bool = True,
        screenshot_dir: Optional[str] = None,
        # Perceptron-related args
        hf_path: str = HF_ISAAC_PATH,
        image_prompt: str = REDACTION_IMAGE_PROMPT,
        device: Optional[str] = None,
        # Ollama text redaction args
        ollama_model: str = OLLAMA_MODEL,
        text_prompt: str = REDACTION_TEXT_PROMPT,
        ollama_host: Optional[str] = None,
    ):
        super().__init__(trajectory_dir, reset_on_run=reset_on_run, screenshot_dir=screenshot_dir)

        # Store configuration for the procedural functions
        self._hf_path = hf_path
        self._image_prompt = image_prompt
        self._device = device
        self._ollama_model = ollama_model
        self._text_prompt = text_prompt
        self._ollama_host = ollama_host

        logger.info("CustomTrajectorySaverCallback initialized (with Ollama and Perceptron redaction)")

    async def _redact_bytes_async(self, data: bytes) -> bytes:
        """Redact image bytes asynchronously."""
        loop = asyncio.get_event_loop()
        # Ensure Isaac is loaded with our config
        ensure_isaac_loaded(self._hf_path, self._device)
        return await loop.run_in_executor(None, redact_image_with_isaac, data, self._image_prompt)

    @override
    async def on_screenshot(self, screenshot: Union[str, bytes], name: str = "screenshot") -> None:
        """Intercept screenshots before they're written; mask sensitive regions; save only redacted images."""
        try:
            if isinstance(screenshot, str):
                # screenshot from ComputerAgent is base64 string (no data: prefix)
                raw = base64.b64decode(screenshot)
            else:
                raw = screenshot

            redacted = await self._redact_bytes_async(raw)
            await super().on_screenshot(redacted, name)
        except Exception as e:
            logger.warning(f"Redaction failed in on_screenshot ({name}), writing original: {e}")
            await super().on_screenshot(screenshot, name)

    @override
    async def on_run_start(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]]) -> None:
        """Base saver writes metadata.json on run start. We pass a redacted copy of kwargs."""
        try:
            redacted_kwargs = await redact_text_tree(
                kwargs, self._ollama_model, self._text_prompt, self._ollama_host
            )
        except Exception as e:
            logger.warning(f"Ollama text redaction failed in on_run_start: {e}")
            redacted_kwargs = kwargs
        await super().on_run_start(redacted_kwargs, old_items)

    @override
    async def on_api_start(self, kwargs: Dict[str, Any]) -> None:
        """base _save_artifact('api_start', { 'kwargs': kwargs }) - pass a redacted copy."""
        try:
            redacted_kwargs = await redact_text_tree(
                kwargs, self._ollama_model, self._text_prompt, self._ollama_host
            )
        except Exception as e:
            logger.warning(f"Ollama text redaction failed in on_api_start: {e}")
            redacted_kwargs = kwargs
        await super().on_api_start(redacted_kwargs)

    @override
    async def on_api_end(self, kwargs: Dict[str, Any], result: Any) -> None:
        """base _save_artifact('api_result', { 'kwargs': kwargs, 'result': result }) - pass redacted copies."""
        try:
            redacted_kwargs = await redact_text_tree(
                kwargs, self._ollama_model, self._text_prompt, self._ollama_host
            )
            redacted_result = await redact_text_tree(
                result, self._ollama_model, self._text_prompt, self._ollama_host
            )
        except Exception as e:
            logger.warning(f"Ollama text redaction failed in on_api_end: {e}")
            redacted_kwargs = kwargs
            redacted_result = result
        await super().on_api_end(redacted_kwargs, redacted_result)

    @override
    async def on_responses(self, kwargs: Dict[str, Any], responses: Dict[str, Any]) -> None:
        """base _save_artifact('agent_response', { 'kwargs': kwargs, 'response': responses }) - pass redacted copies."""
        try:
            redacted_kwargs = await redact_text_tree(
                kwargs, self._ollama_model, self._text_prompt, self._ollama_host
            )
            redacted_responses = await redact_text_tree(
                responses, self._ollama_model, self._text_prompt, self._ollama_host
            )
        except Exception as e:
            logger.warning(f"Ollama text redaction failed in on_responses: {e}")
            redacted_kwargs = kwargs
            redacted_responses = responses
        await super().on_responses(redacted_kwargs, redacted_responses)

    @override
    async def on_run_end(self, kwargs: Dict[str, Any], old_items: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> None:
        """Base saver amends metadata.json with completion status, total usage, and new_items."""
        try:
            redacted_kwargs = await redact_text_tree(
                kwargs, self._ollama_model, self._text_prompt, self._ollama_host
            )
            redacted_new_items = await redact_text_tree(
                new_items, self._ollama_model, self._text_prompt, self._ollama_host
            )
        except Exception as e:
            logger.warning(f"Ollama text redaction failed in on_run_end: {e}")
            redacted_kwargs = kwargs
            redacted_new_items = new_items
        await super().on_run_end(redacted_kwargs, old_items, redacted_new_items)

    @override
    async def on_computer_call_end(self, item: Dict[str, Any], result: List[Dict[str, Any]]) -> None:
        """Ensure the computer_call_output image embedded in the step result is redacted."""
        try:
            # Find the first computer_call_output with an image_url
            for r in result:
                if (
                    isinstance(r, dict)
                    and r.get("type") == "computer_call_output"
                    and isinstance(r.get("output"), dict)
                    and r["output"].get("type") == "input_image"
                ):
                    image_url = r["output"].get("image_url", "")
                    if isinstance(image_url, str) and image_url.startswith("data:image"):
                        header, b64 = image_url.split(",", 1)
                        raw = base64.b64decode(b64)
                        redacted = await self._redact_bytes_async(raw)
                        r["output"]["image_url"] = f"{header},{base64.b64encode(redacted).decode('ascii')}"
                    break
        except Exception as e:
            logger.warning(f"Redaction failed in on_computer_call_end, proceeding unredacted: {e}")

        await super().on_computer_call_end(item, result)

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

        global _force_host_address
        if _force_host_address:
            global _force_host
            _force_host = getattr(computer, "host", None)

        custom_saver = CustomTrajectorySaverCallback(
            trajectory_dir="trajectories",
            reset_on_run=True,
            #screenshot_dir="trajectories/screenshots"  # optional
        )

        # Create ComputerAgent with new API
        agent = ComputerAgent(
            # Supported models:
            
            # == OpenAI CUA (computer-use-preview) ==
            # model="openai/computer-use-preview",

            # == Anthropic CUA (Claude > 3.5) ==
            # model="anthropic/claude-opus-4-20250514", 
            model="anthropic/claude-sonnet-4-20250514",
            # model="anthropic/claude-3-7-sonnet-20250219",
            # model="anthropic/claude-3-5-sonnet-20241022",

            # == UI-TARS ==
            # model="huggingface-local/ByteDance-Seed/UI-TARS-1.5-7B",
            # model="mlx/mlx-community/UI-TARS-1.5-7B-6bit",
            # model="ollama_chat/0000/ui-tars-1.5-7b",

            # == Omniparser + Any LLM ==
            # model="omniparser+anthropic/claude-opus-4-20250514",
            # model="omniparser+ollama_chat/gemma3:12b-it-q4_K_M",

            tools=[computer],
            only_n_most_recent_images=3,
            verbosity=logging.DEBUG,
            #trajectory_dir="trajectories",
            use_prompt_caching=True,
            max_trajectory_budget=1.0,
            callbacks=[custom_saver],
        )

        # Example tasks to demonstrate the agent
        tasks = [
            "Look for a repository named trycua/cua on GitHub.",
            "Check the open issues, open the most recent one and read it.",
            "Clone the repository in users/lume/projects if it doesn't exist yet.",
            "Open the repository with an app named Cursor (on the dock, black background and white cube icon).",
            "From Cursor, open Composer if not already open.",
            "Focus on the Composer text area, then write and submit a task to help resolve the GitHub issue.",
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
    parser = argparse.ArgumentParser(description="Run ComputerAgent with redaction")
    parser.add_argument("-nfa", "--no-flash-attention", action="store_true", 
                       help="Disable flash attention for Isaac model")
    parser.add_argument("-fha", "--force-host-address", action="store_true", 
                       help="Force the use of a specified address for a host computer server")
    args = parser.parse_args()
    
    # Store the patch flag globally so callbacks can access it
    global _no_flash_attention
    global _force_host_address
    _no_flash_attention = args.no_flash_attention
    _force_host_address = args.force_host_address

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