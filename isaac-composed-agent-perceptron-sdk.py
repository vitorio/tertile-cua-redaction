"""Example demonstrating the ComputerAgent capabilities with Perceptron grounding.

This file registers a click-only loop for Perceptron models (e.g. "perceptron/isaac-0.1")
and shows two usages:
  - Click-only: call agent.predict_click(...) by providing image_b64
  - Full run: use a composed model "perceptron/...+<thinking-LLM>" so the built-in
    composed-grounded loop handles step planning and calls Perceptron for clicks.

Requirements:
  pip install perceptron
  export PERCEPTRON_PROVIDER=fal
  export FAL_KEY=...   (or PERCEPTRON_API_KEY=...)
"""

import asyncio
import logging
import traceback
import signal
import base64
from typing import Any, Dict, List, Optional, Tuple

# Perceptron SDK (pip install perceptron)
try:
    from perceptron import question as perceptron_question
except Exception:
    perceptron_question = None  # will raise at use time

from computer import Computer, VMProviderType
from agent import ComputerAgent, register_agent
from agent.types import AgentCapability

try:
    from agent.loops.base import AsyncAgentConfig  # type: ignore
except Exception:
    AsyncAgentConfig = object  # fallback

import json

def _perceptron_result_to_json(result: Any) -> str:
    """
    Convert a Perceptron result object into a JSON string with polygons/boxes/points.
    Returns: JSON str like:
      {
        "text": "...",
        "polygons": [ { "points": [{"x":..,"y":..}, ...], "mention": "..." }, ... ],
        "boxes": [ { "x1":..,"y1":..,"x2":..,"y2":..,"mention":"..." }, ... ],
        "points": [ { "x":..,"y":..,"mention":"..." }, ... ],
        "errors": [...]
      }
    """
    payload = {
        "text": getattr(result, "text", None),
        "polygons": [],
        "boxes": [],
        "points": [],
        "errors": getattr(result, "errors", []) or [],
    }
    pts = getattr(result, "points", None) or []
    for p in pts:
        # polygon
        if hasattr(p, "hull"):
            poly = {
                "points": [{"x": int(getattr(pt, "x")), "y": int(getattr(pt, "y"))} for pt in getattr(p, "hull")],
                "mention": getattr(p, "mention", None),
            }
            payload["polygons"].append(poly)
        # box
        elif hasattr(p, "top_left") and hasattr(p, "bottom_right"):
            tl = getattr(p, "top_left")
            br = getattr(p, "bottom_right")
            try:
                payload["boxes"].append({
                    "x1": int(getattr(tl, "x")), "y1": int(getattr(tl, "y")),
                    "x2": int(getattr(br, "x")), "y2": int(getattr(br, "y")),
                    "mention": getattr(p, "mention", None),
                })
            except Exception:
                continue
        # point
        elif hasattr(p, "x") and hasattr(p, "y"):
            try:
                payload["points"].append({
                    "x": int(getattr(p, "x")), "y": int(getattr(p, "y")),
                    "mention": getattr(p, "mention", None),
                })
            except Exception:
                continue
    return json.dumps(payload)

# -----------------------------------------------------------------------------
# Perceptron click-only loop
# -----------------------------------------------------------------------------
# IMPORTANT:
#  - Regex excludes '+' so composed models (e.g. "perceptron/...+...") are NOT matched here.
#  - Priority set lower than composed-grounded loop (which uses priority=1),
#    so composed model selection will prefer the built-in composed loop.
@register_agent(models=r"(?i)^perceptron/[^+]+$", priority=0)
class PerceptronClickOnlyConfig(AsyncAgentConfig):
    """Click-only loop for Perceptron models."""

    async def predict_step(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError("PerceptronClickOnlyConfig supports click prediction only.")

    async def predict_click(
        self,
        model: str,
        image_b64: str,
        instruction: str,
        **kwargs,
    ) -> Optional[Tuple[int, int]]:
        """Return (x, y) or None. Adds grounding-specific clarifier to avoid text labels."""
        if perceptron_question is None:
            raise RuntimeError(
                "Perceptron SDK not available. Install with: pip install perceptron "
                "and set PERCEPTRON_PROVIDER and FAL_KEY/PERCEPTRON_API_KEY."
            )

        # Decode base64 (handle both raw and data URLs)
        try:
            if image_b64.startswith("data:"):
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise ValueError(f"Invalid base64 image input for Perceptron: {e}")

        # Add a clarifier so the grounder aims at the glyph center, not the caption/label.
        clarified = (
            f"{instruction}. Click the center of the icon/button graphic (not the text label). "
            f"If the item has a text caption below it, target the glyph area above the text."
        )

        # 1) Try to get a point directly
        try:
            res_point = perceptron_question(image_bytes, clarified, expects="point", stream=False)
            coords = self._extract_first_point(res_point)
            if coords is not None:
                return coords
        except Exception:
            pass

        # 2) Fallback to expects='box' and use a slightly top-biased 'center'
        try:
            res_box = perceptron_question(image_bytes, clarified, expects="box", stream=False)
            coords = self._extract_biased_center_from_first_box(res_box, bias=0.45)
            if coords is not None:
                return coords
        except Exception:
            pass

        return None

    def get_capabilities(self) -> List[AgentCapability]:
        return ["click"]

    @staticmethod
    def _extract_first_point(result: Any) -> Optional[Tuple[int, int]]:
        pts = getattr(result, "points", None) or []
        for p in pts:
            if hasattr(p, "x") and hasattr(p, "y"):
                try:
                    return (int(getattr(p, "x")), int(getattr(p, "y")))
                except Exception:
                    continue
        return None

    @staticmethod
    def _extract_biased_center_from_first_box(result: Any, bias: float = 0.45) -> Optional[Tuple[int, int]]:
        """
        Return a 'center' with vertical bias (default 45% from top instead of 50%),
        helpful for icon+label cases where naive center lands on the caption.
        """
        pts = getattr(result, "points", None) or []
        for b in pts:
            tl = getattr(b, "top_left", None)
            br = getattr(b, "bottom_right", None)
            if tl is None or br is None:
                continue
            try:
                x1, y1 = int(getattr(tl, "x")), int(getattr(tl, "y"))
                x2, y2 = int(getattr(br, "x")), int(getattr(br, "y"))
                cx = (x1 + x2) // 2
                # bias vertical center toward the top of the box
                cy = int(round(y1 + (y2 - y1) * float(bias)))
                return (cx, cy)
            except Exception:
                continue
        return None

# -----------------------------------------------------------------------------
# Example driver
# -----------------------------------------------------------------------------
from utils import load_dotenv_files, handle_sigint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_agent_example():
    """Demonstrate two usages: click-only and composed full run."""
    print("\n=== Example: ComputerAgent with Perceptron grounding (click-only and composed) ===")

    try:
        # Always manage Computer lifecycle with async context
        async with Computer(
            use_host_computer_server=True,
            host="gtzs.local",
            os_type="linux",
            verbosity=logging.DEBUG,
        ) as computer:
            # ------------------ Perceptron segmentation tool ------------------
            # This closure becomes a "function tool" the LLM can call.
            # It returns polygon/box/point results as a JSON string.
            async def perceptron_segment(
                instruction: str,
                expects: str = "polygon",
                image_b64: Optional[str] = None,
            ) -> str:
                """
                Segment regions in the current screen using Perceptron.

                Parameters
                ----------
                instruction : str
                    What to segment (e.g., "Firefox icon", "the login form").
                expects : str
                    One of "polygon", "box", or "point" (default "polygon").
                image_b64 : str, optional
                    If provided, use this image instead of taking a new screenshot (base64 PNG).

                Returns
                -------
                str
                    JSON string with keys: text, polygons, boxes, points, errors.
                """
                if perceptron_question is None:
                    return json.dumps({"error": "Perceptron SDK not installed/configured"})

                # Obtain image bytes
                if image_b64:
                    try:
                        b64 = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
                        image_bytes = base64.b64decode(b64)
                    except Exception as e:
                        return json.dumps({"error": f"invalid image_b64: {e}"})
                else:
                    # Take a fresh screenshot from the live computer
                    img_bytes = await computer.interface.screenshot()
                    image_bytes = img_bytes  # already bytes

                # Normalize expects
                e = (expects or "polygon").lower()
                if e not in {"polygon", "box", "point"}:
                    e = "polygon"

                # Call Perceptron
                try:
                    # Add a small clarifier to bias toward glyphs rather than captions
                    clarified = (
                        f"{instruction}. Return the precise boundaries of the target. "
                        f"If there is a text label below the icon, segment the icon glyph region."
                    )
                    res = perceptron_question(image_bytes, clarified, expects=e, stream=False)
                    return _perceptron_result_to_json(res)
                except Exception as err:
                    return json.dumps({"error": str(err)})

            # -----------------------------------------------------------------
            # 1) Click-only usage: explicitly pass image_b64 to predict_click
            # -----------------------------------------------------------------
            agent_click_only = ComputerAgent(
                model="perceptron/isaac-0.1+anthropic/claude-sonnet-4-20250514",  # matches our click-only loop
                tools=[computer, perceptron_segment],
                verbosity=logging.DEBUG,
            )

            # Take a fresh screenshot and call predict_click
            img_bytes = await computer.interface.screenshot()
            image_b64 = base64.b64encode(img_bytes).decode("utf-8")

            coords = await agent_click_only.predict_click(
                instruction="Double-click the Firefox icon",
                image_b64=image_b64,
            )
            print(f"[Click-only] Predicted coords: {coords}")

            # -----------------------------------------------------------------
            # 2) Full run via composed model (Perceptron + LLM for planning)
            #    IMPORTANT: our loop no longer matches model strings with '+',
            #    so the built-in composed-grounded loop will take over.
            # -----------------------------------------------------------------
            model_string = "perceptron/isaac-0.1+anthropic/claude-sonnet-4-20250514"

            agent_full = ComputerAgent(
                model=model_string,
                tools=[computer],
                only_n_most_recent_images=3,
                verbosity=logging.DEBUG,
                trajectory_dir="trajectories",
                use_prompt_caching=False,
                max_trajectory_budget=1.0,
                # New: steer the thinking model’s element_description so grounding aims correctly
                instructions=(
                    "When producing computer actions that require element_description, describe the clickable "
                    "graphic area of the control/icon, not its text caption. Target the visual glyph center. "
                    "For desktop icons, target the icon glyph above its label text."
                    "You can call the `perceptron_segment` tool to get precise polygon/box boundaries "
                    "of UI elements from the current screen. Use it when you need accurate regions "
                    "before clicking, dragging, or highlighting. Prefer polygons for irregular shapes."
                ),
            )

            history: List[Dict[str, Any]] = []
            task = "Open a browser and navigate to github.com; search for 'trycua/cua'."
            print(f"\n[Composed] Executing task: {task}")
            history.append({"role": "user", "content": task})

            async for result in agent_full.run(history, stream=False):
                history += result.get("output", [])
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("text"):
                                print("Agent:", part.get("text"))
                    elif item.get("type") == "computer_call":
                        action = item.get("action", {})
                        print("Computer Action:", action.get("type"), action)
                    elif item.get("type") == "computer_call_output":
                        print("Computer Output: [Screenshot/Result]")

            print("✅ Composed task completed.")

    except Exception as e:
        logger.error(f"Error in run_agent_example: {e}")
        traceback.print_exc()
        raise


def main():
    try:
        load_dotenv_files()
        signal.signal(signal.SIGINT, handle_sigint)
        asyncio.run(run_agent_example())
    except Exception as e:
        print(f"Error running example: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
