"""
NeuroVia Base Agent — Foundation for all specialist agents.
Defines the agent protocol, message types, and shared infrastructure.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from abc import ABC, abstractmethod

from neurovia_agents.llm_client import get_llm_client, LLMResponse

log = logging.getLogger("neurovia.agents")


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentMessage:
    """A message passed between agents or to/from the LLM."""
    role: MessageRole
    content: str
    agent_name: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """The outcome of an agent's work."""
    success: bool
    output: Any  # The structured result (dict, DataFrame reference, etc.)
    summary: str  # Human-readable summary
    agent_name: str = ""
    duration: float = 0.0
    llm_provider: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """Represents a tool the agent wants to invoke."""
    name: str
    arguments: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all NeuroVia agents.
    
    Each agent has:
      - A name and role description
      - A system prompt that defines its expertise
      - A set of tools it can call (Python functions on the DataFrame)
      - A think→act→observe loop backed by the LLM
    """

    def __init__(self, name: str, role: str, system_prompt: str,
                 max_iterations: int = 5):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.status = AgentStatus.IDLE
        self._history: list[dict] = []
        self._tools: dict[str, callable] = {}
        self._status_callback: Optional[callable] = None

        # Register tools from subclass
        self._register_tools()

    @abstractmethod
    def _register_tools(self):
        """Subclasses register their available tools here."""
        ...

    @abstractmethod
    def _build_task_prompt(self, context: dict) -> str:
        """Build the user prompt for the specific task given context."""
        ...

    def register_tool(self, name: str, func: callable, description: str,
                      parameters: dict):
        """Register a tool the agent can call."""
        self._tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters,
        }

    def on_status_change(self, callback: callable):
        """Register a callback for status updates: callback(agent_name, status, detail)."""
        self._status_callback = callback

    def _emit_status(self, status: AgentStatus, detail: str = ""):
        self.status = status
        if self._status_callback:
            self._status_callback(self.name, status, detail)

    def _get_tools_description(self) -> str:
        """Format tool descriptions for the system prompt."""
        if not self._tools:
            return "No tools available."
        lines = []
        for name, spec in self._tools.items():
            params = json.dumps(spec["parameters"], indent=2)
            lines.append(f"### {name}\n{spec['description']}\nParameters:\n```json\n{params}\n```")
        return "\n\n".join(lines)

    def _build_system_message(self) -> str:
        tools_desc = self._get_tools_description()
        return f"""{self.system_prompt}

## Available Tools
{tools_desc}

## Response Format
Always respond with valid JSON in this exact structure:
{{
  "thinking": "Your reasoning about the task",
  "action": {{
    "tool": "tool_name or FINISH",
    "arguments": {{}},
    "explanation": "Why you chose this action"
  }},
  "summary": "Brief human-readable summary of current step"
}}

When you have gathered enough information or completed the task, use tool="FINISH" and put your final result in arguments.result.
"""

    def run(self, context: dict) -> AgentResult:
        """
        Execute the agent's think→act→observe loop.
        
        Args:
            context: Dict with at minimum {"dataframe": pd.DataFrame}
                     May contain other keys from previous agents.
        
        Returns:
            AgentResult with the outcome
        """
        start = time.time()
        self._emit_status(AgentStatus.THINKING, "Starting analysis...")

        client = get_llm_client()
        self._history = [
            {"role": "system", "content": self._build_system_message()},
            {"role": "user", "content": self._build_task_prompt(context)},
        ]

        last_result = None

        for iteration in range(self.max_iterations):
            self._emit_status(AgentStatus.THINKING, f"Iteration {iteration + 1}/{self.max_iterations}")

            try:
                response = client.complete(
                    messages=self._history,
                    temperature=0.2,
                    json_mode=True,
                )
            except RuntimeError as e:
                self._emit_status(AgentStatus.ERROR, str(e))
                return AgentResult(
                    success=False, output=None, summary=f"LLM call failed: {e}",
                    agent_name=self.name, duration=time.time() - start,
                )

            # Parse LLM response
            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                parsed = self._extract_json(response.content)
                if parsed is None:
                    self._history.append({"role": "assistant", "content": response.content})
                    self._history.append({
                        "role": "user",
                        "content": "Your response was not valid JSON. Please respond with the exact JSON format specified."
                    })
                    continue

            self._history.append({"role": "assistant", "content": json.dumps(parsed)})

            action = parsed.get("action", {})
            tool_name = action.get("tool", "FINISH")
            arguments = action.get("arguments", {})
            summary = parsed.get("summary", "")

            # Check if done
            if tool_name == "FINISH":
                self._emit_status(AgentStatus.DONE, summary)
                return AgentResult(
                    success=True,
                    output=arguments.get("result", parsed),
                    summary=summary or parsed.get("thinking", "Completed."),
                    agent_name=self.name,
                    duration=time.time() - start,
                    llm_provider=response.provider,
                    metadata={"iterations": iteration + 1},
                )

            # Execute tool
            if tool_name in self._tools:
                self._emit_status(AgentStatus.EXECUTING, f"Running {tool_name}...")
                try:
                    tool_result = self._tools[tool_name]["function"](context, **arguments)
                    observation = json.dumps(tool_result, default=str)
                except Exception as e:
                    observation = json.dumps({"error": str(e)})

                self._history.append({
                    "role": "user",
                    "content": f"Tool `{tool_name}` returned:\n```json\n{observation}\n```\nContinue your analysis."
                })
                last_result = tool_result
            else:
                self._history.append({
                    "role": "user",
                    "content": f"Unknown tool `{tool_name}`. Available: {list(self._tools.keys())}. Use FINISH when done."
                })

        # Max iterations reached
        self._emit_status(AgentStatus.DONE, "Reached max iterations")
        return AgentResult(
            success=True,
            output=last_result,
            summary="Completed (max iterations reached).",
            agent_name=self.name,
            duration=time.time() - start,
            metadata={"iterations": self.max_iterations, "truncated": True},
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Try to extract a JSON object from text that may have extra content."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
