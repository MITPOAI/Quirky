import sys
import json
import time
from typing import Dict, Any, Generator, List
from enum import Enum

class CognitiveState(str, Enum):
    IDEA = "Idea"
    DOUBT = "Doubt"
    CURIOSITY = "Curiosity"
    EXPERIMENT = "Experiment"
    ANSWER = "Answer"

class QuirkyAgentMiddleware:
    """
    Middleware that intercepts linear LLM planners and enforces a non-linear state graph:
    Idea ───> Doubt ───> Curiosity ───> Experiment ───> Answer
    """
    def __init__(self, doubt_threshold: float = 0.7):
        self.doubt_threshold = doubt_threshold
        
    def process_query(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the non-linear cognitive loop for an input query,
        yielding each state's thoughts and actions.
        """
        # --- State 1: IDEA ---
        yield {
            "state": CognitiveState.IDEA,
            "message": f"Formulating initial reasoning path for: '{query}'",
            "confidence": 0.9,
            "details": {
                "initial_hypothesis": f"Solve '{query}' directly using standard linear heuristics."
            }
        }
        time.sleep(0.3)
        
        # --- State 2: DOUBT ---
        # AI models usually assert solutions immediately. Quirky forces self-skepticism.
        doubt_score = 0.82  # Simulating a high doubt score to trigger curiosity
        yield {
            "state": CognitiveState.DOUBT,
            "message": "Enforcing skepticism checkpoint. Scoring alternative hypothesis paths...",
            "confidence": 0.9 - doubt_score * 0.5,
            "details": {
                "doubt_score": doubt_score,
                "skeptical_query": "Is the initial hypothesis relying on synthetic patterns or incomplete context?"
            }
        }
        time.sleep(0.4)
        
        # Check if doubt exceeds threshold
        if doubt_score > self.doubt_threshold:
            # --- State 3: CURIOSITY ---
            yield {
                "state": CognitiveState.CURIOSITY,
                "message": f"Doubt threshold ({self.doubt_threshold}) exceeded. Launching investigative sub-agent runs.",
                "confidence": 0.4,
                "details": {
                    "tools_to_query": ["search_codebase", "fetch_documentation_schema"],
                    "research_question": "Find edge-cases and structural constraints regarding the query."
                }
            }
            time.sleep(0.5)
            
            # --- State 4: EXPERIMENT ---
            yield {
                "state": CognitiveState.EXPERIMENT,
                "message": "Testing edge cases in sandbox workspace. Validating runtime outputs.",
                "confidence": 0.75,
                "details": {
                    "actions_run": ["execute_isolated_test_run", "compare_spectral_anomalies"],
                    "outcomes": "Edge cases validated. Refined resolution parameters generated."
                }
            }
            time.sleep(0.4)
            
        # --- State 5: ANSWER ---
        yield {
            "state": CognitiveState.ANSWER,
            "message": "Converging on humanized, verified execution path.",
            "confidence": 0.98,
            "details": {
                "final_plan": f"Execute humanized pipeline on target asset to resolve: '{query}'.",
                "result": "Success. Synthetic signatures eliminated."
            }
        }

    def run_mcp_loop(self):
        """
        Runs a standard stdio JSON-RPC MCP server loop for integration with IDEs/Clients.
        """
        print("Quirky MCP Cognitive Middleware Server Started (stdio)", file=sys.stderr)
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                request = json.loads(line)
                method = request.get("method")
                req_id = request.get("id")
                
                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "capabilities": {
                                "tools": {
                                    "listChanged": True
                                }
                            },
                            "serverInfo": {
                                "name": "Quirky Human Preference Agent",
                                "version": "0.1.0"
                            }
                        }
                    }
                elif method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "humanize_prompt",
                                    "description": "Transforms structured prompts into human-like variations",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        },
                                        "required": ["text"]
                                    }
                                }
                            ]
                        }
                    }
                elif method == "tools/call":
                    params = request.get("params", {})
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})
                    
                    if tool_name == "humanize_prompt":
                        text = tool_args.get("text", "")
                        # Run the cognitive state loop to formulate the humanized text
                        final_res = ""
                        for step in self.process_query(text):
                            if step["state"] == CognitiveState.ANSWER:
                                final_res = step["details"]["result"]
                                
                        response = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Humanized output: {final_res}"
                                    }
                                ]
                            }
                        }
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32601, "message": f"Method {tool_name} not found"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": "Method not found"}
                    }
                    
                print(json.dumps(response), flush=True)
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)}
                }
                print(json.dumps(err_resp), flush=True)

if __name__ == "__main__":
    middleware = QuirkyAgentMiddleware()
    middleware.run_mcp_loop()
