import json
from openai import OpenAI
from tools import VideoTools
import time

class DVDAgent:
    def __init__(self, tools: VideoTools, video_length_secs: float, llm_client: OpenAI, model_name: str):
        self.tools = tools
        self.video_length_secs = video_length_secs
        self.client = llm_client
        self.model = model_name
        
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "clip_search",
                    "description": "Searches for events in a video clip database based on a given event description and retrieves relevant clip captions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_description": {"type": "string"},
                            "top_k": {"type": "integer"}
                        },
                        "required": ["event_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "global_browse",
                    "description": "Analyzes a video database to answer a detailed question by searching for relevant clips and generating a comprehensive answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "frame_inspect",
                    "description": "Extracts video frames based on exact time ranges. CRITICAL: Use the exact timestamps returned by clip_search or global_browse.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "time_ranges_hhmmss": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}}
                            }
                        },
                        "required": ["question", "time_ranges_hhmmss"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Call this function after confirming the answer to the user's question to finish the conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"}
                        },
                        "required": ["answer"]
                    }
                }
            }
        ]

    def run(self, question: str, max_steps: int = 10):
        system_prompt = (
            "You are a helpful assistant who answers multi-step questions by sequentially invoking functions. "
            "Follow the THINK -> ACT -> OBSERVE loop:\n"
            " - THOUGHT: Reason step-by-step about which function to call next.\n"
            " - ACTION: Call exactly one function that moves you closer to the final answer.\n"
            " - OBSERVATION: Summarize the function's output.\n"
            "You MUST plan extensively before each function call and reflect on the outcomes of previous calls.\n"
            "Only pass arguments verbatim from the user or function outputs. Do not invent timestamps.\n"
            "If function output indicates a system error or an administrative requirement, prioritize resolving that protocol."
        )
        
        user_prompt = (
            "Carefully read the timestamps and narration. Pay attention to the causal order of events and object details.\n\n"
            "Available tools:\n"
            "- use `global_browse` for general context.\n"
            "- use `clip_search` for keyword-based retrieval.\n"
            "- use `frame_inspect` for visual confirmation of specific details.\n\n"
            "After locating an answer, ALWAYS perform a CONFIRM call with `frame_inspect`.\n\n"
            f"Total video length: {self.video_length_secs} seconds.\n"
            f"Question: {question}\n"
            "Answer with the option's letter from the given choices directly."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        reasoning_trace = []
        clean_question = question[:60].replace('\n', ' ')
        print(f"\n[Agent Logic Start]: {clean_question}...")
        
        for step in range(max_steps):
            if step == max_steps - 1:
                messages.append({"role": "user", "content": "Limit reached. Call the finish function immediately."})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                temperature=0.0
            )
            
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            
            tool_calls_to_execute = []
            
            if getattr(message, "tool_calls", None):
                tool_calls_to_execute = message.tool_calls
                if message.content:
                    print(f"   [Thought]: {message.content.strip()[:100]}...")
                    reasoning_trace.append({"step": step, "type": "thought", "content": message.content.strip()})
                    
            elif message.content:
                content_str = message.content.strip()
                print(f"   [Thought]: {content_str[:150]}...")
                reasoning_trace.append({"step": step, "type": "thought", "content": content_str})
                
                # Fallback for manual JSON function parsing if the model doesn't use the tool API correctly
                start = content_str.find('{')
                end = content_str.rfind('}')
                if start != -1 and end != -1:
                    try:
                        json_str = content_str[start:end+1]
                        data = json.loads(json_str)
                        func_name = None
                        args = {}
                        
                        # 1. Normal standard parsing
                        for key in ["tool_name", "name", "tool"]:
                            if key in data:
                                func_name = data[key]
                                args = data.get("arguments", data.get("parameters", {}))
                                break
                                
                        # 2. Heuristic fault-tolerance patch for Claude models
                        # If no standard tool_name is found, infer the intended tool based on field characteristics
                        if not func_name:
                            if "start_time" in data or "timestamp" in data or "time_ranges_hhmmss" in data:
                                # If time parameters appear, route to frame_inspect
                                func_name = "frame_inspect"
                                args = {"time_ranges_hhmmss": [["00:00:00", "00:00:05"]], "question": question}
                            elif "query" in data or "keyword" in data or "keywords" in data:
                                # If search keywords appear, route to clip_search
                                func_name = "clip_search"
                                args = {"event_description": data.get("query", data.get("keyword", "search"))}
                            elif "answer" in data:
                                # If an answer field appears, route to finish
                                func_name = "finish"
                                args = {"answer": str(data["answer"])}

                        if func_name:
                            class DummyFunction:
                                def __init__(self, n, a):
                                    self.name = n
                                    self.arguments = json.dumps(a) if isinstance(a, dict) else str(a)
                            class DummyToolCall:
                                def __init__(self, func):
                                    self.id = "call_fallback_internal"
                                    self.function = func
                            tool_calls_to_execute.append(DummyToolCall(DummyFunction(func_name, args)))
                    except Exception:
                        pass

            if tool_calls_to_execute:
                for tool_call in tool_calls_to_execute:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    if func_name == "clip_search":
                        query_arg = args.get("event_description", args.get("query", "unknown"))
                        print(f"   [Action | Search]: Retrieving clips for -> '{query_arg}'")
                        obs = self.tools.clip_search(event_description=query_arg, top_k=args.get("top_k", 16))
                        
                    elif func_name == "global_browse":
                        print(f"   [Action | Global]: Getting video overview...")
                        obs = self.tools.global_browse(args.get("query", ""))
                        
                    elif func_name == "frame_inspect":
                        time_ranges = args.get("time_ranges_hhmmss", [])
                        print(f"   [Action | Inspect]: Checking frames for ranges -> {time_ranges}")
                        obs = self.tools.frame_inspect(
                            question=args.get("question", question), 
                            time_ranges_hhmmss=time_ranges
                        )
                        
                    elif func_name == "finish":
                        final_ans = args.get("answer", "")
                        print(f"\n   [Final Answer Received]:")
                        print(f"   --------------------------------------------------")
                        print(f"   {final_ans}")
                        print(f"   --------------------------------------------------")
                        reasoning_trace.append({"step": step, "type": "final_answer", "content": final_ans})
                        return final_ans, reasoning_trace
                        
                    else:
                        obs = f"Invalid function name: '{func_name}'"
                        
                    safe_obs = str(obs) if obs else "Observation: No data returned."
                    obs_preview = safe_obs.replace('\n', ' ')[:80]
                    print(f"   [Observation]: {obs_preview}...")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": getattr(tool_call, "id", "call_fallback_internal"),
                        "content": safe_obs
                    })
                    reasoning_trace.append({"step": step, "type": "tool_call", "name": func_name, "args": args})
                    reasoning_trace.append({"step": step, "type": "observation", "content": safe_obs})
                    print("   [⏳ Frequency Control] Action completed, waiting 3 seconds to prevent rate limiting...")
                    time.sleep(3)
        print("[Max steps reached without conclusion]")
        return "Failed", reasoning_trace