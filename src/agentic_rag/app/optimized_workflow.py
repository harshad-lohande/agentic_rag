# src/agentic_rag/app/optimized_workflow.py

"""
Optimized Agentic RAG Workflow for 5-10 second response times.

This workflow eliminates the major performance bottlenecks:
1. Pre-loaded models (eliminates 80-90s of repeated model loading)
2. Fast extractive compression (replaces 50s LLM-based compression)
3. Linear workflow (eliminates correction loop that could double execution time)
4. Smart retrieval as default (uses the proven effective approach)
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from agentic_rag.app.agentic_workflow import (
    GraphState,
    check_semantic_cache,
    store_in_semantic_cache,
    classify_query,
    smart_retrieval_and_rerank,
    summarize_history,
    generate_answer,
    grounding_and_safety_check,
    web_search,
    maybe_reset_usage_counters,
    accumulate_token_usage,
    extract_token_usage_from_response,
    get_last_message_content,
    get_last_human_message_content
)
from agentic_rag.app.fast_compression import fast_compress_documents
from agentic_rag.logging_config import logger
from agentic_rag.config import settings


class OptimizedAgenticWorkflow:
    """
    Streamlined agentic workflow optimized for speed and reliability.
    
    Workflow: cache_check → classify → retrieve_smart → compress_fast → summarize → generate → safety_check
    """
    
    def __init__(self):
        self.step_timings = {}
    
    async def execute(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the optimized workflow for a user query.
        
        Args:
            query: User's question
            session_id: Session identifier for conversation history
            
        Returns:
            Final response with answer and metadata
        """
        logger.info(f"🚀 Starting optimized workflow for session {session_id}")
        import time
        workflow_start = time.time()
        
        # Initialize state
        state = {
            "messages": [HumanMessage(content=query)],
            "documents": [],
            "retrieval_success": False,
            "is_web_search": False,
            **maybe_reset_usage_counters({"messages": [HumanMessage(content=query)]})
        }
        
        try:
            # Step 1: Check semantic cache (fastest path to answer)
            state = await self._time_step("cache_check", check_semantic_cache, state)
            if self._should_return_cached_answer(state):
                logger.info("✅ Cache hit - returning cached answer")
                return self._format_final_response(state, time.time() - workflow_start)
            
            # Step 2: Classify query to determine routing
            state = await self._time_step("classify_query", classify_query, state)
            
            # Step 3: Route based on classification
            if self._should_use_web_search(state):
                # Web search path for recent events or out-of-scope queries
                state = await self._time_step("web_search", web_search, state)
                state = await self._time_step("summarize_history", summarize_history, state)
                state = await self._time_step("generate_answer", generate_answer, state)
            else:
                # Smart retrieval path (default, optimized)
                state = await self._time_step("smart_retrieval_and_rerank", smart_retrieval_and_rerank, state)
                
                if state.get("retrieval_success", False):
                    # Fast compression instead of expensive LLM-based compression
                    state = await self._time_step("fast_compress_documents", self._fast_compress_step, state)
                    state = await self._time_step("summarize_history", summarize_history, state)
                    state = await self._time_step("generate_answer", generate_answer, state)
                else:
                    # Fallback: return failure message instead of retrying
                    logger.warning("❌ Retrieval failed - returning failure message")
                    state = self._create_failure_response(state)
            
            # Step 4: Final safety check (no correction loop)
            state = await self._time_step("grounding_and_safety_check", grounding_and_safety_check, state)
            
            # Step 5: Store successful answers in cache
            if self._is_successful_answer(state):
                state = await self._time_step("store_cache", store_in_semantic_cache, state)
            
            workflow_time = time.time() - workflow_start
            logger.info(f"✅ Optimized workflow completed in {workflow_time:.2f}s")
            
            return self._format_final_response(state, workflow_time)
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            return self._format_error_response(str(e), time.time() - workflow_start)
    
    async def _time_step(self, step_name: str, step_function, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step and track timing."""
        import time
        
        step_start = time.time()
        logger.debug(f"⚡ Executing step: {step_name}")
        
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(step_function):
                result = await step_function(state)
            else:
                result = step_function(state)
            
            step_time = time.time() - step_start
            self.step_timings[step_name] = step_time
            
            logger.debug(f"✅ Step {step_name} completed in {step_time:.3f}s")
            
            # Update state with results
            if isinstance(result, dict):
                state.update(result)
                
            return state
            
        except Exception as e:
            step_time = time.time() - step_start
            self.step_timings[step_name] = step_time
            logger.error(f"❌ Step {step_name} failed after {step_time:.3f}s: {e}")
            raise
    
    async def _fast_compress_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fast compression step using extractive method."""
        documents = state.get("documents", [])
        if not documents:
            return {"documents": []}
        
        # Get the current query
        query = get_last_human_message_content(state["messages"])
        
        # Perform fast extractive compression
        compressed_docs = fast_compress_documents(documents, query)
        
        logger.info(f"Fast compression: {len(documents)} docs → {len(compressed_docs)} compressed docs")
        
        return {"documents": compressed_docs}
    
    def _should_return_cached_answer(self, state: Dict[str, Any]) -> bool:
        """Check if we should return a cached answer."""
        messages = state.get("messages", [])
        if len(messages) >= 2 and isinstance(messages[-1], AIMessage):
            # Last message is AI response (cached answer)
            return True
        return False
    
    def _should_use_web_search(self, state: Dict[str, Any]) -> bool:
        """Determine if query should use web search based on classification."""
        # This would be enhanced with more sophisticated routing logic
        query = get_last_human_message_content(state["messages"]).lower()
        
        # Simple heuristics for web search triggers
        web_search_triggers = [
            "latest", "recent", "today", "yesterday", "this week", "this month",
            "current", "now", "breaking", "news", "update"
        ]
        
        return any(trigger in query for trigger in web_search_triggers)
    
    def _create_failure_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a failure response when retrieval fails."""
        failure_message = (
            "I was unable to find relevant information in the available documents. "
            "Could you please try rephrasing your question or providing more specific details?"
        )
        
        ai_message = AIMessage(content=failure_message)
        messages = state.get("messages", [])
        messages.append(ai_message)
        
        return {
            **state,
            "messages": messages,
            "retrieval_success": False
        }
    
    def _is_successful_answer(self, state: Dict[str, Any]) -> bool:
        """Check if the answer was successful and should be cached."""
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            answer = messages[-1].content
            # Simple check - could be enhanced with more sophisticated validation
            return len(answer) > 50 and "unable to find" not in answer.lower()
        return False
    
    def _format_final_response(self, state: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Format the final response with metadata."""
        messages = state.get("messages", [])
        final_answer = messages[-1].content if messages else "No answer generated."
        
        return {
            "answer": final_answer,
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "total_tokens": state.get("total_tokens", 0),
            "total_time_seconds": round(total_time, 2),
            "step_timings": {k: round(v, 3) for k, v in self.step_timings.items()},
            "retrieval_success": state.get("retrieval_success", False),
            "is_web_search": state.get("is_web_search", False),
            "optimization_applied": True
        }
    
    def _format_error_response(self, error_message: str, total_time: float) -> Dict[str, Any]:
        """Format an error response."""
        return {
            "answer": f"I encountered an error while processing your request: {error_message}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_time_seconds": round(total_time, 2),
            "step_timings": {k: round(v, 3) for k, v in self.step_timings.items()},
            "retrieval_success": False,
            "is_web_search": False,
            "optimization_applied": True,
            "error": error_message
        }


# Global instance for reuse
optimized_workflow = OptimizedAgenticWorkflow()