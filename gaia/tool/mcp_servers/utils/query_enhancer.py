"""Query enhancement and classification using Gradient API."""

import logging
import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger("gaia-agent")

load_dotenv()


class QueryEnhancer:
    """Query enhancement and classification using Gradient API."""

    def __init__(self):
        self.api_key = os.environ.get("GRADIENT_API_KEY", "")
        if not self.api_key:
            raise ValueError("GRADIENT_API_KEY environment variable is not set")

        self.base_url = os.environ.get("GRADIENT_BASE_URL", "https://inference.do-ai.run/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = "anthropic-claude-4.6-sonnet"
    
    def enhance_and_classify_query(self, query: str) -> Dict[str, str]:
        """
        Enhance and classify a query using GPT-4.1-mini.
        
        Args:
            query: Original search query
            
        Returns:
            Dictionary with enhanced_query, query_type, and query_topic
        """
        system_prompt = """You are a query enhancement and classification expert. Your task is to:

1. Check if the query is in natural language and easy to understand format. If not, rephrase it into natural language that is clear and easy to understand for a search engine.

2. Classify the query type:
   - "factual": Simple factual questions that require direct answers
   - "multi-step": Complex questions requiring multi-step reasoning or analysis

3. Classify the query topic:
   - "academic": Questions about research, scientific papers, scholarly topics, or academic subjects
   - "general": General web search topics, news, everyday questions, or non-academic content

Return your response in the following JSON format:
{
    "enhanced_query": "rephrased query in natural language",
    "query_type": "factual" or "multi-step",
    "query_topic": "academic" or "general"
}

Examples:

Input: "2+2=?"
Output: {
    "enhanced_query": "What is 2 plus 2?",
    "query_type": "factual",
    "query_topic": "general"
}

Input: "ML transformer arch papers 2023"
Output: {
    "enhanced_query": "What are the latest machine learning transformer architecture research papers published in 2023?",
    "query_type": "multi-step",
    "query_topic": "academic"
}

Input: "Who won the 2023 Nobel Prize in Physics?"
Output: {
    "enhanced_query": "Who won the 2023 Nobel Prize in Physics?",
    "query_type": "factual",
    "query_topic": "general"
}

Always return valid JSON."""

        try:
            response = self.client.chat.completions.create(
                extra_headers={},
                extra_body={},
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Please enhance and classify this query: {query}"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "enhanced_query": {
                                    "type": "string",
                                    "description": "Rephrased query in natural language"
                                },
                                "query_type": {
                                    "type": "string",
                                    "enum": ["factual", "multi-step"],
                                    "description": "Type of query: factual or multi-step reasoning"
                                },
                                "query_topic": {
                                    "type": "string",
                                    "enum": ["academic", "general"],
                                    "description": "Topic of query: academic or general"
                                }
                            },
                            "required": ["enhanced_query", "query_type", "query_topic"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            parsed_result = json.loads(content)
            
            # Validate required fields
            required_fields = ["enhanced_query", "query_type", "query_topic"]
            for field in required_fields:
                if field not in parsed_result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate enum values
            if parsed_result["query_type"] not in ["factual", "multi-step"]:
                raise ValueError(f"Invalid query_type: {parsed_result['query_type']}")
            
            if parsed_result["query_topic"] not in ["academic", "general"]:
                raise ValueError(f"Invalid query_topic: {parsed_result['query_topic']}")
            
            return parsed_result
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            raise ValueError(f"OpenRouter API request failed: {str(e)}")


def enhance_query_for_perplexity(query: str) -> Dict[str, Any]:
    """
    Enhance query and return configuration for Perplexity search.
    
    Args:
        query: Original search query
        
    Returns:
        Dictionary with enhanced query and Perplexity configuration
    """
    try:
        result = _get_enhancer().enhance_and_classify_query(query)

        # Determine Perplexity model and search type
        if result["query_type"] == "factual":
            model = "sonar"
        else:  # multi-step
            model = "sonar-reasoning-pro"
        
        search_type = "academic" if result["query_topic"] == "academic" else "web"
        
        return {
            "enhanced_query": result["enhanced_query"],
            "query_type": result["query_type"],
            "query_topic": result["query_topic"],
            "perplexity_model": model,
            "search_type": search_type
        }
    except Exception as e:
        # Fallback to defaults if enhancement fails
        logger.warning("Query enhancement failed: %s -- using defaults", e)
        return {
            "enhanced_query": query,  # Use original query
            "query_type": "factual",
            "query_topic": "general",
            "perplexity_model": "sonar-pro",
            "search_type": "web"
        }


def _classify_perplexica_mode(query_type: str, query_topic: str) -> str:
    """Map query classification to a Perplexica search mode.

    - factual         → speed   (2 iterations, no reasoning)
    - multi-step general  → balanced (6 iterations, reasoning)
    - multi-step academic → quality  (25 iterations, reasoning)
    """
    if query_type == "factual":
        return "speed"
    if query_topic == "academic":
        return "balanced"
    
    return "balanced"


# Cached instance — avoids creating a new OpenAI HTTP client on every call.
_cached_enhancer: "QueryEnhancer | None" = None


def _get_enhancer() -> QueryEnhancer:
    global _cached_enhancer
    if _cached_enhancer is None:
        _cached_enhancer = QueryEnhancer()
    return _cached_enhancer


# Convenience function
def enhance_and_classify(query: str) -> Dict[str, str]:
    """
    Convenience function to enhance and classify a query.

    Args:
        query: Original search query

    Returns:
        Dictionary with enhanced_query, query_type, query_topic, and perplexica_mode
    """
    try:
        result = _get_enhancer().enhance_and_classify_query(query)
        result["perplexica_mode"] = _classify_perplexica_mode(
            result["query_type"], result["query_topic"]
        )
        return result
    except Exception as e:
        # Fallback to defaults if enhancement fails
        logger.warning("Query enhancement failed: %s -- using defaults", e)
        return {
            "enhanced_query": query,
            "query_type": "factual",
            "query_topic": "general",
            "perplexica_mode": "speed",
        }


if __name__ == "__main__":
    # Test examples
    test_queries = [
        "2+2=?",
        "ML transformer arch papers 2023",
        "Who won the 2023 Nobel Prize in Physics?",
        "explain quantum entanglement mechanism research",
        "weather today NYC"
    ]
    
    for query in test_queries:
        try:
            result = enhance_and_classify(query)
            print(f"Query: {query}")
            print(f"Result: {json.dumps(result, indent=2)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing '{query}': {e}")