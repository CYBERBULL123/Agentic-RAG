"""
Google Gemini LLM Integration with Streaming Support
"""

import os
from typing import Generator, List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class GeminiLLM:
    """Google Gemini LLM client with streaming and function calling support."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "models/gemini-2.5-flash",
                 temperature: float = 0.3,
                 top_p: float = 0.95,
                 max_tokens: int = 4096):
        """
        Initialize Gemini LLM client.
        
        Args:
            api_key: Google API key (if None, reads from env)
            model_name: Gemini model to use
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini LLM with model: {model_name}")
    
    def generate_response(self, 
                         prompt: str,
                         context: str = "",
                         system_instruction: str = "") -> str:
        """
        Generate a single response from Gemini.
        
        Args:
            prompt: User prompt
            context: Additional context to include
            system_instruction: System-level instruction
            
        Returns:
            Generated response text
        """
        try:
            # Combine context and prompt
            full_prompt = self._build_full_prompt(prompt, context, system_instruction)
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def stream_response(self, 
                       prompt: str,
                       context: str = "",
                       system_instruction: str = "") -> Generator[str, None, None]:
        """
        Generate streaming response from Gemini.
        
        Args:
            prompt: User prompt
            context: Additional context to include
            system_instruction: System-level instruction
            
        Yields:
            Chunks of generated text
        """
        try:
            # Combine context and prompt
            full_prompt = self._build_full_prompt(prompt, context, system_instruction)
            
            # Generate streaming response
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens
                ),
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
    
    def analyze_document(self, 
                        document_text: str, 
                        query: str = "Analyze this document") -> str:
        """
        Analyze a document and extract key information.
        
        Args:
            document_text: Full document text
            query: Analysis query
            
        Returns:
            Analysis result
        """
        analysis_prompt = f"""
        Please analyze the following document and provide a comprehensive summary:
        
        Document:
        {document_text[:4000]}  # Limit document length
        
        Query: {query}
        
        Please provide:
        1. Main topics and themes
        2. Key facts and information
        3. Important entities (people, places, organizations)
        4. Summary in 3-5 sentences
        """
        
        return self.generate_response(analysis_prompt)
    
    def generate_embeddings_text(self, text: str) -> str:
        """
        Generate text optimized for embedding creation.
        
        Args:
            text: Original text
            
        Returns:
            Optimized text for embedding
        """
        prompt = f"""
        Optimize the following text for semantic search and embedding creation.
        Make it clear, concise, and preserve key information:
        
        Original text:
        {text[:2000]}
        
        Optimized version:
        """
        
        return self.generate_response(prompt)
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted information
        """
        extraction_prompt = f"""
        Extract key information from the following text and structure it as JSON:
        
        Text:
        {text[:3000]}
        
        Please extract:
        - title: Main title or subject
        - summary: Brief summary (2-3 sentences)
        - keywords: List of important keywords
        - entities: People, places, organizations mentioned
        - topics: Main topics discussed
        - date: Any dates mentioned
        
        Return as valid JSON format.
        """
        
        try:
            response = self.generate_response(extraction_prompt)
            # Try to parse as JSON, fallback to text if needed
            import json
            return json.loads(response)
        except:
            return {"raw_response": response}
    
    def _build_full_prompt(self, 
                          prompt: str, 
                          context: str = "", 
                          system_instruction: str = "") -> str:
        """
        Build the full prompt with context and system instructions.
        
        Args:
            prompt: User prompt
            context: Additional context
            system_instruction: System instruction
            
        Returns:
            Complete prompt
        """
        parts = []
        
        if system_instruction:
            parts.append(f"System: {system_instruction}")
        
        if context:
            parts.append(f"Context:\n{context}")
        
        parts.append(f"User: {prompt}")
        
        return "\n\n".join(parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            models = genai.list_models()
            current_model = None
            
            for model in models:
                if self.model_name in model.name:
                    current_model = {
                        "name": model.name,
                        "display_name": model.display_name,
                        "description": model.description,
                        "input_token_limit": getattr(model, 'input_token_limit', 'Unknown'),
                        "output_token_limit": getattr(model, 'output_token_limit', 'Unknown')
                    }
                    break
            
            return current_model or {"name": self.model_name, "status": "Model info not found"}
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}


# Singleton instance for global access
_gemini_llm = None

def get_gemini_llm() -> GeminiLLM:
    """Get global Gemini LLM instance."""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = GeminiLLM()
    return _gemini_llm