# Example Usage and Testing

## Basic Usage Examples

### 1. Simple Questions
- "What is the weather like today?"
- "Tell me about artificial intelligence"
- "What's 25 * 47?"

### 2. Document-Based Queries (after uploading documents)
- "Summarize the uploaded document"
- "What are the key points from my files?"
- "Find information about [topic] in my documents"

### 3. Mixed Queries (Web + Documents)
- "Compare current AI trends with information from my documents"
- "What's the latest news about [topic] and how does it relate to my research?"

### 4. Function Calling Examples
- "Calculate 15% of 250"
- "What time is it now?"
- "Compute (45 + 67) * 2.5"

## Sample Documents to Test

Create test files to upload:

### test_doc.txt
```
This is a sample document about machine learning.

Machine learning is a subset of artificial intelligence that focuses on the ability of machines to receive data and learn for themselves without being explicitly programmed.

Key concepts include:
- Supervised learning
- Unsupervised learning
- Neural networks
- Deep learning

Applications include image recognition, natural language processing, and recommendation systems.
```

### research_notes.md
```
# AI Research Notes

## Current Trends
- Large Language Models (LLMs) are becoming more capable
- Multimodal AI is gaining traction
- Edge AI deployment is increasing

## Challenges
- Data privacy concerns
- Computational resource requirements
- Model interpretability

## Future Directions
- More efficient training methods
- Better human-AI collaboration
- Ethical AI development
```

## Testing Workflow

1. **Upload Documents**: Add the sample files above
2. **Test Knowledge Base**: Ask "What did I upload about machine learning?"
3. **Test Web Search**: Ask "What are the latest AI developments?"
4. **Test Memory**: Have a multi-turn conversation
5. **Test Functions**: Ask for calculations or current time
6. **Provide Feedback**: Rate responses to test RLHF system

## Expected Behaviors

- **Context Awareness**: Agent remembers previous messages
- **Source Attribution**: Responses indicate web vs document sources
- **Streaming**: Text appears progressively (if enabled)
- **Error Handling**: Graceful handling of API issues
- **Memory Management**: Conversation history persists within session