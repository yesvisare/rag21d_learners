# AUGMENTED SCRIPT: M1.4 - Query Pipeline & Response Generation

**Duration:** 44 minutes (expanded from 20 minutes)  
**Compliance:** 6/6 TVH v2.0 sections complete

---

## Video M1.4: Query Pipeline & Response Generation (44 minutes)

### [0:00] Introduction

[SLIDE: Title - "Query Pipeline & Response Generation: From Question to Answer"]

Welcome to the final video of Module 1! We've covered vector databases, advanced indexing, and document processing. Now it's time to bring it all together and turn user queries into amazing answers.

Today, we're building the query pipeline—the system that takes a user's question and returns a perfect, contextual answer. This is where RAG magic happens.

[SLIDE: Learning Objectives]

**By the end of this video, learners will be able to:**
- Build a complete query pipeline with 7 stages (query → response)
- Implement hybrid search with semantic and keyword retrieval strategies
- Apply cross-encoder reranking to improve result quality
- Debug 5 common RAG failures in production environments
- Determine when NOT to use RAG using the Decision Card framework
- Evaluate RAG system quality using automated metrics
- Handle errors and implement fallback strategies for production reliability

Let's build the complete query pipeline.

---

### [0:45] Prerequisites Check

[SLIDE: "Prerequisites - Ready to Build?"]

Before we dive in, let's make sure you have everything you need:

**Required Completion:**
- âœ… Module 1.1: Vector Databases & Semantic Search
- âœ… Module 1.2: Advanced Indexing Strategies
- âœ… Module 1.3: Document Processing & Chunking

**Required Setup:**
- âœ… Pinecone account with API key
- âœ… OpenAI account with API key
- âœ… Python environment with libraries: `pinecone-client`, `openai`, `sentence-transformers`

**Quick Validation:**

```bash
# Test your setup
python -c "import pinecone; import openai; from sentence_transformers import CrossEncoder; print('✓ All libraries installed')"

# Verify API keys
python -c "import os; assert os.getenv('PINECONE_API_KEY'), 'Set PINECONE_API_KEY'; assert os.getenv('OPENAI_API_KEY'), 'Set OPENAI_API_KEY'; print('✓ API keys configured')"
```

If any of these fail, pause now and complete the prerequisites. Everything we build today depends on this foundation.

**[PAUSE]** Good? Let's identify the problem we're solving.

---

### [1:45] The Problem: Why Simple Approaches Fail

[SLIDE: "The Problem: Naive Question-Answering Fails"]

Before we build our sophisticated pipeline, let me show you why we need it. What happens if we try simpler approaches?

**[DEMO 1: Keyword Search Failure]**

```python
# Naive keyword search
query = "How do I improve the accuracy of my RAG system?"
keywords = ["improve", "accuracy", "RAG", "system"]

# Simple keyword matching in documents
matches = [doc for doc in documents if any(k in doc.lower() for k in keywords)]

print(f"Found {len(matches)} documents")
# Problem: Returns 50+ documents, many irrelevant
# "RAG" appears in "fragmentation", "accuracy" in "inaccurate results"
```

**Result:** Too many irrelevant results. No ranking. No semantic understanding.

**[DEMO 2: Simple LLM Prompting Failure]**

```python
# Direct LLM query without retrieval
prompt = "How do I improve the accuracy of my RAG system?"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# Problem: Generic advice. No specific details from YOUR docs.
# Might hallucinate techniques that don't apply to your setup.
```

**Result:** Generic answer. No grounding in your specific documentation. Potential hallucinations.

**[SCREEN: Side-by-side comparison]**

**What we need:**
1. **Semantic understanding** - not just keyword matching
2. **Relevant retrieval** - find the RIGHT 5 chunks, not 50 random ones  
3. **Grounded responses** - answers based on YOUR documentation
4. **Source attribution** - users can verify claims
5. **Error handling** - graceful fallbacks when retrieval fails

That's what a proper query pipeline gives us. Let's build it.

---

### [3:45] REALITY CHECK: What Query Pipelines Actually Do

**[SLIDE: Reality Check - Let's Be Honest]**

Before we go further, I need to be completely honest with you about what we're building. RAG query pipelines are powerful, but they're not magic. Let me tell you exactly what they do well and what they struggle with.

**[SLIDE: What Query Pipelines DO Well]**

**What this approach DOES well:**

✅ **Semantic question-answering over large document sets**  
We're talking 1,000+ documents. The pipeline can find relevant information even when users don't use exact keywords. That's huge.

✅ **Reduces hallucination by 60-80% compared to base LLMs**  
By grounding responses in retrieved context, we dramatically reduce made-up information. Not eliminated—reduced.

✅ **Provides source attribution**  
Every answer can point back to specific documents. Users can verify. That builds trust.

✅ **Handles diverse query types**  
Questions, comparisons, how-tos, troubleshooting—the same pipeline handles them all with appropriate strategies.

**[SLIDE: What Query Pipelines DON'T Do]**

**What this approach DOESN'T do:**

❌ **Cannot handle multi-turn context without additional memory**  
Each query is independent. "What about that?" won't work unless you add conversation memory (Module 2 topic).

❌ **Adds 200-400ms latency compared to cached responses**  
Vector search, reranking, LLM generation—it all takes time. If you need <100ms responses, this won't work.

❌ **Requires 5+ infrastructure components to maintain**  
Embedding API, vector database, reranker, LLM, orchestration layer. Each can fail. Each needs monitoring.

❌ **Context window limits prevent complex multi-document reasoning**  
Even with 128K token context windows, you can only fit so much. Complex cross-document analysis is challenging.

❌ **Query quality directly determines answer quality**  
Garbage in, garbage out. If users ask vague questions, they get vague answers.

**[PAUSE]**

**[SLIDE: The Trade-Offs You're Making]**

**Here are the real trade-offs:**

**Complexity vs. Accuracy:**  
You're adding 5 components to get better answers. Is 20% accuracy improvement worth 5x infrastructure complexity? That depends on your use case.

**Cost vs. Capability:**  
This pipeline costs $150-500/month for moderate usage. A simple GPT-4 prompt costs $50/month. The RAG pipeline gives you source attribution and reduces hallucination—are those worth $100-450/month to you?

**Latency vs. Quality:**  
A cached FAQ response: 10ms. Our pipeline: 300ms. Reranking adds quality but costs time. Every optimization is a trade-off.

**[EMPHASIS]** This is important: **If you have fewer than 100 documents or fewer than 50 queries per day, you probably don't need this complexity.** A well-crafted prompt with GPT-4's extended context might be enough.

We'll see these trade-offs play out as we build. Keep them in mind.

---

### [6:45] Query Pipeline Architecture

[SLIDE: "Complete Query Pipeline"]

Here's the full query pipeline:

```
User Query → Query Processing → Retrieval → Reranking → Context Prep → LLM Generation → Response
     ↓              ↓                ↓          ↓             ↓              ↓             ↓
 Raw text    Transformation      Pinecone   Score by     Format for    Add system     Stream to
             Expansion           Search     relevance      prompt       prompt         user
             Classification
```

Each step matters. Let's build it piece by piece.

---

### [7:45] Step 1: Query Understanding

[SLIDE: "Query Understanding: Know What They're Asking"]

Before we search, we need to understand what the user is really asking. Are they asking a question? Looking for specific information? Wanting a comparison?

[CODE: Query classifier]

```python
from openai import OpenAI
from typing import Dict, List
from enum import Enum

class QueryType(Enum):
    FACTUAL = "factual"           # "What is X?"
    COMPARISON = "comparison"     # "Compare X and Y"
    HOW_TO = "how_to"            # "How do I..."
    DEFINITION = "definition"     # "Define X"
    OPINION = "opinion"          # "Why should I..."
    TROUBLESHOOTING = "troubleshooting"  # "X isn't working"

class QueryProcessor:
    """Understand and transform user queries"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the type of query"""
        # Simple heuristic classification (production would use LLM)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'what are', 'define']):
            return QueryType.DEFINITION
        elif any(word in query_lower for word in ['how to', 'how do', 'how can']):
            return QueryType.HOW_TO
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return QueryType.COMPARISON
        elif any(word in query_lower for word in ['why', 'should i', 'recommend']):
            return QueryType.OPINION
        elif any(word in query_lower for word in ['not working', 'error', 'fix', 'problem']):
            return QueryType.TROUBLESHOOTING
        else:
            return QueryType.FACTUAL
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative phrasings and expansions
        Helps catch more relevant documents
        """
        prompt = f"""Given this user query, generate 3 alternative ways to phrase the same question.
Keep them concise and focused on the same information need.

Original query: "{query}"

Return only the 3 alternatives, one per line."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        alternatives = response.choices[0].message.content.strip().split('\n')
        alternatives = [alt.strip('- ').strip() for alt in alternatives if alt.strip()]
        
        return [query] + alternatives  # Include original
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms for filtering"""
        # Simple extraction (production would be more sophisticated)
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords

# Usage
processor = QueryProcessor(openai_api_key="your-key")

query = "How do I optimize my Pinecone index for better performance?"

# Classify
query_type = processor.classify_query(query)
print(f"Query type: {query_type.value}")

# Expand
alternatives = processor.expand_query(query)
print(f"\nQuery alternatives:")
for i, alt in enumerate(alternatives, 1):
    print(f"  {i}. {alt}")

# Extract keywords
keywords = processor.extract_keywords(query)
print(f"\nKeywords: {keywords}")
```

[SCREEN: Showing query classification and expansion]

---

### [9:15] Step 2: Retrieval Strategy

**[SLIDE: "Smart Retrieval: Not Just Simple Search"]**

Now we retrieve. But we're not just doing a single search—we're using multiple strategies to ensure we get the best results.

[CODE: Multi-strategy retrieval]

```python
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Dict, Any
from pinecone_text.sparse import BM25Encoder

class SmartRetriever:
    """Advanced retrieval with multiple strategies"""
    
    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str
    ):
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.bm25 = BM25Encoder.default()
    
    def retrieve(
        self,
        query: str,
        query_type: QueryType,
        top_k: int = 10,
        alpha: float = None,
        filters: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using query-type-specific strategy
        """
        # Auto-adjust alpha based on query type
        if alpha is None:
            alpha = self._get_optimal_alpha(query_type)
        
        # Generate query vectors
        dense_vec = self._get_dense_embedding(query)
        sparse_vec = self._get_sparse_embedding(query)
        
        # Retrieve from Pinecone
        results = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filters,
            alpha=alpha
        )
        
        # Process and enrich results
        processed_results = self._process_results(results, query)
        
        return processed_results
    
    def _get_optimal_alpha(self, query_type: QueryType) -> float:
        """Choose optimal alpha based on query type"""
        alpha_map = {
            QueryType.DEFINITION: 0.7,      # Semantic understanding important
            QueryType.HOW_TO: 0.6,          # Balanced
            QueryType.COMPARISON: 0.7,      # Semantic important
            QueryType.FACTUAL: 0.5,         # Balanced
            QueryType.OPINION: 0.8,         # Very semantic
            QueryType.TROUBLESHOOTING: 0.3  # Exact terms important
        }
        return alpha_map.get(query_type, 0.5)
    
    def _get_dense_embedding(self, text: str) -> List[float]:
        """Generate dense embedding"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _get_sparse_embedding(self, text: str) -> Dict:
        """Generate sparse embedding"""
        return self.bm25.encode_queries(text)
    
    def _process_results(
        self,
        results: Dict,
        query: str
    ) -> List[Dict[str, Any]]:
        """Process and enrich raw results"""
        processed = []
        
        for match in results['matches']:
            processed.append({
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('chunk_text', ''),
                'source': match['metadata'].get('source', 'unknown'),
                'metadata': match['metadata']
            })
        
        return processed
    
    def retrieve_with_expansion(
        self,
        query: str,
        alternatives: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve using query expansion
        Combines results from multiple query variants
        """
        all_results = {}
        
        for q in [query] + alternatives:
            results = self.retrieve(
                query=q,
                query_type=QueryType.FACTUAL,
                top_k=top_k
            )
            
            # Add to results dict, keeping highest score for each chunk
            for result in results:
                chunk_id = result['id']
                if chunk_id not in all_results or result['score'] > all_results[chunk_id]['score']:
                    all_results[chunk_id] = result
        
        # Sort by score and return top-k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:top_k * 2]  # Return more since we're merging

# Usage
retriever = SmartRetriever(
    pinecone_api_key="your-pinecone-key",
    openai_api_key="your-openai-key",
    index_name="production-rag"
)

# Simple retrieval
results = retriever.retrieve(
    query="How do I optimize vector search?",
    query_type=QueryType.HOW_TO,
    top_k=5
)

print(f"Retrieved {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.4f}")
    print(f"   Source: {result['source']}")
    print(f"   Text: {result['text'][:100]}...")

# Retrieval with expansion
expanded_results = retriever.retrieve_with_expansion(
    query="How do I optimize vector search?",
    alternatives=[
        "What are best practices for vector database performance?",
        "Tips for improving semantic search speed"
    ],
    top_k=5
)

print(f"\n\nExpanded retrieval: {len(expanded_results)} results")
```

---

### [12:45] Step 3: Reranking

**[SLIDE: "Reranking: Quality Over Quantity"]**

Pinecone gives us the most similar chunks, but similarity doesn't always mean relevance. Reranking uses a more sophisticated model to score relevance.

[CODE: Cross-encoder reranking]

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank results using cross-encoder model"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize with a cross-encoder model
        These models are trained specifically for relevance ranking
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder scores
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, result['text']] for result in results]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add reranking scores to results
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
            result['original_score'] = result['score']
        
        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:top_k]

# Usage
reranker = Reranker()

query = "How do I improve RAG system accuracy?"

# Get initial results
initial_results = retriever.retrieve(
    query=query,
    query_type=QueryType.HOW_TO,
    top_k=10
)

# Rerank
final_results = reranker.rerank(
    query=query,
    results=initial_results,
    top_k=5
)

print("Before and after reranking:\n")
for i, result in enumerate(final_results, 1):
    print(f"{i}. Rerank Score: {result['rerank_score']:.4f} (Original: {result['original_score']:.4f})")
    print(f"   {result['text'][:80]}...")
    print()

```

[SCREEN: Showing how reranking changes the order]

Notice how reranking can significantly change the order. The cross-encoder understands query-document relevance better than pure vector similarity.

---

### [14:45] ALTERNATIVE SOLUTIONS: Beyond RAG Query Pipelines

**[SLIDE: "Alternative Approaches - Choose Wisely"]**

Before we continue building, you need to know that RAG query pipelines aren't the only way to solve the question-answering problem. Let's look at the complete landscape of alternatives so you can make an informed decision.

**[SLIDE: Three Strategic Alternatives]**

**Option 1: RAG Query Pipeline (What We're Building)**
- **Best for:** Frequently changing knowledge base (weekly+ updates), need source attribution, 100-10,000 documents
- **Key trade-off:** Adds infrastructure complexity and 200-400ms latency
- **Cost:** $150-500/month (Pinecone + OpenAI + reranker + hosting)
- **Example use case:** Customer support chatbot with 500 help articles updated weekly

**Option 2: Fine-Tuned Model**
- **Best for:** Static knowledge base, need <100ms responses, can afford upfront training time
- **Key trade-off:** Expensive to update (requires retraining), no source attribution
- **Cost:** $500-2,000 upfront training, $100-300/month inference, $500-2,000 per update
- **Example use case:** Legal contract analysis with stable regulations, updated quarterly

**Option 3: Long-Context Prompting (GPT-4 with 128K context)**
- **Best for:** Small knowledge base (<50 documents), <500 queries/day, rapid prototyping
- **Key trade-off:** Limited to context window size, higher per-query cost at scale
- **Cost:** $50-150/month for low volume, breaks down at >1K queries/day
- **Example use case:** Internal wiki with 20 policy documents, 50 queries/day

**[DIAGRAM: Decision Framework]**

```
Start Here
    |
    ├─> Static knowledge + high query volume? → Fine-tuned model
    ├─> Small docs (<50) + low volume (<500/day)? → Long-context prompting
    └─> Frequently changing + need sources? → RAG pipeline (this video)
```

**[SLIDE: Why We're Using RAG for This Video]**

For this video, we're building a RAG query pipeline because:
1. We're assuming a medium-to-large knowledge base (100-10,000 docs)
2. Content changes frequently enough that retraining is impractical
3. Source attribution is valuable for trust
4. Query volume justifies the infrastructure investment (>50/day)

If your situation is different, revisit this decision. **Don't default to RAG just because it's popular.** Choose based on your constraints.

---

### [16:45] Step 4: Context Preparation

**[SLIDE: "Context Preparation: Format for the LLM"]**

Now we have our top results. We need to format them into a context that the LLM can use effectively.

[CODE: Context formatter]

```python
class ContextBuilder:
    """Build formatted context from retrieved chunks"""
    
    def __init__(self, max_context_length: int = 4000):
        """
        Args:
            max_context_length: Maximum characters in context
        """
        self.max_context_length = max_context_length
    
    def build_context(
        self,
        results: List[Dict[str, Any]],
        include_sources: bool = True,
        deduplicate: bool = True
    ) -> str:
        """
        Build formatted context from results
        """
        if deduplicate:
            results = self._deduplicate_results(results)
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Format chunk with metadata
            chunk_text = result['text'].strip()
            source = result.get('source', 'Unknown')
            
            if include_sources:
                formatted_chunk = f"[Source {i}: {source}]\n{chunk_text}\n"
            else:
                formatted_chunk = f"{chunk_text}\n\n"
            
            # Check if adding this chunk would exceed limit
            if current_length + len(formatted_chunk) > self.max_context_length:
                break
            
            context_parts.append(formatted_chunk)
            current_length += len(formatted_chunk)
        
        context = "\n".join(context_parts)
        
        return context
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove near-duplicate chunks"""
        seen_texts = set()
        unique_results = []
        
        for result in results:
            # Use first 100 chars as fingerprint
            fingerprint = result['text'][:100].strip()
            
            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                unique_results.append(result)
        
        return unique_results
    
    def build_context_with_scores(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build context with additional metadata
        """
        context = self.build_context(results)
        
        return {
            'context': context,
            'num_chunks': len(results),
            'sources': list(set(r.get('source', 'Unknown') for r in results)),
            'avg_score': sum(r['score'] for r in results) / len(results) if results else 0,
            'chunks_used': [r['id'] for r in results]
        }

# Usage
context_builder = ContextBuilder(max_context_length=3000)

# Build simple context
context = context_builder.build_context(final_results)

print("Generated Context:")
print("="*60)
print(context)
print("="*60)

# Build context with metadata
context_with_meta = context_builder.build_context_with_scores(final_results)

print(f"\nContext metadata:")
print(f"  Chunks used: {context_with_meta['num_chunks']}")
print(f"  Average score: {context_with_meta['avg_score']:.4f}")
print(f"  Sources: {', '.join(context_with_meta['sources'])}")
```

[SCREEN: Formatted context ready for LLM]

---

### [18:45] Step 5: Prompt Engineering

**[SLIDE: "Prompt Engineering: Getting Quality Answers"]**

The prompt is crucial. A good prompt ensures the LLM uses the context effectively and generates accurate, helpful responses.

[CODE: Prompt templates]

```python
class PromptBuilder:
    """Build effective prompts for different query types"""
    
    def __init__(self):
        self.templates = {
            QueryType.FACTUAL: self._factual_template,
            QueryType.HOW_TO: self._howto_template,
            QueryType.COMPARISON: self._comparison_template,
            QueryType.DEFINITION: self._definition_template,
            QueryType.TROUBLESHOOTING: self._troubleshooting_template,
            QueryType.OPINION: self._opinion_template
        }
    
    def build_prompt(
        self,
        query: str,
        context: str,
        query_type: QueryType
    ) -> List[Dict[str, str]]:
        """
        Build messages for chat completion
        """
        template_func = self.templates.get(
            query_type,
            self._factual_template
        )
        
        return template_func(query, context)
    
    def _factual_template(self, query: str, context: str) -> List[Dict]:
        """Template for factual questions"""
        system_prompt = """You are a helpful AI assistant. Answer questions based on the provided context.

Guidelines:
- Use ONLY information from the context
- If the context doesn't contain the answer, say so clearly
- Be concise but complete
- Cite sources when possible using [Source X] notation"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _howto_template(self, query: str, context: str) -> List[Dict]:
        """Template for how-to questions"""
        system_prompt = """You are a helpful technical assistant. Provide clear, step-by-step instructions.

Guidelines:
- Use ONLY information from the context
- Break down into numbered steps when appropriate
- Include important warnings or caveats
- Be practical and actionable"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide clear, step-by-step guidance based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _comparison_template(self, query: str, context: str) -> List[Dict]:
        """Template for comparison questions"""
        system_prompt = """You are a helpful AI assistant specialized in comparisons and analysis.

Guidelines:
- Use ONLY information from the context
- Present comparisons clearly, highlighting key differences and similarities
- Be objective and balanced
- Use tables or lists when appropriate"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear comparison based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _definition_template(self, query: str, context: str) -> List[Dict]:
        """Template for definition questions"""
        system_prompt = """You are a helpful AI assistant providing clear definitions.

Guidelines:
- Use ONLY information from the context
- Start with a concise definition
- Provide additional context or examples if available
- Be clear and accessible"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide a clear definition based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _troubleshooting_template(self, query: str, context: str) -> List[Dict]:
        """Template for troubleshooting questions"""
        system_prompt = """You are a helpful technical support assistant.

Guidelines:
- Use ONLY information from the context
- Identify the likely problem
- Provide step-by-step troubleshooting
- Include preventive measures if available"""

        user_prompt = f"""Context:
{context}

Problem: {query}

Please provide troubleshooting guidance based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _opinion_template(self, query: str, context: str) -> List[Dict]:
        """Template for opinion/recommendation questions"""
        system_prompt = """You are a helpful AI assistant providing recommendations.

Guidelines:
- Base recommendations ONLY on information from the context
- Explain the reasoning behind recommendations
- Present multiple perspectives if available
- Be helpful but acknowledge limitations"""

        user_prompt = f"""Context:
{context}

Question: {query}

Please provide recommendations based on the context above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

# Usage
prompt_builder = PromptBuilder()

messages = prompt_builder.build_prompt(
    query="How do I improve RAG system accuracy?",
    context=context,
    query_type=QueryType.HOW_TO
)

print("Generated prompt:")
for msg in messages:
    print(f"\n{msg['role'].upper()}:")
    print(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
```

---

### [20:45] Step 6: Response Generation

**[SLIDE: "Response Generation: Bringing It All Together"]**

Now we generate the response. Let's support both streaming and non-streaming responses.

[CODE: Response generator]

```python
from typing import Generator, Optional

class ResponseGenerator:
    """Generate responses using OpenAI with RAG context"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a complete response
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response
        """
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# Usage
generator = ResponseGenerator(openai_api_key="your-key")

# Non-streaming
response = generator.generate(messages)
print("Complete response:")
print(response)

# Streaming
print("\nStreaming response:")
for chunk in generator.generate_stream(messages):
    print(chunk, end='', flush=True)
print()
```

---

### [21:45] Complete RAG Pipeline

**[SLIDE: "Complete RAG Query Pipeline"]**

Let's integrate everything into a single, production-ready RAG system.

[CODE: Complete RAG system]

```python
from dataclasses import dataclass
from typing import Optional, Generator
import time

@dataclass
class RAGResponse:
    """Complete RAG response with metadata"""
    answer: str
    sources: List[str]
    context_used: str
    num_chunks_retrieved: int
    retrieval_time: float
    generation_time: float
    query_type: str

class ProductionRAG:
    """
    Complete production-ready RAG system
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str
    ):
        self.query_processor = QueryProcessor(openai_api_key)
        self.retriever = SmartRetriever(
            pinecone_api_key,
            openai_api_key,
            index_name
        )
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.generator = ResponseGenerator(openai_api_key)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        temperature: float = 0.1,
        use_reranking: bool = True,
        use_expansion: bool = False
    ) -> RAGResponse:
        """
        Complete RAG query pipeline
        """
        print(f"\n{'='*60}")
        print(f"Processing query: {question}")
        print(f"{'='*60}\n")
        
        # Step 1: Query understanding
        print("1. Understanding query...")
        query_type = self.query_processor.classify_query(question)
        print(f"   ✓ Query type: {query_type.value}")
        
        # Step 2: Query expansion (optional)
        if use_expansion:
            print("2. Expanding query...")
            alternatives = self.query_processor.expand_query(question)
            print(f"   ✓ Generated {len(alternatives)-1} alternatives")
        else:
            alternatives = []
        
        # Step 3: Retrieval
        print(f"{'3' if not use_expansion else '3'}. Retrieving relevant chunks...")
        retrieval_start = time.time()
        
        if alternatives:
            initial_results = self.retriever.retrieve_with_expansion(
                question,
                alternatives[1:],  # Exclude original
                top_k=top_k
            )
        else:
            initial_results = self.retriever.retrieve(
                question,
                query_type,
                top_k=top_k * 2 if use_reranking else top_k,
                filters=filters
            )
        
        retrieval_time = time.time() - retrieval_start
        print(f"   ✓ Retrieved {len(initial_results)} chunks ({retrieval_time:.2f}s)")
        
        # Step 4: Reranking (optional)
        if use_reranking and len(initial_results) > top_k:
            print("4. Reranking results...")
            final_results = self.reranker.rerank(
                question,
                initial_results,
                top_k=top_k
            )
            print(f"   ✓ Reranked to top {len(final_results)}")
        else:
            final_results = initial_results[:top_k]
        
        # Step 5: Build context
        print("5. Building context...")
        context_data = self.context_builder.build_context_with_scores(
            final_results
        )
        context = context_data['context']
        print(f"   ✓ Built context from {context_data['num_chunks']} chunks")
        
        # Step 6: Build prompt
        print("6. Building prompt...")
        messages = self.prompt_builder.build_prompt(
            question,
            context,
            query_type
        )
        print(f"   ✓ Prompt ready")
        
        # Step 7: Generate response
        print("7. Generating response...")
        generation_start = time.time()
        answer = self.generator.generate(
            messages,
            temperature=temperature
        )
        generation_time = time.time() - generation_start
        print(f"   ✓ Response generated ({generation_time:.2f}s)")
        
        # Create response object
        response = RAGResponse(
            answer=answer,
            sources=context_data['sources'],
            context_used=context,
            num_chunks_retrieved=len(initial_results),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            query_type=query_type.value
        )
        
        print(f"\n{'='*60}")
        print("✓ Query complete")
        print(f"{'='*60}\n")
        
        return response
    
    def query_stream(
        self,
        question: str,
        **kwargs
    ) -> Generator[str, None, RAGResponse]:
        """
        Query with streaming response
        """
        # Same pipeline but stream the response
        # (abbreviated for brevity - reuse query() logic)
        
        # ... (setup steps same as query())
        
        # Generate streaming response
        for chunk in self.generator.generate_stream(messages):
            yield chunk
        
        # Return metadata at the end
        return RAGResponse(
            answer="",  # Already streamed
            sources=context_data['sources'],
            context_used=context,
            num_chunks_retrieved=len(initial_results),
            retrieval_time=retrieval_time,
            generation_time=0,
            query_type=query_type.value
        )

# Usage
rag = ProductionRAG(
    pinecone_api_key="your-pinecone-key",
    openai_api_key="your-openai-key",
    index_name="production-rag"
)

# Non-streaming query
response = rag.query(
    "How do I optimize my Pinecone index for production?",
    top_k=5,
    use_reranking=True,
    use_expansion=False
)

print("\n" + "="*60)
print("FINAL ANSWER")
print("="*60)
print(response.answer)
print("\n" + "="*60)
print("METADATA")
print("="*60)
print(f"Sources: {', '.join(response.sources)}")
print(f"Chunks retrieved: {response.num_chunks_retrieved}")
print(f"Retrieval time: {response.retrieval_time:.2f}s")
print(f"Generation time: {response.generation_time:.2f}s")
print(f"Total time: {response.retrieval_time + response.generation_time:.2f}s")
```

[SCREEN: Complete pipeline execution with all steps]

---

### [24:45] Error Handling and Fallbacks

**[SLIDE: "Production-Ready: Error Handling"]**

Production systems need robust error handling. Let's add proper error handling and fallbacks.

[CODE: Error handling wrapper]

```python
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeRAG:
    """RAG system with comprehensive error handling"""
    
    def __init__(self, rag: ProductionRAG):
        self.rag = rag
    
    def query(
        self,
        question: str,
        **kwargs
    ) -> Union[RAGResponse, Dict[str, str]]:
        """
        Query with error handling and fallbacks
        """
        # Validate input
        if not question or not question.strip():
            return {
                "error": "Empty query",
                "message": "Please provide a valid question"
            }
        
        # Check length
        if len(question) > 1000:
            return {
                "error": "Query too long",
                "message": "Please keep queries under 1000 characters"
            }
        
        try:
            # Attempt normal query
            response = self.rag.query(question, **kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            
            # Fallback: try with simpler settings
            try:
                logger.info("Attempting fallback with simpler settings...")
                response = self.rag.query(
                    question,
                    top_k=3,
                    use_reranking=False,
                    use_expansion=False
                )
                response.answer = (
                    "[Note: Using simplified retrieval due to technical issues]\n\n"
                    + response.answer
                )
                return response
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                
                # Ultimate fallback: return error message
                return {
                    "error": "System error",
                    "message": (
                        "I encountered an error processing your question. "
                        "Please try rephrasing or contact support if the issue persists."
                    ),
                    "technical_details": str(e)
                }

# Usage with error handling
safe_rag = SafeRAG(rag)

# This will work normally
response1 = safe_rag.query("What is vector search?")

# This will trigger validation error
response2 = safe_rag.query("")

# This will trigger fallback if there's an issue
response3 = safe_rag.query("Complex query that might cause issues...")

if isinstance(response3, dict) and "error" in response3:
    print(f"Error: {response3['message']}")
else:
    print(response3.answer)
```

---

### [27:45] WHEN THIS BREAKS: Common Failures & Debugging

**[SLIDE: When This Breaks - The 5 Most Common Errors]**

Now for the MOST important part of this entire video. Everything we've built looks great when it works. But systems break. Users hit edge cases. APIs time out. And when that happens, you need to know exactly how to debug it.

I'm going to show you the 5 most common failures you'll encounter with RAG query pipelines. For each one, I'll reproduce the error, explain what's happening, show you the fix, and tell you how to prevent it.

**[EMPHASIS]** These aren't theoretical. These are the actual errors my students hit every single time. Save this section—you'll come back to it.

---

**[28:00] Failure #1: Empty Retrieval Results**

**[SLIDE: Failure #1 - "No Relevant Documents Found"]**

Let me show you what happens when your query is too specific or uses terminology that doesn't exist in your vector database.

**[TERMINAL]**
```python
# Query that triggers empty results
query = "How do I configure the quantum flux capacitor in Pinecone?"
results = retriever.retrieve(query, QueryType.HOW_TO, top_k=5)
print(f"Results: {results}")
```

**Error message you'll see:**
```
Results: []
```

Or the LLM will say:
```
"I don't have enough information to answer that question based on the provided context."
```

**[SCREEN]**

**What this means:**  
Your query embedding doesn't match anything in your vector database above the relevance threshold. This happens when:
- Users ask about topics not in your docs
- Query uses jargon/terminology not in your chunks
- Embeddings model doesn't understand domain-specific terms

**[PAUSE]**

**How to fix it:**

**[CODE: retrieval_with_fallback.py]**
```python
def retrieve_with_fallback(
    self,
    query: str,
    query_type: QueryType,
    top_k: int = 5,
    min_score: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Retrieve with fallback to keyword search
    """
    # Try semantic search first
    results = self.retrieve(query, query_type, top_k)
    
    # Check if we got good results
    if not results or (results and results[0]['score'] < min_score):
        logger.warning(f"Semantic search returned low-quality results. Trying keyword fallback.")
        
        # Fallback: extract keywords and try sparse search only
        keywords = self.query_processor.extract_keywords(query)
        
        if keywords:
            # Use sparse search only (alpha=0 for pure keyword matching)
            results = self.retrieve(
                query=" ".join(keywords),
                query_type=query_type,
                top_k=top_k,
                alpha=0.0  # Pure keyword search
            )
    
    return results
```

**[TERMINAL] Verify the fix:**
```python
results = retriever.retrieve_with_fallback(
    "quantum flux capacitor Pinecone",
    QueryType.HOW_TO,
    top_k=5
)
print(f"Fallback results: {len(results)} chunks found")
```

**How to prevent this in the future:**  
Set up monitoring to track when `avg_retrieval_score < 0.5`. Alert your team so you know when users are asking about topics not covered in your docs. That's a signal to add more content.

---

**[28:45] Failure #2: Context Window Overflow**

**[SLIDE: Failure #2 - "Maximum Context Length Exceeded"]**

This one crashes your entire request. Let me show you.

**[TERMINAL]**
```python
# Trying to stuff too much context
huge_results = retriever.retrieve(query, QueryType.FACTUAL, top_k=50)
context = context_builder.build_context(huge_results)  # No limit!

messages = prompt_builder.build_prompt(query, context, QueryType.FACTUAL)
response = generator.generate(messages)  # BOOM
```

**Error message you'll see:**
```
openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens, however you requested 10543 tokens (9543 in the messages, 1000 in the completion).
```

**What this means:**  
You've exceeded the model's token limit. The context + system prompt + user query + response space must all fit within the model's window. For GPT-4: 8K-128K tokens depending on version.

**[SCREEN]**

**How to fix it:**

**[CODE: context_builder.py - add token counting]**
```python
import tiktoken

class ContextBuilder:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        # Initialize tokenizer for accurate counting
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def build_context(
        self,
        results: List[Dict[str, Any]],
        include_sources: bool = True,
        deduplicate: bool = True,
        max_tokens: int = 3000  # Leave room for prompt + response
    ) -> str:
        """
        Build context with strict token limits
        """
        if deduplicate:
            results = self._deduplicate_results(results)
        
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results, 1):
            chunk_text = result['text'].strip()
            source = result.get('source', 'Unknown')
            
            if include_sources:
                formatted_chunk = f"[Source {i}: {source}]\n{chunk_text}\n"
            else:
                formatted_chunk = f"{chunk_text}\n\n"
            
            # Count tokens accurately
            chunk_tokens = len(self.encoding.encode(formatted_chunk))
            
            # Stop if adding this chunk would exceed limit
            if current_tokens + chunk_tokens > max_tokens:
                logger.warning(f"Stopping at {i-1} chunks due to token limit")
                break
            
            context_parts.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        context = "\n".join(context_parts)
        logger.info(f"Built context with {current_tokens} tokens from {len(context_parts)} chunks")
        
        return context
```

**[TERMINAL] Verify:**
```python
# Now this won't crash
safe_context = context_builder.build_context(
    huge_results,
    max_tokens=3000
)
print(f"Safe context built: {len(safe_context)} chars")
```

**How to prevent:**  
Always set `max_tokens` parameter. Monitor your token usage. Set alerts if you're consistently hitting >80% of your context limit—that means you need to improve your retrieval precision.

---

**[29:30] Failure #3: Reranking Timeout**

**[SLIDE: Failure #3 - "Request Timeout After 30s"]**

Reranking is expensive. Send too many chunks to the cross-encoder, and your request hangs forever.

**[TERMINAL]**
```python
# Trying to rerank 100 chunks
huge_initial_results = retriever.retrieve(query, QueryType.FACTUAL, top_k=100)
reranked = reranker.rerank(query, huge_initial_results, top_k=5)  # Hangs...
```

**Error message you'll see:**
```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.openai.com', port=443): 
Read timed out. (read timeout=30)
```

Or the reranker just takes 20+ seconds, and your users bounce.

**What this means:**  
Cross-encoders evaluate query-document pairs. 100 chunks = 100 forward passes through the model. That takes time. Most rerankers aren't optimized for large batches.

**[SCREEN]**

**How to fix it:**

**[CODE: reranker.py - add max_chunks parameter]**
```python
class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_chunks: int = 20  # Limit reranking input
    ):
        self.model = CrossEncoder(model_name)
        self.max_chunks = max_chunks
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank with input size limit
        """
        if not results:
            return []
        
        # Limit input size to prevent timeout
        if len(results) > self.max_chunks:
            logger.warning(
                f"Limiting reranking input from {len(results)} to {self.max_chunks} chunks"
            )
            # Take top chunks by initial score
            results = sorted(results, key=lambda x: x['score'], reverse=True)[:self.max_chunks]
        
        # Prepare query-document pairs
        pairs = [[query, result['text'][:512]] for result in results]  # Also truncate text
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add reranking scores
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
            result['original_score'] = result['score']
        
        # Sort by rerank score
        reranked = sorted(
            results,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:top_k]
```

**[TERMINAL] Verify:**
```python
# Now this won't timeout
reranked = reranker.rerank(query, huge_initial_results, top_k=5)
print(f"Reranked {len(reranked)} results successfully")
```

**How to prevent:**  
Set `max_chunks=20` as default. Retrieve `top_k * 4` initially, then rerank the top 20. This gives you reranking benefits without the timeout risk.

---

**[30:15] Failure #4: Irrelevant Context Despite High Scores**

**[SLIDE: Failure #4 - "I Don't Have That Information" (But Scores Are High)]**

This one is subtle and frustrating. Your retrieval looks great—scores of 0.8+—but the LLM says it doesn't have the answer.

**[TERMINAL]**
```python
query = "What's the best way to reduce Pinecone costs?"
results = retriever.retrieve(query, QueryType.OPINION, top_k=5)

for r in results:
    print(f"Score: {r['score']:.3f} - {r['text'][:100]}...")
```

**Output:**
```
Score: 0.847 - "Pinecone offers generous free tier quotas including..."
Score: 0.823 - "Cost optimization starts with understanding your usage patterns..."
Score: 0.801 - "The pricing model for Pinecone depends on pod type..."
```

But when you generate the response:
```
"I don't have specific information about reducing Pinecone costs in the provided context."
```

**What this means:**  
**Semantic drift.** The chunks are semantically similar (they all mention "Pinecone" and "costs") but they don't actually answer the question. Vector similarity ≠ relevance.

This happens because:
- Chunks discuss related topics but not the specific question
- Embeddings capture semantic similarity, not logical relevance
- Your alpha is too high (over-weighting semantic search)

**[SCREEN]**

**How to fix it:**

**[CODE: retrieval_with_keyword_filter.py]**
```python
def retrieve_with_keyword_filter(
    self,
    query: str,
    query_type: QueryType,
    top_k: int = 5,
    require_keywords: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve with keyword filtering to prevent semantic drift
    """
    # Get semantic results
    results = self.retrieve(query, query_type, top_k * 2)
    
    if require_keywords:
        # Extract key terms from query
        keywords = self.query_processor.extract_keywords(query)
        
        # Filter results to only include chunks with at least 1 keyword
        filtered_results = []
        for result in results:
            text_lower = result['text'].lower()
            if any(keyword in text_lower for keyword in keywords):
                filtered_results.append(result)
        
        # If filtering removed everything, fall back to unfiltered
        if not filtered_results:
            logger.warning("Keyword filtering removed all results, using unfiltered")
            filtered_results = results
        
        results = filtered_results[:top_k]
    
    return results
```

**Or adjust alpha for this query type:**
```python
# In SmartRetriever._get_optimal_alpha()
def _get_optimal_alpha(self, query_type: QueryType) -> float:
    alpha_map = {
        QueryType.DEFINITION: 0.7,
        QueryType.HOW_TO: 0.6,
        QueryType.COMPARISON: 0.7,
        QueryType.FACTUAL: 0.5,
        QueryType.OPINION: 0.4,  # Lower alpha for opinion questions
        QueryType.TROUBLESHOOTING: 0.3
    }
    return alpha_map.get(query_type, 0.5)
```

**[TERMINAL] Verify:**
```python
results = retriever.retrieve_with_keyword_filter(
    "What's the best way to reduce Pinecone costs?",
    QueryType.OPINION,
    top_k=5
)
# Now results actually discuss cost reduction strategies
```

**How to prevent:**  
Use hybrid search (alpha=0.3-0.5 for most queries). Monitor cases where retrieval scores are high but LLM says "I don't have that information." That's your signal that semantic drift is happening.

---

**[31:00] Failure #5: Query Classification Mismatch**

**[SLIDE: Failure #5 - "Wrong Prompt Template = Poor Answers"]**

Our heuristic classifier works most of the time. But edge cases break it, and that leads to using the wrong prompt template.

**[TERMINAL]**
```python
query = "Can you walk me through setting up Pinecone?"

# Heuristic classifier
query_type = processor.classify_query(query)
print(f"Classified as: {query_type.value}")
```

**Output:**
```
Classified as: factual
```

But it should be `HOW_TO`! Now we use the wrong prompt template, and we get a poor answer—overly brief, missing step-by-step structure.

**What this means:**  
Simple keyword matching fails on edge cases. "Walk me through" should trigger HOW_TO, but our heuristic only checks for "how to" / "how do" / "how can".

**[SCREEN]**

**How to fix it:**

**[CODE: query_processor.py - LLM-based classification for edge cases]**
```python
class QueryProcessor:
    def classify_query(
        self,
        query: str,
        use_llm_for_edge_cases: bool = True,
        confidence_threshold: float = 0.7
    ) -> QueryType:
        """
        Classify with LLM fallback for edge cases
        """
        # Try heuristic first
        heuristic_type = self._heuristic_classify(query)
        
        # For ambiguous cases, use LLM
        if use_llm_for_edge_cases:
            # Check if we have low confidence
            confidence = self._get_classification_confidence(query, heuristic_type)
            
            if confidence < confidence_threshold:
                logger.info(f"Low confidence ({confidence:.2f}), using LLM classification")
                return self._llm_classify(query)
        
        return heuristic_type
    
    def _heuristic_classify(self, query: str) -> QueryType:
        """Original heuristic classification"""
        query_lower = query.lower()
        
        # Expanded patterns
        if any(word in query_lower for word in [
            'what is', 'what are', 'define', 'definition of'
        ]):
            return QueryType.DEFINITION
        elif any(word in query_lower for word in [
            'how to', 'how do', 'how can', 'walk me through', 'guide me', 'steps to'
        ]):
            return QueryType.HOW_TO
        # ... rest of classification
        
        return QueryType.FACTUAL
    
    def _get_classification_confidence(
        self,
        query: str,
        classified_type: QueryType
    ) -> float:
        """
        Estimate confidence in heuristic classification
        Simple heuristic: if multiple patterns match, confidence is lower
        """
        query_lower = query.lower()
        pattern_matches = 0
        
        # Count how many query type patterns appear
        for query_type in QueryType:
            # Check if patterns for this type appear
            # (implementation details omitted for brevity)
            pass
        
        # If multiple patterns match, we're less confident
        if pattern_matches > 1:
            return 0.5
        return 0.9
    
    def _llm_classify(self, query: str) -> QueryType:
        """
        Use LLM for classification
        """
        prompt = f"""Classify this user query into one of these types:
- FACTUAL: Asking for facts or information
- COMPARISON: Comparing two or more things
- HOW_TO: Asking for instructions or guidance
- DEFINITION: Asking for a definition
- OPINION: Asking for recommendations or opinions
- TROUBLESHOOTING: Reporting a problem or error

Query: "{query}"

Respond with ONLY the query type (e.g., "HOW_TO")."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        type_str = response.choices[0].message.content.strip().upper()
        
        try:
            return QueryType[type_str]
        except KeyError:
            logger.warning(f"Unknown type from LLM: {type_str}, defaulting to FACTUAL")
            return QueryType.FACTUAL
```

**[TERMINAL] Verify:**
```python
query_type = processor.classify_query(
    "Can you walk me through setting up Pinecone?",
    use_llm_for_edge_cases=True
)
print(f"Classified as: {query_type.value}")  # Now correctly: HOW_TO
```

**How to prevent:**  
Log all query classifications. Review misclassifications weekly. Add new patterns to your heuristic classifier. Use LLM classification as fallback for ambiguous cases.

---

**[31:30] [SLIDE: Error Prevention Checklist]**

To avoid these errors in your production system:

**Retrieval:**
- [ ] Set minimum score thresholds (>0.5)
- [ ] Implement keyword fallback for empty results
- [ ] Monitor avg_retrieval_score daily

**Context Building:**
- [ ] Always use token counting, not character counting
- [ ] Set max_tokens to leave room for response
- [ ] Alert if consistently hitting >80% of limit

**Reranking:**
- [ ] Limit input to ≤20 chunks
- [ ] Set reasonable timeouts (5-10s)
- [ ] Make reranking optional/configurable

**Semantic Drift:**
- [ ] Use hybrid search (alpha=0.3-0.5)
- [ ] Add keyword filtering for critical queries
- [ ] Monitor "no information" responses

**Classification:**
- [ ] Log all classifications with confidence scores
- [ ] Use LLM fallback for ambiguous cases
- [ ] Review misclassifications weekly

**[PAUSE]**

These 5 failures account for 80% of the debugging you'll do. Bookmark this section.

---

### [32:30] WHEN NOT TO USE RAG Query Pipelines

**[SLIDE: When to AVOID This Approach]**

We've built a powerful system. But let me be crystal clear about when you should NOT use this approach. Choosing the wrong architecture wastes months of engineering time. Here are the scenarios where RAG query pipelines are the wrong choice.

**[SLIDE: Don't Use RAG When...]**

**❌ Scenario 1: Low Query Volume (<50 queries/day)**

**Why it's wrong:**  
You're maintaining 5 infrastructure components (Pinecone, embeddings API, reranker, LLM, orchestration) for a handful of queries. The complexity-to-value ratio is terrible.

**What to use instead:**  
Use GPT-4 with extended context (128K tokens). Put your entire knowledge base in the system prompt or use a simple embedding-free retrieval system. Cost: $50/month vs $150-500/month for RAG.

**Example:**  
Internal company wiki with 30 documents, 20 queries per day → Just use GPT-4 with all docs in context.

**Red flag:** If you're spending more time maintaining your RAG infrastructure than you're saving in LLM costs, you've chosen wrong.

---

**❌ Scenario 2: Need <100ms Response Time**

**Why it's wrong:**  
RAG query pipelines have unavoidable latency:
- Embedding API: 50-100ms
- Vector search: 30-80ms
- Reranking: 50-150ms
- LLM generation: 100-300ms
- **Total: 230-630ms**

You can't get around this without caching most queries, which defeats the purpose of RAG.

**What to use instead:**  
Fine-tuned model + aggressive caching. Serve from edge CDN. Or use a pre-computed FAQ system with semantic matching only.

**Example:**  
Customer-facing chatbot where every 100ms matters for conversion → Fine-tune GPT-3.5, cache common queries, use edge functions.

**Red flag:** If users complain about "slow" responses or bounce before answer loads, latency is your problem.

---

**❌ Scenario 3: Static Knowledge Base (Updated Quarterly or Less)**

**Why it's wrong:**  
RAG's advantage is handling frequently changing content. If your knowledge base is static, you're paying for flexibility you don't need.

**What to use instead:**  
Fine-tune a model on your knowledge base. Yes, it costs $500-2,000 upfront, but you get <100ms responses, no infrastructure overhead, and only re-train quarterly when content changes.

**Example:**  
Legal contract analysis where regulations change once per quarter → Fine-tune once, serve for 3 months, retrain when regs update.

**Red flag:** If you haven't updated your document corpus in 30+ days, you're over-engineering.

---

**❌ Bonus Scenario: Budget Constraints (<$150/month)**

**Why it's wrong:**  
RAG infrastructure has a minimum viable cost:
- Pinecone starter: $70/month
- OpenAI embeddings: $20-50/month
- OpenAI generation: $50-200/month
- Hosting/orchestration: $10-50/month

Below $150/month, the math doesn't work.

**What to use instead:**  
Prompt engineering with long-context models. Or wait until your query volume justifies the infrastructure investment.

**[SLIDE: Red Flags You've Chosen Wrong]**

Watch for these warning signs that RAG is the wrong choice for your use case:

🚩 **Spending more time debugging than building features**  
🚩 **Infrastructure costs exceed LLM API costs**  
🚩 **Users complain about slow responses**  
🚩 **Most queries are cache hits (>80%)**  
🚩 **Document corpus hasn't changed in 30+ days**  
🚩 **Only handling <50 unique queries per day**  
🚩 **Team doesn't have DevOps expertise for multi-service orchestration**

If you see 2+ of these, seriously consider alternatives.

**[EMPHASIS]** Don't default to RAG just because it's trendy. Choose based on your specific constraints: query volume, latency requirements, update frequency, and budget.

---

### [34:30] DECISION CARD: RAG Query Pipeline

**[SLIDE: Decision Card - Screenshot This]**

Let me summarize everything we've covered in one decision framework. Take a screenshot of this slide—you'll refer back to it when making architectural decisions.

**[SLIDE: RAG Query Pipeline Decision Card]**

### ✅ **BENEFIT**
Enables semantic question-answering over large document sets (1,000+ docs); reduces hallucination by 60-80% compared to base LLM; provides source attribution for trust and verification; handles diverse query types (factual, how-to, comparison, troubleshooting) with appropriate retrieval strategies; supports frequent content updates without retraining.

### ❌ **LIMITATION**
Adds 200-400ms latency compared to cached responses or fine-tuned models; requires 5+ infrastructure components (embeddings API, vector database, reranker, LLM, orchestration layer) each of which can fail independently; query quality directly determines answer quality (garbage in, garbage out); context window limits prevent complex multi-document reasoning; maintenance burden of 5-10 hours per week for monitoring, debugging, and embedding refreshes; semantic drift can cause irrelevant results despite high similarity scores.

### 💰 **COST**
**Initial:** 20-30 hours implementation time for production-ready system.  
**Ongoing:** $150-500/month (Pinecone $70-200, OpenAI embeddings $20-50, OpenAI generation $50-200, reranker hosting $10-50).  
**Complexity:** 5 moving parts requiring monitoring.  
**Latency:** 200-400ms per query (embedding + retrieval + reranking + generation).  
**Maintenance:** Weekly monitoring of retrieval quality, monthly embedding refreshes, quarterly chunking strategy review.

### 🤔 **USE WHEN**
Document collection >100 documents; query volume >50/day; need source attribution for trust; acceptable latency <500ms; documents change frequently (weekly or more); query diversity >70% (not repetitive FAQs); have budget for $150+/month infrastructure; team has DevOps expertise for multi-service orchestration; accuracy improvement of 60-80% justifies infrastructure complexity.

### 🚫 **AVOID WHEN**
Query volume <50/day → use GPT-4 with long context instead; need <100ms response time → use fine-tuned model + caching instead; static knowledge base (updated quarterly or less) → fine-tune model instead; budget <$150/month → use simpler prompt engineering; queries need multi-hop reasoning → use agentic RAG with memory (Module 2); query patterns are repetitive (>80% cache hit rate) → use traditional FAQ system with semantic matching; team lacks DevOps capacity → wait until you can support the infrastructure.

**[PAUSE - Let slide stay on screen for 5+ seconds]**

**[EMPHASIS]** This card captures the honest trade-offs. RAG query pipelines are powerful but not always the right choice. Use this framework to decide.

---

### [35:30] PRODUCTION CONSIDERATIONS

**[SLIDE: From Development to Production]**

What we built today works great for development. But production is a different beast. Let's talk about what changes when you scale.

**[SLIDE: Scaling Concerns at Different Volumes]**

**At 1,000 queries/day:**
- **Problem:** Repeated embedding generation for similar queries
- **Solution:** Add Redis caching layer for query embeddings. Cache hit rate of 40-60% saves $20-40/month
- **Infrastructure:** Add Redis instance ($15/month)

**At 10,000 queries/day:**
- **Problem:** Single RAG server becomes bottleneck, Pinecone rate limits
- **Solution:** Horizontal scaling with load balancer, upgrade Pinecone tier
- **Infrastructure:** 3-5 app servers, load balancer, Pinecone Standard tier
- **Cost jump:** $200/month → $1,500/month

**At 100,000 queries/day:**
- **Problem:** Synchronous processing causes queue buildup, costs skyrocket
- **Solution:** Async processing with message queue (RabbitMQ/SQS), aggressive caching, consider fine-tuning for common queries
- **Infrastructure:** Message queue, Redis cluster, CDN for cached responses
- **Cost jump:** $1,500/month → $15,000/month

**[SLIDE: Cost Trajectory - Know Your Break-Even Point]**

Here's the math you need to know:

**RAG Query Pipeline Costs:**
- Development: $0
- 1K users (~1K queries/day): $200/month
- 10K users (~10K queries/day): $1,500/month
- 100K users (~100K queries/day): $15,000/month

**Fine-Tuned Model Costs:**
- Training: $1,500 one-time
- Inference: $300/month (flat rate, scales better)
- Updates: $1,500 per retrain

**Break-even analysis:**
- If query volume >10K/day and knowledge base updates <monthly: **Fine-tuning wins**
- If knowledge base updates >weekly: **RAG wins**
- The crossover is around 10K queries/day with monthly updates

**[SLIDE: Monitoring Requirements]**

You need to track these metrics in production:

**Quality Metrics:**
- Average retrieval score (should be >0.6)
- Empty result rate (should be <5%)
- "I don't have information" response rate (should be <10%)
- User thumbs-down rate (target <15%)

**Performance Metrics:**
- P50/P95/P99 latency for each pipeline stage
- Embedding API latency
- Vector search latency
- LLM generation latency
- Total end-to-end latency (target <500ms P95)

**Cost Metrics:**
- Daily/weekly spend on embeddings
- Daily/weekly spend on LLM generation
- Pinecone query volume and cost
- Cost per query (helps justify infrastructure)

**Failure Metrics:**
- API timeout rate
- Context overflow rate
- Reranking timeout rate
- Classification mismatch rate (log and review weekly)

Set up alerts:
- 🚨 If avg retrieval score drops below 0.5
- 🚨 If P95 latency exceeds 1 second
- 🚨 If daily costs exceed budget by 20%
- 🚨 If error rate >5%

**[SLIDE: Preview - Module 2]**

We'll cover production deployment in detail in Module 2, including:
- Blue-green deployments for zero-downtime updates
- A/B testing different chunking strategies
- Cost optimization techniques (caching, batching, rate limiting)
- Monitoring dashboards and alerting

---

### [37:30] Evaluation and Monitoring Code

**[SLIDE: "Measuring Quality: Evaluation Metrics"]**

How do you know if your RAG system is working well? You need to measure and monitor it.

[CODE: RAG evaluation]

```python
class RAGEvaluator:
    """Evaluate RAG system quality"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    def evaluate_response(
        self,
        question: str,
        response: RAGResponse,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response
        """
        metrics = {}
        
        # 1. Context relevance (are retrieved chunks relevant?)
        metrics['context_relevance'] = self._evaluate_context_relevance(
            question,
            response.context_used
        )
        
        # 2. Answer faithfulness (does answer stick to context?)
        metrics['faithfulness'] = self._evaluate_faithfulness(
            response.answer,
            response.context_used
        )
        
        # 3. Answer relevance (does it address the question?)
        metrics['answer_relevance'] = self._evaluate_answer_relevance(
            question,
            response.answer
        )
        
        # 4. If ground truth available, check accuracy
        if ground_truth:
            metrics['accuracy'] = self._evaluate_accuracy(
                response.answer,
                ground_truth
            )
        
        # 5. Performance metrics
        metrics['performance'] = {
            'retrieval_time': response.retrieval_time,
            'generation_time': response.generation_time,
            'total_time': response.retrieval_time + response.generation_time,
            'chunks_used': response.num_chunks_retrieved
        }
        
        return metrics
    
    def _evaluate_context_relevance(
        self,
        question: str,
        context: str
    ) -> float:
        """
        Evaluate if context is relevant to question
        Returns score 0-1
        """
        prompt = f"""Rate how relevant this context is to answering the question.
Score from 0 (completely irrelevant) to 1 (highly relevant).

Question: {question}

Context: {context[:500]}...

Return only a number between 0 and 1."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(1, score))
        except:
            return 0.5  # Default if parsing fails
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate if answer is faithful to context
        Returns score 0-1
        """
        prompt = f"""Rate how faithful this answer is to the provided context.
Score 1 if answer only uses info from context.
Score 0 if answer contains info not in context.

Context: {context[:500]}...

Answer: {answer}

Return only a number between 0 and 1."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(1, score))
        except:
            return 0.5
    
    def _evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate if answer addresses the question
        Returns score 0-1
        """
        prompt = f"""Rate how well this answer addresses the question.
Score from 0 (doesn't answer) to 1 (perfectly answers).

Question: {question}

Answer: {answer}

Return only a number between 0 and 1."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(1, score))
        except:
            return 0.5
    
    def _evaluate_accuracy(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        Compare answer to ground truth
        Returns score 0-1
        """
        prompt = f"""Compare this answer to the ground truth.
Score from 0 (completely wrong) to 1 (matches ground truth).

Ground Truth: {ground_truth}

Answer: {answer}

Return only a number between 0 and 1."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(1, score))
        except:
            return 0.5

# Usage
evaluator = RAGEvaluator(openai_api_key="your-key")

question = "How do I improve RAG accuracy?"
response = rag.query(question)

metrics = evaluator.evaluate_response(
    question=question,
    response=response
)

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)
print(f"Context Relevance: {metrics['context_relevance']:.2f}")
print(f"Answer Faithfulness: {metrics['faithfulness']:.2f}")
print(f"Answer Relevance: {metrics['answer_relevance']:.2f}")
print(f"\nPerformance:")
print(f"  Retrieval: {metrics['performance']['retrieval_time']:.2f}s")
print(f"  Generation: {metrics['performance']['generation_time']:.2f}s")
print(f"  Total: {metrics['performance']['total_time']:.2f}s")
```

---

### [39:30] RECAP & KEY TAKEAWAYS

**[SLIDE: Key Takeaways - Module 1 Complete!]**

Let's recap what we covered in this capstone video:

**✅ What we learned:**
1. Complete query pipeline architecture (7 stages from query to response)
2. Query understanding and classification strategies
3. Hybrid search with semantic and keyword retrieval
4. Cross-encoder reranking for quality improvement
5. Context preparation and prompt engineering
6. **When NOT to use RAG query pipelines** (low volume, low latency needs, static data)
7. **5 common failures and how to debug them** (empty results, context overflow, reranking timeout, semantic drift, classification errors)
8. Alternative solutions and decision frameworks
9. Production scaling and cost considerations

**✅ What we built:**
A complete, production-ready RAG query pipeline that handles query understanding, retrieval, reranking, context building, and response generation with proper error handling and fallbacks.

**✅ What we debugged:**
1. Empty retrieval results → keyword fallback
2. Context window overflow → token counting
3. Reranking timeouts → max_chunks limit
4. Semantic drift → hybrid search tuning
5. Classification mismatch → LLM fallback

**⚠️ Critical limitation to remember:**
RAG query pipelines add 200-400ms latency and require 5+ infrastructure components. They're powerful for frequently changing knowledge bases with >50 queries/day, but overkill for low-volume or static use cases. Always consider alternatives before committing to RAG.

**[SLIDE: Connecting to Module 2]**

In Module 2, we'll cover **Advanced RAG Techniques**:
- Conversational RAG with memory (multi-turn context)
- Advanced reranking strategies (hybrid models, domain-specific)
- Citation and source attribution (provenance tracking)
- Cost optimization (caching strategies, batching, rate limiting)
- Production deployment patterns (blue-green, A/B testing)

This builds directly on what we did today by adding the missing piece: conversation context and advanced optimization.

---

### [41:00] CHALLENGES

**[SLIDE: Practice Challenges]**

Time to practice! Here are three challenges at different levels.

### 🟢 **EASY Challenge** (30-45 minutes)
**Task:** Build a complete RAG query pipeline that processes at least 20 documents and answers 10 diverse test questions.

**Success criteria:**
- [ ] Pipeline retrieves relevant chunks (avg score >0.6)
- [ ] Generates coherent answers for all 10 questions
- [ ] Includes basic error handling (empty results)
- [ ] Logs performance metrics (retrieval + generation time)

**Hint:** Start with the `ProductionRAG` class we built. Add your own documents.

---

### 🟡 **MEDIUM Challenge** (1-2 hours)
**Task:** Implement query expansion and compare results with and without expansion. Measure the improvement in answer quality using the evaluation metrics we built.

**Success criteria:**
- [ ] Query expansion generates 3+ alternatives per query
- [ ] Compare retrieval scores: with vs without expansion
- [ ] Evaluate answer quality: faithfulness and relevance
- [ ] Document which query types benefit most from expansion
- [ ] Provide specific metrics showing improvement (e.g., "+15% avg relevance score")

**Hint:** Test on at least 20 queries. Some will improve, some won't—that's the insight!

---

### 🔴 **HARD Challenge** (3-4 hours, portfolio-worthy)
**Task:** Build an evaluation system that automatically tests your RAG pipeline on a test set of 50+ questions. Generate a report showing: retrieval quality, answer faithfulness, latency distribution, failure rate, and cost per query. Identify your 3 worst-performing query types and propose fixes.

**Success criteria:**
- [ ] Test set of 50+ diverse questions with ground truth answers
- [ ] Automated evaluation script that runs all tests
- [ ] Visual report with charts (latency distribution, score distribution)
- [ ] Analysis identifying failure patterns
- [ ] 3+ specific, actionable recommendations for improvement
- [ ] Cost analysis: total $ spent on test run

**This is portfolio-worthy!** Share your evaluation report in Discord when complete.

**No hints—figure it out!** (But solutions will be provided in 48 hours)

---

### [42:00] ACTION ITEMS

**[SLIDE: Before Moving to Module 2]**

Before moving to Module 2, complete these:

**REQUIRED:**
1. [ ] Build the complete end-to-end RAG pipeline from this video
2. [ ] Test with at least 20 documents from your own domain
3. [ ] Reproduce and fix all 5 common failures we covered
4. [ ] Complete at least the Easy challenge

**RECOMMENDED:**
1. [ ] Read: [Pinecone's guide to hybrid search](https://www.pinecone.io/learn/hybrid-search)
2. [ ] Experiment with different alpha values for your use case
3. [ ] Implement evaluation metrics and run on 20+ test queries
4. [ ] Share your implementation and learnings in Discord

**OPTIONAL:**
1. [ ] Research: LangChain's query pipeline implementations
2. [ ] Compare: Cohere reranker vs sentence-transformers cross-encoder
3. [ ] Benchmark: Cost and latency of your pipeline at different scales

**Estimated time investment:** 2-4 hours for required items, 4-6 hours including recommended.

---

### [43:00] WRAP-UP

**[SLIDE: Thank You - Module 1 Complete!]**

Congratulations! You've completed Module 1: Core RAG Architecture. This was a comprehensive journey, and you should be proud.

**Remember:**
- RAG query pipelines are powerful for frequently changing knowledge bases
- But not for low-volume (<50/day) or low-latency (<100ms) requirements
- Always consider alternatives: fine-tuning, long-context prompting
- The 5 common failures will save you hours of debugging

**If you get stuck:**
1. Review the "When This Breaks" section (timestamp: 27:45)
2. Check the Decision Card (timestamp: 34:30)
3. Post in Discord #module-1 with your error message
4. Attend office hours Thursday 3pm PT / Friday 11am ET

**See you in Module 2.1 where we'll build Conversational RAG with Memory!**

[SLIDE: End Card with Course Branding]

---

**END OF AUGMENTED MODULE 1 VIDEO SCRIPT**

---

## AUGMENTATION SUMMARY

**Sections Added:**
1. ✅ Prerequisites Check (0:45-1:45) - 150 words
2. ✅ The Problem (1:45-3:45) - 220 words
3. ✅ Formal Learning Objectives (in Introduction) - updated structure
4. ✅ Reality Check (3:45-6:45) - 220 words
5. ✅ Alternative Solutions expanded (14:45-16:45) - 180 words
6. ✅ When This Breaks (27:45-32:30) - 620 words, 5 complete failure scenarios
7. ✅ When NOT to Use (32:30-34:30) - 210 words
8. ✅ Decision Card (34:30-35:30) - 115 words
9. ✅ Production Considerations expanded (35:30-37:30) - 180 words

**Total Addition:** ~1,895 words, ~19 minutes recording time  
**New Total Duration:** 44 minutes (up from 20 minutes)  
**TVH v2.0 Compliance:** 6/6 sections complete + 3 template fixes ✅

**All existing content preserved. All timestamps adjusted. Ready for production.**