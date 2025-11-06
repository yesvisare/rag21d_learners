# Video M2.2: Prompt Optimization & Model Selection (40 min)
**Duration:** 38-40 min | **Audience:** Intermediate | **Prereqs:** M2.1 (Caching), Working RAG system

---

## OBJECTIVES
By the end of this video, learners will be able to:
- Engineer RAG-specific prompts that reduce token usage by 30-50%
- Implement intelligent model routing to match query complexity
- Apply token optimization techniques without sacrificing quality
- **Recognize when NOT to use prompt optimization** and what alternatives exist
- Debug the 5 most common prompt optimization failures

---

<!-- ============================================================ -->
<!-- INSERTION #1: PREREQUISITE CHECK -->
<!-- Added per audit - validates setup before teaching -->
<!-- ============================================================ -->

## PREREQUISITE CHECK

**[0:00] [SLIDE: Prerequisites]**

Before we dive into prompt optimization, let's make sure you have everything you need:

**Required:**
- [ ] Completed: M2.1 (Caching Strategies)
- [ ] Have working: RAG system from M1.x with vector database
- [ ] Installed: `openai>=1.0.0`, `tiktoken>=0.5.0`
- [ ] API access: OpenAI API key with credits available
- [ ] Environment: Python 3.9+ with ability to run code examples

**Estimated time:** 38-40 minutes for this video + 60-90 minutes practice

**Quick validation:**

[TERMINAL]
```bash
# Verify installations
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import tiktoken; print('Tiktoken: OK')"

# Test API access
echo $OPENAI_API_KEY | head -c 10
# Should show: sk-proj-XX
```

**Expected output:**
```
OpenAI: 1.12.0
Tiktoken: OK
sk-proj-XX
```

If any checks fail, pause here and set up your environment. We'll be running real code examples that bill to your API account, so make sure everything works before proceeding.

---

## INTRO (1:00-3:00)

**[1:00] [SLIDE: "M2.2: Prompt Optimization & Model Selection"]**

Welcome back! In M2.1, we cut costs with caching. Now we're going to optimize the most expensive part of your RAG system: the LLM calls themselves.

Here's a truth bomb: **Your prompt is costing you money**. Every token you send costs money. Every token the model generates costs money. And if your prompt isn't optimized? You're paying for unnecessary tokens while getting worse results.

**[1:30] [SLIDE: "Token Cost Reality Check"]**
```
Bad Prompt (350 tokens in + 200 out):
  550 tokens × $0.000003 = $0.00165 per query
  10,000 queries/day = $16.50/day = $495/month

Optimized Prompt (180 tokens in + 150 out):
  330 tokens × $0.000003 = $0.00099 per query
  10,000 queries/day = $9.90/day = $297/month

Savings: $198/month (40% reduction)
Just from optimizing prompts!
```

**What we're building today:**
A production-ready prompt optimization system with intelligent model routing and token management.

**Why this matters:**
Prompt optimization is the fastest way to cut costs without touching infrastructure. But it comes with trade-offs we need to understand.

**What we'll cover:**
1. Prompt engineering specifically for RAG
2. Choosing the right model for each task
3. Token optimization techniques
4. **Important:** When NOT to use these optimizations and what breaks

**Connection to M2.1:**
Last video we cached responses. This video optimizes what happens when cache misses occur - making those LLM calls cheaper and faster.

Let's make your prompts work harder so your wallet doesn't have to.

---

<!-- ============================================================ -->
<!-- INSERTION #2: REALITY CHECK -->
<!-- Added per audit - sets honest expectations early -->
<!-- ============================================================ -->

## REALITY CHECK: What Prompt Optimization Actually Does (3:00-5:30)

**[3:00] [SLIDE: Reality Check - Let's Be Honest]**

Before we get excited about those cost savings, let's talk about what prompt optimization can and cannot do. This is critical.

**[PAUSE]**

**What prompt optimization DOES well:**
- ✅ **Reduces token usage 30-50%** - Measured savings across real production systems without changing infrastructure
- ✅ **Cuts API costs proportionally** - If you're spending $500/month on LLM calls, optimization can bring that to $250-350/month
- ✅ **Improves response latency 10-20%** - Fewer tokens means faster generation, reducing time-to-first-token

**[3:45] What it DOESN'T do:**

- ❌ **Cannot fix poor retrieval quality** - If your vector search returns irrelevant documents, no prompt engineering will salvage the output. Garbage in, garbage out.
- ❌ **Won't improve response quality beyond baseline** - Optimization trades verbosity for conciseness. You might actually lose nuance with aggressive optimization.
- ❌ **Doesn't solve scaling bottlenecks** - If your system is slow because of database queries or network latency, prompt optimization won't help.

**[EMPHASIS]** Here's what nobody tells you: **Aggressive prompt optimization can degrade answer quality**. When you cut a prompt from 350 tokens to 140 tokens, you're removing context and instructions. Sometimes that context matters.

**[4:15] [SLIDE: The Trade-Offs You're Making]**

**The real trade-offs:**
- You gain **cost savings** but risk **response quality degradation**
- Works great for **high-volume, simple queries** but poorly for **complex reasoning tasks**
- Saves money in **production** but adds **development and monitoring overhead**

**Cost structure honesty:**
- Upfront: 4-8 hours to implement and test prompt variants
- Ongoing: 2-4 hours/month monitoring performance and tuning
- Hidden cost: Need A/B testing infrastructure to measure impact

**[5:00] [DIAGRAM: Quality vs Cost Curve]**

[Draw curve showing diminishing returns - aggressive optimization saves tokens but quality drops sharply after a certain point]

We'll see these trade-offs throughout the video. The goal isn't to optimize aggressively - it's to find the sweet spot where you save money without hurting user experience.

---

## RAG-SPECIFIC PROMPT ENGINEERING (5:30-9:00)

**[5:30] [SLIDE: "What Makes RAG Prompts Different?"]**

RAG prompts are unique because you're always dealing with retrieved context. The challenge is getting the model to:
1. Actually use the context you provide
2. Not hallucinate beyond the context
3. Handle irrelevant or contradictory retrieved documents
4. Stay concise to reduce output tokens

**[6:00] [CODE: "rag_prompt_templates.py"]**

```python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """
    Structured prompt template for RAG systems.
    """
    system_prompt: str
    user_template: str
    tokens_estimate: int
    use_case: str

class RAGPromptLibrary:
    """
    Production-tested prompt templates for different RAG scenarios.
    """
    
    # BASELINE (Non-optimized)
    BASIC_RAG = PromptTemplate(
        system_prompt="""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain enough information to answer fully, say so. Be comprehensive and detailed in your response.""",
        
        user_template="""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above.""",
        
        tokens_estimate=350,  # Typical input tokens
        use_case="baseline_comparison"
    )
    
    # OPTIMIZED VERSION 1: Concise
    CONCISE_RAG = PromptTemplate(
        system_prompt="""Answer based solely on provided context. If uncertain, state "Insufficient information." Be concise.""",
        
        user_template="""Context:
{context}

Q: {question}
A:""",
        
        tokens_estimate=180,  # ~50% reduction
        use_case="cost_optimization"
    )
    
    # OPTIMIZED VERSION 2: Structured Output
    STRUCTURED_RAG = PromptTemplate(
        system_prompt="""Extract information from context to answer questions. Format: Answer in 1-2 sentences, then list key points if relevant. If context insufficient, respond: "Not in context." """,
        
        user_template="""Context: {context}

Question: {question}
Answer:""",
        
        tokens_estimate=160,
        use_case="structured_responses"
    )
    
    # OPTIMIZED VERSION 3: JSON Output (for programmatic use)
    JSON_RAG = PromptTemplate(
        system_prompt="""Return JSON only: {"answer": "...", "confidence": "high|medium|low", "sources": [...]}. Use context provided.""",
        
        user_template="""{context}

Q: {question}
JSON:""",
        
        tokens_estimate=140,
        use_case="api_integration"
    )
    
    # OPTIMIZED VERSION 4: Domain-Specific (Customer Support)
    SUPPORT_RAG = PromptTemplate(
        system_prompt="""You're a support agent. Answer using the knowledge base context. Be helpful but brief. If context lacks info, offer to escalate.""",
        
        user_template="""KB Articles:
{context}

Customer: {question}
Agent:""",
        
        tokens_estimate=170,
        use_case="customer_support"
    )
    
    # OPTIMIZED VERSION 5: Few-Shot RAG
    FEW_SHOT_RAG = PromptTemplate(
        system_prompt="""Answer based on context. Examples:
Q: What's the return policy?
A: 30-day returns with receipt. Exceptions: personalized items.

Q: Shipping time?
A: 3-5 business days standard, 1-2 days express.""",
        
        user_template="""Context: {context}

Q: {question}
A:""",
        
        tokens_estimate=210,  # Higher but better quality
        use_case="improved_accuracy"
    )
    
    # ADVANCED: Multi-Document RAG
    MULTI_DOC_RAG = PromptTemplate(
        system_prompt="""Synthesize information from multiple sources. Cite source numbers [1], [2], etc. If sources conflict, note discrepancies.""",
        
        user_template="""Sources:
{context}

Query: {question}
Synthesis:""",
        
        tokens_estimate=190,
        use_case="research_synthesis"
    )

def format_context_optimally(
    documents: List[Dict],
    max_tokens: int = 1500,
    include_metadata: bool = False
) -> str:
    """
    Format retrieved documents to minimize token usage while
    maintaining quality.
    """
    formatted_docs = []
    total_tokens = 0
    
    for i, doc in enumerate(documents, 1):
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        content = doc['content']
        doc_tokens = len(content) // 4
        
        if total_tokens + doc_tokens > max_tokens:
            # Truncate to fit
            remaining_chars = (max_tokens - total_tokens) * 4
            content = content[:remaining_chars] + "..."
            
        if include_metadata:
            formatted_doc = f"[{i}] (Score: {doc.get('score', 0):.2f})\n{content}"
        else:
            formatted_doc = f"[{i}] {content}"
        
        formatted_docs.append(formatted_doc)
        total_tokens += len(formatted_doc) // 4
        
        if total_tokens >= max_tokens:
            break
    
    return "\n\n".join(formatted_docs)

def optimize_query_for_retrieval(query: str) -> str:
    """
    Optimize user query for better vector search results.
    This runs BEFORE embedding.
    """
    # Remove common filler words that don't add semantic value
    filler_words = {'um', 'uh', 'like', 'you know', 'basically', 'actually'}
    
    words = query.lower().split()
    filtered = [w for w in words if w not in filler_words]
    
    # Expand common abbreviations for better matching
    expansions = {
        'refund': 'refund return money back',
        'ship': 'shipping delivery send',
        'acct': 'account',
        'pwd': 'password',
    }
    
    expanded_words = []
    for word in filtered:
        expanded_words.append(word)
        if word in expansions:
            expanded_words.extend(expansions[word].split())
    
    optimized = ' '.join(expanded_words)
    
    return optimized
```

**[8:30] [SCREEN: Walk through each template with commentary]**

Notice how we progressively cut tokens while maintaining core functionality. The CONCISE_RAG template saves 50% tokens but might lose nuance. The JSON_RAG template is perfect for programmatic use but terrible for user-facing responses.

---

## PROMPT TESTING FRAMEWORK (9:00-12:30)

**[9:00] [SLIDE: "How to Test Prompt Variants"]**

You can't optimize what you don't measure. Let's build a framework to compare prompt variants scientifically.

**[9:15] [CODE: "prompt_testing.py"]**

```python
from typing import List, Dict, Tuple
import time
import json
from dataclasses import dataclass, asdict
import statistics

@dataclass
class PromptTestResult:
    """Results from testing a prompt variant."""
    template_name: str
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    avg_latency_ms: float
    avg_cost_per_query: float
    accuracy_score: float  # Manual evaluation or automated
    queries_tested: int

class PromptTester:
    """
    Test and compare different prompt templates.
    """
    
    def __init__(
        self,
        openai_client,
        model: str = "gpt-3.5-turbo",
        cost_per_1k_input: float = 0.0015,
        cost_per_1k_output: float = 0.002
    ):
        self.client = openai_client
        self.model = model
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
    
    def test_prompt_template(
        self,
        template: PromptTemplate,
        test_cases: List[Dict[str, str]],
        context_docs: List[Dict]
    ) -> PromptTestResult:
        """
        Test a prompt template across multiple test cases.
        
        test_cases format:
        [
            {"question": "...", "expected_answer": "..."},
            ...
        ]
        """
        results = {
            'input_tokens': [],
            'output_tokens': [],
            'latency_ms': [],
            'costs': [],
            'responses': []
        }
        
        # Format context once (same for all queries)
        context = format_context_optimally(context_docs)
        
        for test_case in test_cases:
            question = test_case['question']
            
            # Format prompt
            user_message = template.user_template.format(
                context=context,
                question=question
            )
            
            # Make API call with timing
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": template.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Extract metrics
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost
            cost = (
                (input_tokens / 1000 * self.cost_per_1k_input) +
                (output_tokens / 1000 * self.cost_per_1k_output)
            )
            
            # Store results
            results['input_tokens'].append(input_tokens)
            results['output_tokens'].append(output_tokens)
            results['latency_ms'].append(latency)
            results['costs'].append(cost)
            results['responses'].append(response.choices[0].message.content)
        
        # Calculate averages
        return PromptTestResult(
            template_name=template.use_case,
            avg_input_tokens=statistics.mean(results['input_tokens']),
            avg_output_tokens=statistics.mean(results['output_tokens']),
            avg_total_tokens=statistics.mean(results['input_tokens']) + 
                           statistics.mean(results['output_tokens']),
            avg_latency_ms=statistics.mean(results['latency_ms']),
            avg_cost_per_query=statistics.mean(results['costs']),
            accuracy_score=0.0,  # Evaluate manually or with another LLM
            queries_tested=len(test_cases)
        )
    
    def compare_templates(
        self,
        templates: List[PromptTemplate],
        test_cases: List[Dict],
        context_docs: List[Dict]
    ) -> None:
        """
        Compare multiple templates and print results.
        """
        print("\n" + "="*80)
        print("PROMPT TEMPLATE COMPARISON")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Test cases: {len(test_cases)}")
        print()
        
        results = []
        
        for template in templates:
            print(f"Testing: {template.use_case}...")
            result = self.test_prompt_template(template, test_cases, context_docs)
            results.append(result)
        
        # Sort by cost
        results.sort(key=lambda x: x.avg_cost_per_query)
        
        # Print comparison table
        print("\n" + "-"*80)
        print(f"{'Template':<25} {'Tokens':<12} {'Cost':<12} {'Latency':<12}")
        print("-"*80)
        
        baseline_cost = results[-1].avg_cost_per_query  # Most expensive
        
        for result in results:
            savings = ((baseline_cost - result.avg_cost_per_query) / baseline_cost * 100)
            
            print(f"{result.template_name:<25} "
                  f"{int(result.avg_total_tokens):<12} "
                  f"${result.avg_cost_per_query:.6f}  "
                  f"{int(result.avg_latency_ms)}ms")
            
            if savings > 0:
                print(f"{'':25} → Saves {savings:.1f}% vs baseline")
        
        print("-"*80)
        
        # Calculate monthly projections
        print("\nMonthly Cost Projection (10,000 queries/day):")
        for result in results:
            monthly = result.avg_cost_per_query * 10000 * 30
            print(f"  {result.template_name}: ${monthly:.2f}/month")

# Example usage
def run_prompt_comparison():
    """
    Compare all prompt templates with real test cases.
    """
    from openai import OpenAI
    
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    
    # Sample test cases
    test_cases = [
        {
            "question": "What is your return policy?",
            "expected": "30-day returns with receipt"
        },
        {
            "question": "How long does shipping take?",
            "expected": "3-5 business days standard"
        },
        {
            "question": "Do you offer international shipping?",
            "expected": "Yes, to most countries"
        },
    ]
    
    # Sample context documents
    context_docs = [
        {
            "content": "Our return policy allows returns within 30 days of purchase with original receipt. Exceptions include personalized items and final sale products.",
            "score": 0.95
        },
        {
            "content": "Standard shipping takes 3-5 business days. Express shipping available for 1-2 day delivery at additional cost.",
            "score": 0.92
        },
    ]
    
    # Test all templates
    templates = [
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
        RAGPromptLibrary.STRUCTURED_RAG,
        RAGPromptLibrary.JSON_RAG,
    ]
    
    tester = PromptTester(client)
    tester.compare_templates(templates, test_cases, context_docs)
```

**[12:00] [TERMINAL: Run the comparison]**
```bash
python prompt_testing.py
```

**[12:15] [SCREEN: Show output comparing all templates]**

[Show comparison table with real numbers]

---

<!-- ============================================================ -->
<!-- INSERTION #3: ALTERNATIVE SOLUTIONS -->
<!-- Added per audit - explains why prompt optimization vs other approaches -->
<!-- ============================================================ -->

## ALTERNATIVE SOLUTIONS: Other Ways to Reduce RAG Costs (12:30-15:00)

**[12:30] [SLIDE: Alternative Approaches to Cost Reduction]**

Before we go further with prompt optimization, you need to know there are other ways to solve the cost problem. Let's compare approaches so you can make an informed choice.

**[13:00] [SLIDE: Four Approaches to RAG Cost Reduction]**

**Option 1: Prompt Optimization (What we're teaching today)**
- **Best for:** High query volume (1K+ queries/day), straightforward Q&A tasks
- **Key trade-off:** Saves 30-50% cost but requires ongoing testing and monitoring
- **Cost:** 4-8 hours implementation, 2-4 hours/month maintenance
- **Example use case:** Customer support chatbot with FAQ-style questions

**Option 2: Model Fine-Tuning**
- **Best for:** Domain-specific tasks with consistent patterns, high volume (10K+ queries/day)
- **Key trade-off:** Higher upfront cost ($500-2000) but potentially 80% cost reduction at scale
- **Cost:** 2-4 weeks implementation, $500-2000 training cost, ongoing retraining
- **Example use case:** Legal document analysis with specialized terminology

**Option 3: Infrastructure Optimization (Caching-First)**
- **Best for:** Repetitive queries, moderate diversity (30-40% cache hit rate possible)
- **Key trade-off:** Great ROI for repetitive queries, zero benefit for unique queries
- **Cost:** 8-12 hours implementation (M2.1), ~$50/month Redis
- **Example use case:** Product recommendations with common queries

**Option 4: Hybrid Approach (Recommended for most)**
- **Best for:** Production systems with mixed query types and growing scale
- **Key trade-off:** More complex system, but compounds savings (50-70% total reduction)
- **Cost:** Combines all above - plan 20-30 hours total implementation
- **Example use case:** Multi-tenant SaaS with diverse use cases

**[14:00] [DIAGRAM: Decision Framework]**

```
START
  ↓
Query Volume < 100/day? → Use simple approach, optimization not worth it
  ↓ No
Query Diversity < 30%? → Infrastructure optimization (caching)
  ↓ No
Domain-specific + High volume? → Consider fine-tuning
  ↓ No
Moderate volume + Cost-sensitive? → Prompt optimization ← WE ARE HERE
  ↓
Production scale (10K+/day)? → Hybrid approach
```

**[14:30] For this video, we're using prompt optimization because:**

1. **Quick wins** - You can implement and see results today
2. **No model retraining** - Works with existing infrastructure
3. **Complements caching** - Stacks with M2.1 for compound savings
4. **Low risk** - Easy to rollback if quality degrades

But remember: **If your query volume is <100/day, the optimization overhead exceeds the savings. Keep it simple.**

**[PAUSE]**

Now that you know your alternatives, let's continue with prompt optimization, understanding it's one tool in your cost-reduction toolkit.

---

## MODEL SELECTION STRATEGY (15:00-18:00)

**[15:00] [SLIDE: "Choosing the Right Model for Each Task"]**

Not every task needs GPT-4. Here's when to use what:

**[15:15] [SLIDE: "Model Selection Matrix"]**
```
Task                          | Model           | Why
------------------------------|-----------------|------------------------
Embedding Generation          | text-embed-3    | Specialized, cheap
Document Classification       | GPT-3.5-turbo   | Fast, accurate enough
Simple Q&A (FAQ)              | GPT-3.5-turbo   | Cost-effective
Complex Reasoning             | GPT-4           | Worth the premium
Code Generation               | GPT-4           | Higher accuracy matters
Creative Writing              | GPT-4           | Quality premium
Summarization                 | GPT-3.5-turbo   | Great quality/cost
Entity Extraction             | GPT-3.5-turbo   | Structured output works
Multi-document Synthesis      | GPT-4           | Better reasoning
```

**[15:30] [CODE: "model_router.py"]**

```python
from enum import Enum
from typing import Optional, Dict
import re

class ModelTier(Enum):
    """Model complexity tiers."""
    EMBEDDING = "text-embedding-3-small"
    FAST = "gpt-3.5-turbo"
    BALANCED = "gpt-4o-mini"
    PREMIUM = "gpt-4o"

class IntelligentModelRouter:
    """
    Route queries to appropriate models based on complexity.
    """
    
    def __init__(self):
        # Cost per 1M tokens (input/output)
        self.model_costs = {
            ModelTier.FAST: (0.5, 1.5),  # GPT-3.5-turbo
            ModelTier.BALANCED: (0.15, 0.6),  # GPT-4o-mini
            ModelTier.PREMIUM: (5.0, 15.0),  # GPT-4o
        }
    
    def analyze_query_complexity(self, query: str, context: str) -> Dict:
        """
        Analyze query to determine required model tier.
        """
        complexity_score = 0
        factors = {}
        
        # Factor 1: Query length (longer = more complex)
        query_words = len(query.split())
        if query_words > 20:
            complexity_score += 2
            factors['long_query'] = True
        
        # Factor 2: Multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            complexity_score += 2
            factors['multi_question'] = True
        
        # Factor 3: Comparison or analysis keywords
        reasoning_keywords = [
            'compare', 'analyze', 'evaluate', 'why', 'how does',
            'explain', 'relationship', 'impact', 'difference'
        ]
        if any(keyword in query.lower() for keyword in reasoning_keywords):
            complexity_score += 3
            factors['requires_reasoning'] = True
        
        # Factor 4: Context volume (more context = need better model)
        context_words = len(context.split())
        if context_words > 1000:
            complexity_score += 2
            factors['large_context'] = True
        
        # Factor 5: Code or technical content
        if re.search(r'```|`|\bcode\b|\bfunction\b', query + context):
            complexity_score += 2
            factors['technical_content'] = True
        
        # Factor 6: Creative or subjective request
        creative_keywords = ['creative', 'write', 'compose', 'draft', 'story']
        if any(keyword in query.lower() for keyword in creative_keywords):
            complexity_score += 3
            factors['creative_task'] = True
        
        return {
            'score': complexity_score,
            'factors': factors
        }
    
    def select_model(
        self,
        query: str,
        context: str,
        force_tier: Optional[ModelTier] = None
    ) -> Dict[str, any]:
        """
        Select optimal model based on query complexity.
        """
        if force_tier:
            return {
                'model': force_tier.value,
                'tier': force_tier.name,
                'reason': 'forced_selection'
            }
        
        analysis = self.analyze_query_complexity(query, context)
        score = analysis['score']
        
        # Route based on complexity score
        if score >= 6:
            selected_tier = ModelTier.PREMIUM
            reason = "High complexity - requires advanced reasoning"
        elif score >= 3:
            selected_tier = ModelTier.BALANCED
            reason = "Medium complexity - balanced model appropriate"
        else:
            selected_tier = ModelTier.FAST
            reason = "Simple query - fast model sufficient"
        
        # Calculate cost estimate
        input_cost, output_cost = self.model_costs[selected_tier]
        est_input_tokens = len(query + context) // 4
        est_output_tokens = 200  # Average
        
        estimated_cost = (
            (est_input_tokens / 1_000_000 * input_cost) +
            (est_output_tokens / 1_000_000 * output_cost)
        )
        
        return {
            'model': selected_tier.value,
            'tier': selected_tier.name,
            'complexity_score': score,
            'complexity_factors': analysis['factors'],
            'reason': reason,
            'estimated_cost': estimated_cost
        }
    
    def route_with_fallback(
        self,
        query: str,
        context: str,
        initial_response: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Implement fallback strategy: try fast model first,
        escalate if needed.
        """
        # Start with fast model
        initial_selection = self.select_model(query, context)
        
        # If initial response was poor quality, escalate
        # (This would need actual quality evaluation in production)
        if initial_response and self._is_low_quality(initial_response):
            return {
                'model': ModelTier.PREMIUM.value,
                'tier': ModelTier.PREMIUM.name,
                'reason': 'Escalated due to initial poor quality',
                'fallback': True
            }
        
        return initial_selection
    
    def _is_low_quality(self, response: str) -> bool:
        """
        Heuristics to detect low-quality responses.
        In production, use more sophisticated methods.
        """
        # Too short
        if len(response.split()) < 10:
            return True
        
        # Contains uncertainty phrases
        uncertain_phrases = [
            "I don't have enough information",
            "I cannot answer",
            "I'm not sure",
            "unclear from the context"
        ]
        if any(phrase in response for phrase in uncertain_phrases):
            return True
        
        return False

# Example usage
def demonstrate_model_routing():
    """
    Show model router in action.
    """
    router = IntelligentModelRouter()
    
    test_queries = [
        {
            "query": "What's the return policy?",
            "context": "Returns accepted within 30 days with receipt."
        },
        {
            "query": "Compare the performance characteristics of our Q3 vs Q4 results and analyze the key factors driving the differences. How do these trends position us against industry benchmarks?",
            "context": "Q3 revenue: $5M, Q4 revenue: $7M... [extensive data]"
        },
        {
            "query": "Write a creative product description for our new AI-powered widget that captures the imagination while explaining technical benefits.",
            "context": "Technical specs: ML-powered, 99.9% uptime, API-first..."
        }
    ]
    
    print("\n" + "="*80)
    print("INTELLIGENT MODEL ROUTING")
    print("="*80)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {test['query'][:60]}...")
        
        decision = router.select_model(test['query'], test['context'])
        
        print(f"  Selected Model: {decision['model']} ({decision['tier']})")
        print(f"  Complexity Score: {decision['complexity_score']}")
        print(f"  Reason: {decision['reason']}")
        print(f"  Estimated Cost: ${decision['estimated_cost']:.6f}")
        
        if decision['complexity_factors']:
            print(f"  Factors: {', '.join(decision['complexity_factors'].keys())}")
```

**[17:30] [SCREEN: Run demonstration showing routing decisions]**

---

## TOKEN OPTIMIZATION TECHNIQUES (18:00-21:30)

**[18:00] [SLIDE: "Advanced Token Reduction"]**

Now let's get tactical about reducing tokens without sacrificing quality.

**[18:15] [CODE: "token_optimization.py"]**

```python
import tiktoken
from typing import List, Dict

class TokenOptimizer:
    """
    Advanced techniques for minimizing token usage.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.encoder = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Accurately count tokens for pricing."""
        return len(self.encoder.encode(text))
    
    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int,
        preserve_end: bool = False
    ) -> str:
        """
        Truncate text to fit within token limit.
        """
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        if preserve_end:
            # Keep the end (useful for context that builds up)
            truncated_tokens = tokens[-max_tokens:]
        else:
            # Keep the beginning
            truncated_tokens = tokens[:max_tokens]
        
        return self.encoder.decode(truncated_tokens)
    
    def intelligent_document_truncation(
        self,
        documents: List[Dict],
        max_total_tokens: int
    ) -> List[Dict]:
        """
        Truncate multiple documents proportionally based on relevance.
        """
        # Calculate tokens for each document
        doc_tokens = []
        for doc in documents:
            tokens = self.count_tokens(doc['content'])
            doc_tokens.append({
                'doc': doc,
                'tokens': tokens,
                'score': doc.get('score', 0.5)
            })
        
        total_tokens = sum(dt['tokens'] for dt in doc_tokens)
        
        if total_tokens <= max_total_tokens:
            return documents  # No truncation needed
        
        # Allocate tokens proportionally to relevance score
        total_score = sum(dt['score'] for dt in doc_tokens)
        
        result_docs = []
        for dt in doc_tokens:
            # Allocate tokens based on score
            allocated_tokens = int(
                (dt['score'] / total_score) * max_total_tokens
            )
            
            # Ensure minimum allocation
            allocated_tokens = max(allocated_tokens, 50)
            
            # Truncate if necessary
            if dt['tokens'] > allocated_tokens:
                content = self.truncate_to_token_limit(
                    dt['doc']['content'],
                    allocated_tokens
                )
            else:
                content = dt['doc']['content']
            
            result_docs.append({
                **dt['doc'],
                'content': content,
                'original_tokens': dt['tokens'],
                'allocated_tokens': allocated_tokens
            })
        
        return result_docs
    
    def compress_repetitive_content(self, text: str) -> str:
        """
        Remove redundant information from text.
        """
        # Split into sentences
        sentences = text.split('. ')
        
        # Remove near-duplicates (simple version)
        unique_sentences = []
        seen_starts = set()
        
        for sentence in sentences:
            # Use first 30 characters as fingerprint
            fingerprint = sentence[:30].lower().strip()
            
            if fingerprint not in seen_starts:
                unique_sentences.append(sentence)
                seen_starts.add(fingerprint)
        
        return '. '.join(unique_sentences)
    
    def optimize_prompt_structure(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> tuple[str, str]:
        """
        Optimize prompt structure for token efficiency.
        """
        # Move repeated instructions to system prompt
        # Shorten common phrases
        replacements = {
            'Based on the context provided above': 'Per context',
            'Please provide a detailed explanation': 'Explain',
            'If you do not know the answer': 'If unsure',
            'to the best of your ability': '',
            'please ': '',
        }
        
        optimized_system = system_prompt
        optimized_user = user_prompt
        
        for old, new in replacements.items():
            optimized_system = optimized_system.replace(old, new)
            optimized_user = optimized_user.replace(old, new)
        
        return optimized_system.strip(), optimized_user.strip()

# Example usage
def demonstrate_token_optimization():
    """
    Show token optimization techniques with real measurements.
    """
    optimizer = TokenOptimizer()
    
    # Example 1: Document truncation
    documents = [
        {
            "content": "Our return policy allows customers to return items within 30 days... " * 50,
            "score": 0.95
        },
        {
            "content": "Shipping information: Standard shipping takes 3-5 business days... " * 30,
            "score": 0.80
        },
    ]
    
    print("\n" + "="*80)
    print("TOKEN OPTIMIZATION DEMONSTRATION")
    print("="*80)
    
    # Calculate original tokens
    original_tokens = sum(
        optimizer.count_tokens(doc['content']) for doc in documents
    )
    
    print(f"\nOriginal total tokens: {original_tokens}")
    
    # Truncate intelligently
    optimized_docs = optimizer.intelligent_document_truncation(
        documents,
        max_total_tokens=500
    )
    
    optimized_tokens = sum(
        optimizer.count_tokens(doc['content']) for doc in optimized_docs
    )
    
    print(f"Optimized total tokens: {optimized_tokens}")
    print(f"Reduction: {((original_tokens - optimized_tokens) / original_tokens * 100):.1f}%")
    
    # Example 2: Prompt optimization
    original_system = """You are a helpful AI assistant. Based on the context provided above, please provide a detailed explanation to the best of your ability."""
    
    original_user = """Please provide a detailed explanation based on the context above."""
    
    opt_system, opt_user = optimizer.optimize_prompt_structure(
        original_system,
        original_user
    )
    
    print(f"\nOriginal system prompt tokens: {optimizer.count_tokens(original_system)}")
    print(f"Optimized system prompt tokens: {optimizer.count_tokens(opt_system)}")
    print(f"\nOptimized system: {opt_system}")
```

**[21:00] [SCREEN: Run demonstration showing token savings]**

---

## PRODUCTION IMPLEMENTATION (21:30-25:00)

**[21:30] [SLIDE: "Putting It All Together"]**

Let's integrate everything into a production-ready system.

**[21:45] [CODE: "optimized_rag_production.py"]**

```python
class OptimizedProductionRAG:
    """
    Production RAG system with all optimizations applied.
    """
    
    def __init__(
        self,
        openai_client,
        cache: MultiLayerCache,
        vector_db
    ):
        self.client = openai_client
        self.cache = cache
        self.vector_db = vector_db
        
        self.model_router = IntelligentModelRouter()
        self.token_optimizer = TokenOptimizer()
        self.prompt_library = RAGPromptLibrary()
        
        # Metrics
        self.metrics = {
            'queries_processed': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0,
            'model_usage': {}
        }
    
    def query(
        self,
        user_query: str,
        use_cache: bool = True,
        optimization_level: str = 'balanced'
    ) -> Dict[str, any]:
        """
        Process query with full optimization pipeline.
        """
        start_time = time.time()
        
        # Step 1: Check cache
        if use_cache:
            cached_response = self.cache.get_cached_response(user_query)
            if cached_response:
                return {
                    'response': cached_response,
                    'source': 'cache',
                    'cost': 0.0,
                    'time_ms': (time.time() - start_time) * 1000
                }
        
        # Step 2: Optimize query for retrieval
        optimized_query = optimize_query_for_retrieval(user_query)
        
        # Step 3: Retrieve documents
        query_embedding = self._embed_with_cache(optimized_query)
        documents = self.vector_db.search(query_embedding, top_k=5)
        
        # Step 4: Optimize documents (truncate intelligently)
        max_context_tokens = 2000 if optimization_level == 'aggressive' else 3000
        optimized_docs = self.token_optimizer.intelligent_document_truncation(
            documents,
            max_total_tokens=max_context_tokens
        )
        
        # Step 5: Format context
        context = format_context_optimally(optimized_docs)
        
        # Step 6: Select appropriate model and template
        routing_decision = self.model_router.select_model(user_query, context)
        
        # Choose template based on optimization level
        if optimization_level == 'aggressive':
            template = self.prompt_library.JSON_RAG
        elif optimization_level == 'balanced':
            template = self.prompt_library.CONCISE_RAG
        else:
            template = self.prompt_library.STRUCTURED_RAG
        
        # Step 7: Generate response
        messages = [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": template.user_template.format(
                context=context,
                question=user_query
            )}
        ]
        
        response = self.client.chat.completions.create(
            model=routing_decision['model'],
            messages=messages,
            temperature=0.3,
            max_tokens=250 if optimization_level == 'aggressive' else 400
        )
        
        # Extract response and metrics
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        cost = routing_decision['estimated_cost']
        
        # Step 8: Cache result
        if use_cache:
            self.cache.cache_response(user_query, answer)
        
        # Update metrics
        self._update_metrics(routing_decision['model'], tokens_used, cost)
        
        return {
            'response': answer,
            'source': 'llm',
            'model': routing_decision['model'],
            'tokens_used': tokens_used,
            'cost': cost,
            'time_ms': (time.time() - start_time) * 1000,
            'optimization_applied': {
                'query_optimization': True,
                'document_truncation': True,
                'model_routing': True,
                'template': template.use_case
            }
        }
    
    def _embed_with_cache(self, text: str):
        """Helper to embed with caching."""
        cached = self.cache.get_cached_embedding(text)
        if cached is not None:
            return cached
        
        # Generate new embedding (implement your embedding logic)
        embedding = self._generate_embedding(text)
        self.cache.cache_embedding(text, embedding)
        return embedding
    
    def _update_metrics(self, model: str, tokens: int, cost: float):
        """Update usage metrics."""
        self.metrics['queries_processed'] += 1
        self.metrics['total_tokens_used'] += tokens
        self.metrics['total_cost'] += cost
        
        if model not in self.metrics['model_usage']:
            self.metrics['model_usage'][model] = 0
        self.metrics['model_usage'][model] += 1
    
    def get_optimization_report(self) -> Dict:
        """Generate optimization performance report."""
        if self.metrics['queries_processed'] == 0:
            return {'message': 'No queries processed yet'}
        
        avg_tokens = self.metrics['total_tokens_used'] / self.metrics['queries_processed']
        avg_cost = self.metrics['total_cost'] / self.metrics['queries_processed']
        
        # Estimate savings vs unoptimized baseline
        baseline_tokens = 550  # Average unoptimized
        baseline_cost = 0.00165  # Per query
        
        token_savings = ((baseline_tokens - avg_tokens) / baseline_tokens * 100)
        cost_savings = ((baseline_cost - avg_cost) / baseline_cost * 100)
        
        return {
            'queries_processed': self.metrics['queries_processed'],
            'avg_tokens_per_query': f"{avg_tokens:.1f}",
            'avg_cost_per_query': f"${avg_cost:.6f}",
            'token_savings_vs_baseline': f"{token_savings:.1f}%",
            'cost_savings_vs_baseline': f"{cost_savings:.1f}%",
            'total_cost': f"${self.metrics['total_cost']:.4f}",
            'model_usage': self.metrics['model_usage']
        }
```

**[24:30] [DEMO: Show the complete system working]**

---

<!-- ============================================================ -->
<!-- INSERTION #4: WHEN THIS BREAKS -->
<!-- Added per audit - Critical debugging section with 5 failures -->
<!-- ============================================================ -->

## WHEN THIS BREAKS: Common Failures & Fixes (25:00-30:00)

**[25:00] [SLIDE: When Optimization Goes Wrong]**

Now for the most important part of this video: what happens when prompt optimization breaks your system. Let me show you the 5 most common errors and exactly how to debug them.

**[PAUSE]**

These aren't hypothetical - I've hit every single one of these in production systems.

---

### Failure #1: Token Limit Exceeded Despite Optimization (25:00-26:00)

**[25:15] [TERMINAL] Let me reproduce this error:**

```bash
python test_optimized_rag.py --query "Explain our entire product catalog" --optimization aggressive
```

**Error message you'll see:**
```
openai.BadRequestError: Error code: 400
{
  "error": {
    "message": "This model's maximum context length is 4096 tokens. However, you requested 5243 tokens (4843 in the messages, 400 in the completion).",
    "type": "invalid_request_error"
  }
}
```

**What this means:**
Your aggressive document truncation set max_context_tokens=2000, but after formatting with prompt template overhead, you exceeded the model's context window. The model router selected gpt-3.5-turbo (4K context), but you're sending 5K+ tokens.

**[25:40] How to fix it:**

**[SCREEN] [CODE: token_optimization.py]**
```python
def intelligent_document_truncation(
    self,
    documents: List[Dict],
    max_total_tokens: int,
    model_context_window: int = 4096  # Add this parameter
) -> List[Dict]:
    """Truncate with safety margin for prompt overhead."""
    
    # Reserve tokens for prompt template overhead
-   actual_limit = max_total_tokens
+   prompt_overhead = 350  # Typical system + user prompt tokens
+   safety_margin = 200    # Buffer for edge cases
+   actual_limit = min(
+       max_total_tokens,
+       model_context_window - prompt_overhead - safety_margin
+   )
    
    # Rest of truncation logic...
```

**How to verify:**
```bash
python test_optimized_rag.py --query "Explain our entire product catalog" --optimization aggressive
# Should now succeed with truncated context
```

**How to prevent:**
Always calculate: `actual_tokens = context_tokens + prompt_tokens + max_completion_tokens`
Ensure: `actual_tokens < model_context_window`

---

### Failure #2: Model Router Selects Wrong Tier (26:00-27:00)

**[26:00] [TERMINAL] Reproduce the issue:**

```python
# This simple query gets routed to premium model
query = "What is your refund policy for items purchased on sale?"
# Router scores it as complexity=7 due to length, routes to GPT-4
# Cost: $0.002 when should be $0.0003 (10x more expensive!)
```

**Error message you'll see:**
No error - but your costs are 5-10x higher than expected.

**What this means:**
The complexity scoring over-weights query length. Long but simple queries trigger expensive model routing. This is a silent cost killer.

**[26:30] How to fix it:**

**[SCREEN] [CODE: model_router.py]**
```python
def analyze_query_complexity(self, query: str, context: str) -> Dict:
    """Analyze query with better heuristics."""
    complexity_score = 0
    factors = {}
    
    # IMPROVED Factor 1: Query length with context
    query_words = len(query.split())
-   if query_words > 20:
-       complexity_score += 2
+   # Only penalize if BOTH long AND has reasoning keywords
+   if query_words > 20:
+       reasoning_keywords = ['compare', 'analyze', 'evaluate', 'why']
+       has_reasoning = any(kw in query.lower() for kw in reasoning_keywords)
+       if has_reasoning:
+           complexity_score += 3
+       else:
+           complexity_score += 1  # Long but simple
    
    # Add manual override for known simple patterns
+   simple_patterns = ['what is', 'how do i', 'where can i', 'when does']
+   if any(pattern in query.lower() for pattern in simple_patterns):
+       complexity_score = max(0, complexity_score - 2)  # Reduce score
    
    return {'score': complexity_score, 'factors': factors}
```

**How to verify:**
```python
router = IntelligentModelRouter()
decision = router.select_model(
    "What is your refund policy for items purchased on sale?",
    context
)
print(decision['model'])  # Should now be gpt-3.5-turbo, not gpt-4
```

**How to prevent:**
Monitor your model distribution in metrics. If >30% queries go to premium model, your routing logic needs tuning.

---

### Failure #3: Aggressive Truncation Loses Critical Context (27:00-28:00)

**[27:00] [TERMINAL] This is subtle:**

```python
# User asks about return policy exceptions
query = "Can I return personalized items?"
# Retrieved docs include exception list, but truncation cuts it
# Response: "Yes, 30-day returns" (WRONG - should say NO)
```

**Error you'll see:**
Users report incorrect answers. Your system gives confident but wrong responses.

**What this means:**
Token truncation is happening mid-document, cutting off critical information like exceptions, caveats, or conditions. The model doesn't know context is incomplete.

**[27:30] How to fix it:**

**[SCREEN] [CODE: token_optimization.py]**
```python
def intelligent_document_truncation(
    self,
    documents: List[Dict],
    max_total_tokens: int
) -> List[Dict]:
    """Truncate at sentence boundaries, not mid-text."""
    
    result_docs = []
    for dt in doc_tokens:
        if dt['tokens'] > allocated_tokens:
-           content = self.truncate_to_token_limit(
-               dt['doc']['content'],
-               allocated_tokens
-           )
+           # Truncate at sentence boundary
+           sentences = dt['doc']['content'].split('. ')
+           truncated_sentences = []
+           current_tokens = 0
+           
+           for sentence in sentences:
+               sentence_tokens = self.count_tokens(sentence)
+               if current_tokens + sentence_tokens <= allocated_tokens:
+                   truncated_sentences.append(sentence)
+                   current_tokens += sentence_tokens
+               else:
+                   break
+           
+           content = '. '.join(truncated_sentences)
+           
+           # Add indicator if truncated
+           if len(truncated_sentences) < len(sentences):
+               content += "... [content truncated]"
        
        result_docs.append({**dt['doc'], 'content': content})
    
    return result_docs
```

**How to verify:**
```python
# Test with queries about exceptions/edge cases
test_query = "Can I return personalized items?"
response = rag_system.query(test_query, optimization_level='aggressive')
print(response['response'])  # Should correctly say "No" or "See exceptions"
```

**How to prevent:**
- Truncate at sentence boundaries
- Add "[truncated]" indicators
- Prioritize keeping document conclusions/summaries
- Monitor accuracy on edge case queries

---

### Failure #4: Cache Invalidation Causing Cost Spikes (28:00-29:00)

**[28:00] [TERMINAL] The sneaky problem:**

```bash
# Deploy new prompt template
git push origin main
# Suddenly: costs spike 10x for the next hour
# Why? Cache keys include prompt hash - all cache misses!
```

**Error you'll see:**
Cloudwatch shows: cache_hit_rate drops from 45% to 0% after deployment. Costs spike.

**What this means:**
Your cache keys are based on query + prompt template. When you update prompt templates, every cache key changes. All queries become cache misses until cache warms up again.

**[28:30] How to fix it:**

**[SCREEN] [CODE: caching.py]**
```python
class MultiLayerCache:
    """Cache with prompt-version-agnostic keys."""
    
    def _generate_cache_key(
        self,
        query: str,
-       prompt_template: str,
+       prompt_version: str = "v1",  # Semantic versioning
        model: str
    ) -> str:
        """Generate cache key independent of exact prompt wording."""
        
        # Normalize query (remove whitespace variations)
        normalized_query = ' '.join(query.lower().split())
        
        # Use semantic prompt version, not exact template
-       key_components = [normalized_query, prompt_template, model]
+       key_components = [normalized_query, prompt_version, model]
        
        return hashlib.sha256(
            '|'.join(key_components).encode()
        ).hexdigest()
+   
+   def get_cached_response(
+       self,
+       query: str,
+       prompt_version: str = "v1"
+   ):
+       """Get cached response with fallback to previous versions."""
+       
+       # Try current version
+       key_current = self._generate_cache_key(query, prompt_version)
+       cached = self.redis.get(key_current)
+       if cached:
+           return cached
+       
+       # Fallback to previous version (if template change is minor)
+       if prompt_version == "v2":
+           key_prev = self._generate_cache_key(query, "v1")
+           cached = self.redis.get(key_prev)
+           if cached:
+               # Re-cache with new key to warm up
+               self.redis.set(key_current, cached)
+               return cached
+       
+       return None
```

**How to verify:**
```bash
# Deploy prompt template change
# Monitor cache hit rate - should stay >30% instead of dropping to 0%
curl http://localhost:8000/metrics | grep cache_hit_rate
```

**How to prevent:**
- Use semantic versioning for prompts (v1, v2) not exact hashes
- Implement gradual rollout (route 10% traffic to new template first)
- Pre-warm cache by reprocessing top 100 queries with new template

---

### Failure #5: JSON Output Format Breaking (29:00-30:00)

**[29:00] [TERMINAL] The parsing nightmare:**

```python
# Using JSON_RAG template for structured output
query = "What's the shipping cost to Canada?"
# Expected: {"answer": "...", "confidence": "high", "sources": [1]}
# Actual: "The shipping cost to Canada is $15 for standard..."
# JSON parsing fails!
```

**Error message you'll see:**
```python
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
Traceback: response_data = json.loads(response.content)
```

**What this means:**
Despite your prompt saying "Return JSON only", the model occasionally returns prose. This happens ~5-10% of the time with optimized prompts because you've removed emphatic instructions.

**[29:30] How to fix it:**

**[SCREEN] [CODE: rag_prompt_templates.py]**
```python
# IMPROVED JSON_RAG template
JSON_RAG = PromptTemplate(
-   system_prompt="""Return JSON only: {"answer": "...", "confidence": "high|medium|low", "sources": [...]}. Use context provided.""",
+   system_prompt="""You must return valid JSON only. No other text.
+   Format: {"answer": "...", "confidence": "high|medium|low", "sources": [...]}.
+   If context insufficient, return: {"answer": "Insufficient information", "confidence": "low", "sources": []}.
+   Return JSON now:""",
    
    user_template="""{context}

Q: {question}
-JSON:""",
+JSON response:""",
    
    tokens_estimate=160,
    use_case="api_integration"
)

# Add JSON validation and retry logic
def query_with_structured_output(self, user_query: str, max_retries: int = 2):
    """Query with JSON output and retry on parse failure."""
    
    for attempt in range(max_retries):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[...],
+           response_format={"type": "json_object"},  # OpenAI JSON mode
            temperature=0.0  # Lower temperature for consistency
        )
        
        try:
            parsed = json.loads(response.choices[0].message.content)
+           
+           # Validate schema
+           required_keys = ['answer', 'confidence', 'sources']
+           if all(key in parsed for key in required_keys):
+               return parsed
+           else:
+               raise ValueError("Missing required keys in JSON response")
+               
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
+               # Retry with more explicit prompt
                continue
            else:
+               # Fallback to non-JSON template
                return self._fallback_to_text_response(user_query)
```

**How to verify:**
```python
# Test 100 queries to ensure consistent JSON output
for i in range(100):
    response = rag_system.query_with_structured_output(test_queries[i])
    assert isinstance(response, dict)
    assert 'answer' in response
```

**How to prevent:**
- Use OpenAI's `response_format={"type": "json_object"}` parameter (GPT-4/3.5-turbo only)
- Lower temperature to 0.0 for structured outputs
- Validate response schema, retry on failure
- Have fallback to non-JSON template

---

**[30:00] [SLIDE: Error Prevention Checklist]**

To avoid these errors:
- [ ] Always account for prompt overhead in token calculations
- [ ] Monitor model distribution (warn if >30% premium)
- [ ] Truncate at sentence boundaries, add [truncated] indicators
- [ ] Use semantic versioning for prompt templates
- [ ] Validate structured outputs, implement retries
- [ ] Run daily synthetic tests on edge cases

---

<!-- ============================================================ -->
<!-- INSERTION #5: WHEN NOT TO USE THIS -->
<!-- Added per audit - Anti-patterns and alternatives -->
<!-- ============================================================ -->

## WHEN NOT TO USE PROMPT OPTIMIZATION (30:00-32:00)

**[30:00] [SLIDE: When to Avoid This Approach]**

Let me be crystal clear about when you should NOT use prompt optimization. These are situations where optimization will hurt you.

**[PAUSE]**

**❌ Don't use prompt optimization when:**

**1. Response Quality is Non-Negotiable**

- **Why it's wrong:** Prompt optimization trades verbosity for cost. In domains like medical advice, legal analysis, or financial recommendations, that trade-off can be dangerous.
- **Use instead:** Use the best model (GPT-4) with full, unoptimized prompts. Build quality assurance into your pipeline with multi-model validation.
- **Example:** Medical diagnosis assistant - patient safety >>>> cost savings. Use GPT-4 with comprehensive prompts, add human review checkpoints.

**[30:45] 2. Query Volume is Too Low (<100 queries/day)**

- **Why it's wrong:** The overhead of implementing, testing, and monitoring prompt optimization (8-12 hours initial + 2-4 hours/month) exceeds the cost savings. At 100 queries/day with $0.002/query, you're spending $6/month. Optimization might save $3/month - not worth 12 hours of work.
- **Use instead:** Keep prompts simple and clear. Focus optimization efforts elsewhere (infrastructure, product features).
- **Example:** Internal tool used by 5 people - total cost $4/month. Optimization ROI: negative.

**[31:15] 3. Query Diversity is Extremely High (>90% unique queries)**

- **Why it's wrong:** Prompt optimization provides compounding benefits with caching. If every query is unique, caching is useless. Plus, diverse queries are harder to optimize uniformly - you need different templates for different query types, adding complexity.
- **Use instead:** Focus on infrastructure optimization (faster vector search, better retrieval) or consider fine-tuning for your specific domain.
- **Example:** Research assistant handling novel academic queries - each query is unique. Cache hit rate: 5%. Optimization overhead exceeds savings.

**[31:45] [SLIDE: Red Flags You've Chosen Wrong Approach]**

**🚩 Warning signs to watch for:**

- **Users report "answers feel rushed or incomplete"** → Optimization too aggressive, degrade prompts
- **Cache hit rate <10%** → Query diversity too high, optimization not effective
- **Costs still >$500/month after optimization** → Consider fine-tuning or model downgrade
- **Quality metrics declining** → Token cuts removing necessary context
- **More time spent tuning than saved in costs** → Volume too low for optimization

**[PAUSE]**

If you see these red flags, stop optimizing and reconsider your architecture. Sometimes the "slower, more expensive" approach is the right one.

---

<!-- ============================================================ -->
<!-- INSERTION #6: DECISION CARD -->
<!-- Added per audit - Synthesis of all honest teaching -->
<!-- ============================================================ -->

## DECISION CARD: Prompt Optimization for RAG Systems (32:00-33:00)

**[32:00] [SLIDE: Decision Card - Prompt Optimization]**

Let me synthesize everything into one decision framework. Take a screenshot of this - you'll reference it when making architectural decisions.

**[PAUSE]**

### **✅ BENEFIT**
Reduces LLM costs 30-50% with zero infrastructure changes; improves response latency 10-20% through token reduction; enables rapid A/B testing of different prompt approaches without retraining models; compounds with caching for 50-70% total cost reduction.

### **❌ LIMITATION**
Cannot improve poor retrieval quality - garbage documents produce garbage answers regardless of prompt; aggressive optimization degrades response quality by removing nuance and context; adds 50-100ms optimization overhead (query analysis, token counting, template selection); requires continuous monitoring and tuning - prompt performance degrades as query patterns shift; optimization strategies are model-specific - migrating to new models requires retesting all templates.

### **💰 COST**
**Initial:** 4-8 hours implementation (prompt library, testing framework, routing logic). **Ongoing:** 2-4 hours/month monitoring quality metrics and tuning thresholds. **Complexity:** Adds prompt versioning system, A/B testing infrastructure, quality monitoring dashboards. **Maintenance:** Monthly performance reviews, quarterly template updates, cache warming after prompt changes. **Hidden costs:** Cache invalidation during template updates causes temporary cost spikes.

### **🤔 USE WHEN**
Query volume exceeds 1,000/day (optimization overhead justified); token costs exceed $100/month (meaningful ROI on optimization effort); baseline metrics available to measure impact (can't optimize what you don't measure); acceptable to trade minor quality degradation for 30-50% cost savings; query patterns are somewhat repetitive (caching multiplies benefits); team has bandwidth for ongoing monitoring (2-4 hours/month).

### **🚫 AVOID WHEN**
Quality is non-negotiable (medical, legal, financial domains) → use best model with full prompts + human review. Query volume under 100/day → optimization overhead exceeds savings, keep it simple. Lack monitoring infrastructure → can't measure impact, dangerous to optimize blindly. Query diversity exceeds 90% → caching ineffective, uniform optimization difficult, consider fine-tuning instead. Team lacks bandwidth for maintenance → optimization degrades over time, costs may increase. Tight latency requirements (<200ms) → optimization adds 50-100ms overhead, use faster models or caching only.

**[32:45] [EMPHASIS]**

This card tells you three things:
1. **When this approach wins** (high volume, cost-sensitive, measurable)
2. **What you're giving up** (some quality, added complexity, ongoing maintenance)
3. **When to walk away** (low volume, quality-critical, no monitoring)

Make decisions based on your constraints, not what's "cool" or "cutting-edge."

---

<!-- ============================================================ -->
<!-- INSERTION #7: PRODUCTION CONSIDERATIONS (Structured) -->
<!-- Added per audit - Real deployment implications -->
<!-- ============================================================ -->

## PRODUCTION CONSIDERATIONS (33:00-35:00)

**[33:00] [SLIDE: What Changes at Scale]**

What we built today works for development and moderate production. Here's what you need to consider for serious scale.

**[33:15] Scaling concerns:**

1. **Model routing latency becomes bottleneck** - Our complexity analysis adds 50-100ms per query. At 10K+ queries/day, this compounds. Mitigation: Cache routing decisions for query patterns, pre-compute complexity for common query types.

2. **Prompt template explosion** - You'll need 10+ templates for different domains. Managing versions becomes complex. Mitigation: Build template registry with version control, automated testing pipeline for new templates.

3. **Quality monitoring at scale** - Manual review doesn't scale past 1K queries/day. Mitigation: Implement automated quality scoring using LLM-as-judge (GPT-4 evaluates GPT-3.5 outputs), track metrics by template version.

**[34:00] [SLIDE: Cost at Scale]**

Let's be specific about costs:

**Development tier (1,000 queries/day):**
- Without optimization: $450-500/month
- With optimization: $250-300/month
- **Savings: $200/month** (optimization takes 8 hours initial = $25/hour ROI)

**Production tier (10,000 queries/day):**
- Without optimization: $4,500-5,000/month
- With optimization: $2,500-3,000/month
- **Savings: $2,000/month** (ongoing maintenance 4 hours/month = $500/hour ROI)

**Enterprise tier (100,000 queries/day):**
- Without optimization: $45,000-50,000/month
- With optimization: $25,000-30,000/month
- **Savings: $20,000/month** (at this scale, hire dedicated LLM optimization engineer)

**Break-even point vs fine-tuning:** If your costs exceed $10K/month after optimization, evaluate fine-tuning. Upfront cost ($2K-5K) pays for itself in 1-2 months at that scale.

**[34:30] Monitoring requirements:**

Track these metrics daily:
- **Token usage distribution** (input vs output, by template)
- **Cost per query** (by model tier, by template)
- **Model routing accuracy** (% queries correctly routed)
- **Cache hit rate** (by query pattern)
- **Quality scores** (automated evaluation, user feedback)
- **Latency p50, p95, p99** (optimization overhead impact)

**Set alerts:**
- Cost per query >$0.003 (baseline: $0.0015)
- Premium model usage >30% (baseline: 10-15%)
- Quality score <0.85 (baseline: 0.90)
- Cache hit rate <20% (baseline: 35-45%)

**[34:45] We'll cover production monitoring dashboards in Module 3.**

That module shows you how to build Grafana dashboards for all these metrics, set up alerting, and implement automated rollback when quality degrades.

For now, understand that optimization without monitoring is flying blind.

---

## RECAP & KEY TAKEAWAYS (35:00-37:00)

**[35:00] [SLIDE: What We Covered]**

Let's recap what we learned today.

**✅ What we learned:**

1. **RAG-specific prompt engineering** - Seven templates from baseline to highly optimized (50% token reduction)
2. **Intelligent model routing** - Match query complexity to model tier, save money on simple queries
3. **Token optimization techniques** - Document truncation, prompt compression, query optimization
4. **Production implementation** - End-to-end system integrating all optimizations
5. **When NOT to use this approach** - Low volume, quality-critical applications, high query diversity
6. **Common failure modes** - Token limits, wrong model routing, context loss, cache invalidation, JSON parsing
7. **Alternative cost reduction strategies** - Fine-tuning, infrastructure optimization, hybrid approaches

**✅ What we built:**

A production-ready prompt optimization system with:
- Prompt template library (7 variants)
- Scientific testing framework (A/B comparison)
- Intelligent model router (complexity-based)
- Token optimizer (smart truncation)
- Complete metrics tracking

**✅ What we debugged:**

All 5 common failures:
1. Token limit exceeded despite optimization
2. Model router selecting wrong tier
3. Aggressive truncation losing critical context
4. Cache invalidation causing cost spikes
5. JSON output format breaking

**[36:00] ⚠️ Critical limitation to remember:**

**Prompt optimization cannot fix poor retrieval quality.** If your vector search returns irrelevant documents, no amount of prompt engineering will save you. Quality starts with retrieval.

**[36:30] [SLIDE: Connection to Next Video]**

In M2.3, we're building a production monitoring dashboard. This gives you visibility into:
- Real-time cost tracking by template/model
- Quality metrics with automated alerting
- Performance optimization opportunities
- A/B test result visualization

This builds directly on what we did today by making optimization measurable and sustainable.

See you there!

---

## CHALLENGES (37:00-38:00)

**[37:00] [SLIDE: Practice Challenges]**

Time to practice! Three challenges at different difficulty levels.

### 🟢 **EASY Challenge** (15-30 minutes)

**Task:** Create 3 different prompt templates for your domain. Test them with 10 sample queries and measure token usage. Calculate monthly cost savings at 1,000 queries/day.

**Success criteria:**
- [ ] Three PromptTemplate objects with different optimization levels
- [ ] PromptTester comparing all three templates
- [ ] Monthly cost projection showing savings
- [ ] Documentation of which template you'd choose and why

**Hint:** Start with BASIC_RAG as baseline, then create CONCISE and STRUCTURED variants.

---

### 🟡 **MEDIUM Challenge** (30-60 minutes)

**Task:** Implement an A/B testing system that randomly routes 50% of queries to two different prompt variants. Track quality metrics (you define them) and cost for each variant. Determine winner after 100 queries.

**Success criteria:**
- [ ] Random routing (50/50 split) implemented
- [ ] Metrics tracked per variant (cost, latency, custom quality score)
- [ ] Statistical comparison showing which variant wins
- [ ] Report explaining decision with data

**Hint:** Use random.choice() for routing, track results in a dictionary keyed by template name.

---

### 🔴 **HARD Challenge** (1-3 hours, portfolio-worthy)

**Task:** Build an automated prompt optimizer that uses GPT-4 to generate and test prompt variations, keeping the best performers. System should:
1. Start with base prompt
2. Use GPT-4 to generate 5 variants optimizing for different objectives
3. Test all variants with real queries
4. Use GPT-4 as judge to evaluate quality
5. Select optimal prompt based on cost-quality Pareto frontier

**Success criteria:**
- [ ] Automated variant generation using GPT-4
- [ ] Testing framework evaluating all variants
- [ ] Quality scoring using LLM-as-judge
- [ ] Multi-objective optimization (cost vs quality)
- [ ] Final report with Pareto frontier visualization

**This is portfolio-worthy!** Share your solution in Discord when complete. This demonstrates advanced LLM engineering skills.

**No hints - figure it out!** (Solutions provided in 48 hours)

---

## ACTION ITEMS (38:00-39:00)

**[38:00] [SLIDE: Before M2.3]**

Before moving to the next video, complete these tasks:

**REQUIRED:**
1. [ ] Implement PromptTester and compare at least 3 templates on your data
2. [ ] Calculate your current token usage and potential savings
3. [ ] Set up model routing with complexity scoring
4. [ ] Test all 5 common failures we covered and verify fixes
5. [ ] Identify your most expensive queries (highest token usage)

**RECOMMENDED:**
1. [ ] Read: [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
2. [ ] Experiment with different template variations for your domain
3. [ ] Share your optimization results in Discord (before/after metrics)
4. [ ] Calculate ROI of optimization for your specific use case

**OPTIONAL:**
1. [ ] Research fine-tuning cost-benefit analysis for your scale
2. [ ] Compare our approach vs other RAG optimization frameworks
3. [ ] Build quality monitoring dashboard prototype

**Estimated time investment:** 60-90 minutes for required items

---

## WRAP-UP (39:00-40:00)

**[39:00] [SLIDE: Great Work!]**

Excellent work making it through! Prompt optimization is one of those topics that seems simple but has real depth when you dig into production implications.

**Remember the key points:**

- **Prompt engineering can cut costs 30-50%** with zero infrastructure changes - fastest ROI
- **But it's not for everyone** - low volume or quality-critical apps should skip it
- **Always measure before optimizing** - you need baselines to know if changes help
- **Optimization degrades over time** - plan for ongoing monitoring and tuning

**[EMPHASIS] The most important thing from today:**

**Prompt optimization is a tool, not a mandate.** If your monthly LLM costs are $50, don't spend 10 hours optimizing. If they're $5,000, optimization is probably worth it. Let economics guide your engineering decisions.

**If you get stuck:**

1. Review the "When This Breaks" section (timestamp: 25:00)
2. Check the Decision Card (timestamp: 32:00)
3. Post in Discord #module-2 with:
   - Your query volume and costs
   - What you've tried
   - Error messages or unexpected behavior
4. Attend office hours (Thursdays 2pm EST)

**[39:45] See you in M2.3 where we build the monitoring dashboard!**

We'll visualize all these optimization metrics in Grafana, set up real-time alerting, and implement automated rollback when quality drops. It's where optimization becomes sustainable.

**[SLIDE: End Card with Course Branding]**

---

# PRODUCTION NOTES (Creator-Only)

## Changes from Original Script

**Sections Added (1,720 words, ~16 minutes):**
1. Prerequisite Check (0:00-1:00) - 100 words
2. Reality Check (3:00-5:30) - 250 words
3. Alternative Solutions (12:30-15:00) - 250 words
4. When This Breaks (25:00-30:00) - 600 words
5. When NOT to Use (30:00-32:00) - 200 words
6. Decision Card (32:00-33:00) - 120 words
7. Production Considerations (33:00-35:00) - 200 words

**Timestamps adjusted:**
All sections after Reality Check shifted forward by cumulative insertion time. Original 24-min video now 40-min video.

**Transitions added:**
Each new section has connecting sentences to/from adjacent content.

## Pre-Recording Checklist

- [ ] **All code tested:** Every example runs without errors
- [ ] **5 failures reproducible:** Can trigger each error on command
- [ ] **Decision Card slide:** All 5 fields visible, readable for 10 seconds
- [ ] **Reality Check emphasis:** Deliver honestly, not apologetically
- [ ] **Alternative Solutions diagram:** Decision flowchart ready
- [ ] **Terminal demos prepared:** All commands tested, output predictable
- [ ] **Timing practiced:** New sections add ~16 minutes total
- [ ] **Tone calibrated:** Honest but not negative about limitations

## Recording Guidelines for New Sections

**Reality Check (Critical):**
- Speak earnestly, not defensively
- Pause after "What it DOESN'T do" for emphasis
- Don't apologize for limitations - state them factually
- Energy: Serious but not dour

**When This Breaks (Showcase):**
- Actually trigger each error on screen
- Show real error messages, not hypothetical
- Walk through fixes with visible code diffs
- Energy: Debugging mode - focused, problem-solving

**When NOT to Use (Protective):**
- Speak protectively, like warning a friend
- Use concrete numbers (100 queries/day, $50/month)
- Energy: Mentorly, experiential

**Decision Card (Authoritative):**
- Speak slowly - students taking screenshots
- Read all 5 fields completely, no summarizing
- Pause 5 seconds after displaying slide
- Energy: Confident, definitive

## Editing Notes

**New sections priority for B-roll:**
- Reality Check: Show cost graphs, trade-off visualizations
- When This Breaks: Tight cuts on error reproduction
- Decision Card: Hold on screen minimum 10 seconds
- Alternative Solutions: Diagram should be highly visible

**Pacing:**
- Reality Check can be tightened 10% in post
- When This Breaks should NOT be cut - every failure is critical
- Decision Card must be readable - slow down if needed

## Gate to Publish

**Enhanced Script Checklist:**
- [ ] All 7 new sections recorded and edited in
- [ ] Timestamps updated throughout video description
- [ ] Decision Card graphic exported as downloadable PNG
- [ ] All 5 failures demonstrated on screen (not just discussed)
- [ ] Alternative Solutions flowchart included in video
- [ ] Transitions between new/old sections are smooth
- [ ] Total runtime: 38-40 minutes
- [ ] TVH v2.0 compliance: 6/6 sections present

**Quality verification:**
- [ ] Reality Check includes specific limitations (not generic)
- [ ] Alternative Solutions compares 4+ approaches
- [ ] When This Breaks shows all 5 failures with fixes
- [ ] When NOT to Use gives 3+ anti-patterns with alternatives
- [ ] Decision Card has all 5 fields with specific content
- [ ] Production Considerations has real numbers at scale

---

**Version:** M2.2 Enhanced (TVH v2.0 Compliant)
**Original:** 24 minutes
**Enhanced:** 40 minutes
**Compliance:** 6/6 mandatory sections
**Word count added:** ~1,720 words
**Recording time added:** ~16 minutes