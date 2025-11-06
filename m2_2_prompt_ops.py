"""
M2.2 Prompt Optimization & Model Selection - Core Module

Production-ready prompt optimization system with intelligent model routing
and token management for RAG systems.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import statistics
import json
import re
import tiktoken
import config


@dataclass
class PromptTemplate:
    """Structured prompt template for RAG systems."""
    system_prompt: str
    user_template: str
    tokens_estimate: int
    use_case: str


class RAGPromptLibrary:
    """Production-tested prompt templates for different RAG scenarios."""

    # BASELINE (Non-optimized)
    BASIC_RAG = PromptTemplate(
        system_prompt="""You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain enough information to answer fully, say so. Be comprehensive and detailed in your response.""",
        user_template="""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above.""",
        tokens_estimate=350,
        use_case="baseline_comparison"
    )

    # OPTIMIZED VERSION 1: Concise
    CONCISE_RAG = PromptTemplate(
        system_prompt="""Answer based solely on provided context. If uncertain, state "Insufficient information." Be concise.""",
        user_template="""Context:
{context}

Q: {question}
A:""",
        tokens_estimate=180,
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

    # OPTIMIZED VERSION 3: JSON Output
    JSON_RAG = PromptTemplate(
        system_prompt="""You must return valid JSON only. No other text.
Format: {"answer": "...", "confidence": "high|medium|low", "sources": [...]}.
If context insufficient, return: {"answer": "Insufficient information", "confidence": "low", "sources": []}.
Return JSON now:""",
        user_template="""{context}

Q: {question}
JSON response:""",
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

    @classmethod
    def get_template_by_name(cls, name: str) -> PromptTemplate:
        """Get template by use_case name."""
        templates = {
            "baseline_comparison": cls.BASIC_RAG,
            "cost_optimization": cls.CONCISE_RAG,
            "structured_responses": cls.STRUCTURED_RAG,
            "api_integration": cls.JSON_RAG,
            "customer_support": cls.SUPPORT_RAG,
        }
        return templates.get(name, cls.CONCISE_RAG)


class TokenEstimator:
    """Rough token math and cost projection helpers."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.encoder = None
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except (KeyError, Exception) as e:
            # Fallback to rough estimation if tiktoken fails
            print(f"⚠️  Tiktoken unavailable ({e.__class__.__name__}), using rough estimation")
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens for pricing."""
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """Calculate cost based on token counts."""
        model = model or self.model
        pricing = config.MODEL_PRICE_TABLE.get(model, config.MODEL_PRICE_TABLE["gpt-3.5-turbo"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def project_monthly_cost(
        self,
        avg_input_tokens: int,
        avg_output_tokens: int,
        queries_per_day: int,
        model: Optional[str] = None
    ) -> Dict[str, float]:
        """Project monthly costs based on usage patterns."""
        cost_per_query = self.estimate_cost(avg_input_tokens, avg_output_tokens, model)
        daily_cost = cost_per_query * queries_per_day
        monthly_cost = daily_cost * 30

        return {
            "cost_per_query": cost_per_query,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "queries_per_day": queries_per_day,
        }


def format_context_optimally(
    documents: List[Dict],
    max_tokens: int = 1500,
    include_metadata: bool = False,
    estimator: Optional[TokenEstimator] = None
) -> str:
    """
    Format retrieved documents to minimize token usage while maintaining quality.

    Args:
        documents: List of document dicts with 'content' and optional 'score'
        max_tokens: Maximum tokens to use for context
        include_metadata: Whether to include document scores
        estimator: TokenEstimator instance for accurate counting

    Returns:
        Formatted context string
    """
    if estimator is None:
        estimator = TokenEstimator()

    formatted_docs = []
    total_tokens = 0

    for i, doc in enumerate(documents, 1):
        content = doc['content']

        # Count tokens accurately
        doc_tokens = estimator.count_tokens(content)

        if total_tokens + doc_tokens > max_tokens:
            # Truncate to fit
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 50:  # Only add if meaningful space left
                content = _truncate_to_token_limit(content, remaining_tokens, estimator)
                content += "... [truncated]"
            else:
                break

        if include_metadata:
            formatted_doc = f"[{i}] (Score: {doc.get('score', 0):.2f})\n{content}"
        else:
            formatted_doc = f"[{i}] {content}"

        formatted_docs.append(formatted_doc)
        total_tokens += estimator.count_tokens(formatted_doc)

        if total_tokens >= max_tokens:
            break

    return "\n\n".join(formatted_docs)


def _truncate_to_token_limit(
    text: str,
    max_tokens: int,
    estimator: TokenEstimator
) -> str:
    """Truncate text at sentence boundary to stay within token limit."""
    sentences = text.split('. ')
    truncated = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimator.count_tokens(sentence + '. ')
        if current_tokens + sentence_tokens <= max_tokens:
            truncated.append(sentence)
            current_tokens += sentence_tokens
        else:
            break

    result = '. '.join(truncated)
    # Ensure we add period back if we had sentences
    if truncated and not result.endswith('.'):
        result += '.'
    return result


class ModelTier(Enum):
    """Model complexity tiers."""
    EMBEDDING = "text-embedding-3-small"
    FAST = "gpt-3.5-turbo"
    BALANCED = "gpt-4o-mini"
    PREMIUM = "gpt-4o"


class ModelRouter:
    """Route queries to appropriate models based on complexity."""

    def __init__(self):
        self.model_costs = config.MODEL_PRICE_TABLE

    def analyze_query_complexity(self, query: str, context: str = "") -> Dict:
        """
        Analyze query to determine required model tier.

        Returns dict with:
            - score: complexity score (0-10+)
            - factors: dict of contributing factors
        """
        complexity_score = 0
        factors = {}

        # Factor 1: Query length with reasoning context
        query_words = len(query.split())
        if query_words > 20:
            reasoning_keywords = ['compare', 'analyze', 'evaluate', 'why', 'how does']
            has_reasoning = any(kw in query.lower() for kw in reasoning_keywords)
            if has_reasoning:
                complexity_score += 3
                factors['long_with_reasoning'] = True
            else:
                complexity_score += 1
                factors['long_simple'] = True

        # Factor 2: Multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            complexity_score += 2
            factors['multi_question'] = True

        # Factor 3: Reasoning keywords
        reasoning_keywords = [
            'compare', 'analyze', 'evaluate', 'why', 'how does',
            'explain', 'relationship', 'impact', 'difference'
        ]
        if any(keyword in query.lower() for keyword in reasoning_keywords):
            complexity_score += 3
            factors['requires_reasoning'] = True

        # Factor 4: Context volume
        context_words = len(context.split()) if context else 0
        if context_words > 1000:
            complexity_score += 2
            factors['large_context'] = True

        # Factor 5: Code or technical content
        if re.search(r'```|`|\bcode\b|\bfunction\b', query + context):
            complexity_score += 2
            factors['technical_content'] = True

        # Reduce score for simple patterns
        simple_patterns = ['what is', 'how do i', 'where can i', 'when does']
        if any(pattern in query.lower() for pattern in simple_patterns):
            complexity_score = max(0, complexity_score - 2)
            factors['simple_pattern'] = True

        return {
            'score': complexity_score,
            'factors': factors
        }

    def select_model(
        self,
        query: str,
        context: str = "",
        force_tier: Optional[ModelTier] = None,
        cost_budget: Optional[float] = None
    ) -> Dict:
        """
        Select optimal model based on query complexity and constraints.

        Args:
            query: User query
            context: Context that will be sent
            force_tier: Override automatic selection
            cost_budget: Maximum cost per query (optional constraint)

        Returns:
            Dict with model selection details
        """
        if force_tier:
            return {
                'model': force_tier.value,
                'tier': force_tier.name,
                'reason': 'forced_selection',
                'complexity_score': 0,
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

        # Check cost budget if specified
        if cost_budget:
            estimator = TokenEstimator(selected_tier.value)
            est_tokens_in = estimator.count_tokens(query + context)
            est_tokens_out = 200
            est_cost = estimator.estimate_cost(est_tokens_in, est_tokens_out, selected_tier.value)

            if est_cost > cost_budget:
                # Downgrade to cheaper model
                selected_tier = ModelTier.FAST
                reason = f"Cost budget constraint (${cost_budget:.6f})"

        return {
            'model': selected_tier.value,
            'tier': selected_tier.name,
            'complexity_score': score,
            'complexity_factors': analysis['factors'],
            'reason': reason,
        }


@dataclass
class PromptTestResult:
    """Results from testing a prompt variant."""
    template_name: str
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    avg_latency_ms: float
    avg_cost_per_query: float
    accuracy_score: float
    queries_tested: int


class PromptTester:
    """Test and compare different prompt templates."""

    def __init__(
        self,
        openai_client=None,
        model: str = "gpt-3.5-turbo",
        dry_run: bool = False
    ):
        """
        Initialize tester.

        Args:
            openai_client: OpenAI client instance (can be None for dry run)
            model: Model to use for testing
            dry_run: If True, skip actual API calls and use estimates
        """
        self.client = openai_client
        self.model = model
        self.dry_run = dry_run or (openai_client is None)
        self.estimator = TokenEstimator(model)

        # Get pricing for this model
        pricing = config.MODEL_PRICE_TABLE.get(model, config.MODEL_PRICE_TABLE["gpt-3.5-turbo"])
        self.cost_per_1k_input = pricing["input"] / 1000  # Convert to per 1k
        self.cost_per_1k_output = pricing["output"] / 1000

    def test_prompt_template(
        self,
        template: PromptTemplate,
        test_cases: List[Dict[str, str]],
        context_docs: List[Dict]
    ) -> PromptTestResult:
        """
        Test a prompt template across multiple test cases.

        Args:
            template: PromptTemplate to test
            test_cases: List of {"question": "...", "expected_answer": "..."}
            context_docs: List of document dicts for context

        Returns:
            PromptTestResult with aggregated metrics
        """
        results = {
            'input_tokens': [],
            'output_tokens': [],
            'latency_ms': [],
            'costs': [],
            'responses': []
        }

        # Format context once
        context = format_context_optimally(context_docs, estimator=self.estimator)

        for test_case in test_cases:
            question = test_case['question']

            # Format prompt
            user_message = template.user_template.format(
                context=context,
                question=question
            )

            if self.dry_run:
                # Estimate tokens and simulate
                input_tokens = self.estimator.count_tokens(
                    template.system_prompt + user_message
                )
                output_tokens = 150  # Average estimate
                latency = 800  # Simulated latency
                response_text = f"[DRY RUN] Simulated response for: {question[:50]}..."
            else:
                # Make actual API call
                start_time = time.time()

                try:
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
                    response_text = response.choices[0].message.content

                except Exception as e:
                    print(f"⚠️ API call failed: {e}")
                    # Fall back to estimates
                    input_tokens = self.estimator.count_tokens(
                        template.system_prompt + user_message
                    )
                    output_tokens = 150
                    latency = 0
                    response_text = f"[ERROR] {str(e)}"

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
            results['responses'].append(response_text)

        # Calculate averages
        return PromptTestResult(
            template_name=template.use_case,
            avg_input_tokens=statistics.mean(results['input_tokens']),
            avg_output_tokens=statistics.mean(results['output_tokens']),
            avg_total_tokens=statistics.mean(results['input_tokens']) +
                           statistics.mean(results['output_tokens']),
            avg_latency_ms=statistics.mean(results['latency_ms']) if results['latency_ms'] else 0,
            avg_cost_per_query=statistics.mean(results['costs']),
            accuracy_score=0.0,  # Would need evaluation logic
            queries_tested=len(test_cases)
        )

    def compare_templates(
        self,
        templates: List[PromptTemplate],
        test_cases: List[Dict],
        context_docs: List[Dict],
        output_format: str = "table"
    ) -> List[PromptTestResult]:
        """
        Compare multiple templates and return results.

        Args:
            templates: List of PromptTemplate objects to compare
            test_cases: Test queries
            context_docs: Context documents
            output_format: "table" or "json"

        Returns:
            List of PromptTestResult objects, sorted by cost
        """
        print("\n" + "="*80)
        print("PROMPT TEMPLATE COMPARISON")
        print("="*80)
        print(f"Model: {self.model}")
        print(f"Test cases: {len(test_cases)}")
        print(f"Mode: {'DRY RUN (estimates only)' if self.dry_run else 'LIVE API calls'}")
        print()

        results = []

        for template in templates:
            print(f"Testing: {template.use_case}...")
            result = self.test_prompt_template(template, test_cases, context_docs)
            results.append(result)

        # Sort by cost
        results.sort(key=lambda x: x.avg_cost_per_query)

        if output_format == "table":
            self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results: List[PromptTestResult]):
        """Print formatted comparison table."""
        print("\n" + "-"*80)
        print(f"{'Template':<25} {'Tokens':<12} {'Cost':<12} {'Latency':<12}")
        print("-"*80)

        baseline_cost = results[-1].avg_cost_per_query if results else 0

        for result in results:
            if baseline_cost > 0:
                savings = ((baseline_cost - result.avg_cost_per_query) / baseline_cost * 100)
            else:
                savings = 0

            print(f"{result.template_name:<25} "
                  f"{int(result.avg_total_tokens):<12} "
                  f"${result.avg_cost_per_query:.6f}  "
                  f"{int(result.avg_latency_ms)}ms")

            if savings > 0:
                print(f"{'':25} → Saves {savings:.1f}% vs baseline")

        print("-"*80)

        # Monthly projections
        print("\nMonthly Cost Projection (10,000 queries/day):")
        for result in results:
            monthly = result.avg_cost_per_query * 10000 * 30
            print(f"  {result.template_name}: ${monthly:.2f}/month")

    def export_results(self, results: List[PromptTestResult], filepath: str):
        """Export results to JSON or CSV."""
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2)
        elif filepath.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(filepath, index=False)
        print(f"Results exported to: {filepath}")


if __name__ == "__main__":
    # Example usage (dry run mode)
    print("M2.2 Prompt Optimization Module")
    print("=" * 80)

    # Load example data
    with open("example_data.json", "r") as f:
        data = json.load(f)

    # Test token estimation
    estimator = TokenEstimator()
    sample_text = data["documents"][0]["content"]
    tokens = estimator.count_tokens(sample_text)
    print(f"\nToken counting example:")
    print(f"Text: {sample_text[:80]}...")
    print(f"Tokens: {tokens}")

    # Test model routing
    router = ModelRouter()
    for query_data in data["test_queries"][:2]:
        query = query_data["question"]
        decision = router.select_model(query, sample_text)
        print(f"\nQuery: {query}")
        print(f"Selected: {decision['model']} (score: {decision['complexity_score']})")
        print(f"Reason: {decision['reason']}")

    # Test prompt comparison (dry run)
    print("\n" + "="*80)
    print("Running template comparison (DRY RUN)...")
    tester = PromptTester(dry_run=True)
    templates = [
        RAGPromptLibrary.BASIC_RAG,
        RAGPromptLibrary.CONCISE_RAG,
        RAGPromptLibrary.STRUCTURED_RAG,
    ]
    results = tester.compare_templates(
        templates,
        data["test_queries"][:3],
        data["documents"]
    )

    print("\n✅ Module loaded successfully. Ready for notebook usage.")
