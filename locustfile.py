"""
Load Testing for RAG System with Locust
Covers: Smoke, Load, Stress, Spike, and Soak test scenarios
"""

import os
from locust import HttpUser, task, between, events
from dotenv import load_dotenv

load_dotenv()

# Test scenario parameters
SCENARIOS = {
    "smoke": {
        "users": 10,
        "spawn_rate": 2,
        "run_time": "2m",
        "description": "Low load verification - system health check"
    },
    "load": {
        "users": 100,
        "spawn_rate": 10,
        "run_time": "10m",
        "description": "Expected normal capacity testing"
    },
    "stress": {
        "users": 1000,
        "spawn_rate": 50,
        "run_time": "15m",
        "description": "Push beyond limits to find breaking points"
    },
    "spike": {
        "users": 500,
        "spawn_rate": 500,  # Instant spike
        "run_time": "5m",
        "description": "Sudden traffic surge simulation"
    },
    "soak": {
        "users": 50,
        "spawn_rate": 5,
        "run_time": "4h",
        "description": "Extended sustained load for memory leak detection"
    }
}


class RAGUser(HttpUser):
    """
    Simulates realistic user behavior for RAG system.
    Task weights: query (10x), retrieval (3x), health check (1x)
    """

    # Realistic thinking time between requests (1-3 seconds)
    wait_time = between(1, 3)

    # Sample queries for realistic testing
    queries = [
        "What is vector similarity search?",
        "Explain RAG architecture",
        "How does embedding work?",
        "What are chunking strategies?",
        "Compare FAISS and Pinecone",
    ]

    @task(10)
    def query_endpoint(self):
        """
        Primary query task - weighted 10x more than others.
        Tests the main /query endpoint with various questions.
        """
        query = self.queries[self.environment.runner.user_count % len(self.queries)]

        with self.client.post(
            "/query",
            json={"query": query, "top_k": 5},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limit hit - expected behavior, not a failure
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(3)
    def retrieval_endpoint(self):
        """
        Document retrieval task - weighted 3x.
        Tests retrieval without full RAG pipeline.
        """
        with self.client.get(
            "/retrieval",
            params={"query": "test query", "limit": 5},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(f"Failed retrieval: {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Health check task - weighted 1x.
        Ensures system responsiveness.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


# Event listeners for test lifecycle
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start with scenario details."""
    print(f"\n{'='*60}")
    print(f"🚀 Load Test Started")
    print(f"Target: {environment.host}")
    print(f"{'='*60}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion with basic statistics."""
    print(f"\n{'='*60}")
    print(f"✅ Load Test Completed")

    stats = environment.stats
    print(f"Total Requests: {stats.total.num_requests}")
    print(f"Total Failures: {stats.total.num_failures}")
    print(f"Avg Response Time: {stats.total.avg_response_time:.2f}ms")

    if stats.total.num_requests > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure Rate: {failure_rate:.2f}%")

    print(f"{'='*60}\n")


# CLI Commands for each scenario:
#
# SMOKE TEST:
# locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 2m --headless
#
# LOAD TEST:
# locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 10m --headless
#
# STRESS TEST:
# locust -f locustfile.py --host=http://localhost:8000 --users 1000 --spawn-rate 50 --run-time 15m --headless
#
# SPIKE TEST:
# locust -f locustfile.py --host=http://localhost:8000 --users 500 --spawn-rate 500 --run-time 5m --headless
#
# SOAK TEST:
# locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 4h --headless
#
# WEB UI MODE (any scenario):
# locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 and configure parameters manually
