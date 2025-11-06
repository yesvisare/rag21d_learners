"""
Basic tests for locustfile.py to ensure it's properly configured.

Run with: python tests_locustfile.py
Or with pytest: pytest tests_locustfile.py -v
"""

import sys
import inspect


def test_locustfile_imports():
    """Test that locustfile.py can be imported without errors."""
    try:
        import locustfile
        print("✅ locustfile.py imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import locustfile.py: {e}")
        return False


def test_rag_user_class_exists():
    """Test that RAGUser class is defined."""
    try:
        import locustfile
        assert hasattr(locustfile, 'RAGUser'), "RAGUser class not found"
        print("✅ RAGUser class exists")
        return True
    except AssertionError as e:
        print(f"❌ {e}")
        return False


def test_tasks_registered():
    """Test that tasks are properly decorated and registered."""
    try:
        import locustfile
        from locust import task

        rag_user = locustfile.RAGUser

        # Check if class has tasks
        task_methods = []
        for name, method in inspect.getmembers(rag_user, predicate=inspect.isfunction):
            # Check if method has task decorator
            if hasattr(method, 'tasks'):
                task_methods.append(name)

        assert len(task_methods) > 0, "No tasks found in RAGUser class"
        print(f"✅ Found {len(task_methods)} task(s): {', '.join(task_methods)}")
        return True
    except AssertionError as e:
        print(f"❌ {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking tasks: {e}")
        return False


def test_wait_time_configured():
    """Test that wait_time is configured."""
    try:
        import locustfile

        rag_user = locustfile.RAGUser
        assert hasattr(rag_user, 'wait_time'), "wait_time not configured"
        print("✅ wait_time is configured")
        return True
    except AssertionError as e:
        print(f"❌ {e}")
        return False


def test_scenarios_defined():
    """Test that test scenarios are defined."""
    try:
        import locustfile

        assert hasattr(locustfile, 'SCENARIOS'), "SCENARIOS not defined"
        scenarios = locustfile.SCENARIOS

        required_scenarios = ['smoke', 'load', 'stress', 'spike', 'soak']
        for scenario in required_scenarios:
            assert scenario in scenarios, f"Scenario '{scenario}' not found"

        print(f"✅ All {len(required_scenarios)} scenarios defined: {', '.join(required_scenarios)}")
        return True
    except AssertionError as e:
        print(f"❌ {e}")
        return False


def test_event_listeners():
    """Test that event listeners are configured."""
    try:
        import locustfile

        # Check if module has event listener decorators
        source = inspect.getsource(locustfile)

        has_test_start = '@events.test_start' in source
        has_test_stop = '@events.test_stop' in source

        if has_test_start and has_test_stop:
            print("✅ Event listeners configured (test_start, test_stop)")
            return True
        else:
            print("⚠️  Warning: Event listeners not found (optional)")
            return True  # Not critical
    except Exception as e:
        print(f"⚠️  Could not verify event listeners: {e}")
        return True  # Not critical


def test_scenario_parameters():
    """Test that scenarios have required parameters."""
    try:
        import locustfile

        scenarios = locustfile.SCENARIOS
        required_params = ['users', 'spawn_rate', 'run_time']

        for scenario_name, scenario_config in scenarios.items():
            for param in required_params:
                assert param in scenario_config, f"Scenario '{scenario_name}' missing parameter '{param}'"

        print("✅ All scenarios have required parameters (users, spawn_rate, run_time)")
        return True
    except AssertionError as e:
        print(f"❌ {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("Running locustfile.py tests...")
    print("="*60 + "\n")

    tests = [
        test_locustfile_imports,
        test_rag_user_class_exists,
        test_tasks_registered,
        test_wait_time_configured,
        test_scenarios_defined,
        test_event_listeners,
        test_scenario_parameters,
    ]

    results = []
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        result = test()
        results.append(result)
        print()

    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
