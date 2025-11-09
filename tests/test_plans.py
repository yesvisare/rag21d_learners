#!/usr/bin/env python3
"""
Smoke Tests for M4.4 Course Wrap-up Planning Tools

Tests to verify:
1. Skills checklist renders correctly
2. Knowledge gaps display properly
3. Learning paths load with decision cards
4. Templates generate successfully
5. CSV files created with correct structure
"""

import os
import sys
import csv
from io import StringIO


def test_imports():
    """Test that all required modules import correctly"""
    print("Testing imports...")
    try:
        import m4_4_planning_tools as tools
        print("✅ Successfully imported m4_4_planning_tools")
        return True
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False


def test_skills_checklist():
    """Test that skills checklist displays correctly"""
    print("\nTesting SkillsChecklist...")
    try:
        from m4_4_planning_tools import SkillsChecklist

        # Check technical skills list exists and has content
        assert len(SkillsChecklist.TECHNICAL_SKILLS) > 0, "Technical skills list is empty"
        assert len(SkillsChecklist.SOFT_SKILLS) > 0, "Soft skills list is empty"

        # Verify specific skills
        assert any("vector search" in skill.lower() for skill in SkillsChecklist.TECHNICAL_SKILLS), \
            "Vector search skill not found"
        assert any("rag" in skill.lower() for skill in SkillsChecklist.TECHNICAL_SKILLS), \
            "RAG skill not found"

        print(f"✅ Skills checklist has {len(SkillsChecklist.TECHNICAL_SKILLS)} technical skills")
        print(f"✅ Skills checklist has {len(SkillsChecklist.SOFT_SKILLS)} soft skills")
        return True
    except Exception as e:
        print(f"❌ Skills checklist test failed: {e}")
        return False


def test_knowledge_gaps():
    """Test that knowledge gaps are properly defined"""
    print("\nTesting KnowledgeGaps...")
    try:
        from m4_4_planning_tools import KnowledgeGaps

        # Check gaps dictionary exists and has content
        assert len(KnowledgeGaps.GAPS) > 0, "Knowledge gaps dictionary is empty"

        # Verify specific categories
        required_categories = [
            "Model Training & Fine-tuning",
            "MLOps & Production ML",
            "Advanced RAG Architectures"
        ]

        for category in required_categories:
            assert category in KnowledgeGaps.GAPS, f"Missing category: {category}"
            assert len(KnowledgeGaps.GAPS[category]) > 0, f"Category {category} has no items"

        print(f"✅ Knowledge gaps has {len(KnowledgeGaps.GAPS)} categories")
        return True
    except Exception as e:
        print(f"❌ Knowledge gaps test failed: {e}")
        return False


def test_learning_paths():
    """Test that learning paths have complete decision cards"""
    print("\nTesting LearningPaths...")
    try:
        from m4_4_planning_tools import LearningPaths

        # Check paths exist
        assert len(LearningPaths.PATHS) == 3, f"Expected 3 paths, found {len(LearningPaths.PATHS)}"

        # Verify each path has required fields
        required_fields = ["duration", "cost", "use_when", "avoid_when"]

        for path_name, path_details in LearningPaths.PATHS.items():
            for field in required_fields:
                assert field in path_details, f"Path '{path_name}' missing field: {field}"

            # Check use_when and avoid_when have content
            assert len(path_details["use_when"]) > 0, f"Path '{path_name}' has empty use_when"
            assert len(path_details["avoid_when"]) > 0, f"Path '{path_name}' has empty avoid_when"

        print(f"✅ All {len(LearningPaths.PATHS)} learning paths have complete decision cards")
        return True
    except Exception as e:
        print(f"❌ Learning paths test failed: {e}")
        return False


def test_action_plan_template():
    """Test that action plan template generates correctly"""
    print("\nTesting ActionPlanTemplate...")
    try:
        from m4_4_planning_tools import ActionPlanTemplate

        # Generate a sample plan
        plan = ActionPlanTemplate.generate("Test Path")

        # Verify plan has required sections
        required_sections = [
            "30-DAY ACTION PLAN",
            "DAYS 1-10",
            "DAYS 11-20",
            "DAYS 21-30"
        ]

        for section in required_sections:
            assert section in plan, f"Action plan missing section: {section}"

        print("✅ Action plan template generates with all required sections")
        return True
    except Exception as e:
        print(f"❌ Action plan template test failed: {e}")
        return False


def test_post_course_mistakes():
    """Test that post-course mistakes are defined"""
    print("\nTesting PostCourseMistakes...")
    try:
        from m4_4_planning_tools import PostCourseMistakes

        # Check mistakes exist
        assert len(PostCourseMistakes.MISTAKES) == 5, \
            f"Expected 5 mistakes, found {len(PostCourseMistakes.MISTAKES)}"

        # Verify each mistake has symptom and solution
        for mistake_name, details in PostCourseMistakes.MISTAKES.items():
            assert "symptom" in details, f"Mistake '{mistake_name}' missing symptom"
            assert "solution" in details, f"Mistake '{mistake_name}' missing solution"

        print(f"✅ All {len(PostCourseMistakes.MISTAKES)} post-course mistakes defined")
        return True
    except Exception as e:
        print(f"❌ Post-course mistakes test failed: {e}")
        return False


def test_skills_csv_generation():
    """Test that skills matrix CSV generates correctly"""
    print("\nTesting Skills Matrix CSV generation...")
    try:
        from m4_4_planning_tools import generate_skills_csv

        # Generate the CSV
        generate_skills_csv()

        # Check file exists
        csv_path = "templates/skills_matrix.csv"
        assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"

        # Verify CSV structure
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # Check required columns
            required_columns = ["Category", "Skill", "Confidence (1-5)", "Notes"]
            for col in required_columns:
                assert col in headers, f"Missing column: {col}"

            # Count rows
            rows = list(reader)
            assert len(rows) > 10, "CSV should have more than 10 skills"

        print(f"✅ Skills matrix CSV created with {len(rows)} skills")
        return True
    except Exception as e:
        print(f"❌ Skills CSV generation test failed: {e}")
        return False


def test_action_plan_file_generation():
    """Test that action plan markdown file generates correctly"""
    print("\nTesting Action Plan File generation...")
    try:
        from m4_4_planning_tools import generate_action_plan_file

        # Generate the file
        generate_action_plan_file()

        # Check file exists
        plan_path = "templates/action_plan.md"
        assert os.path.exists(plan_path), f"Action plan file not created at {plan_path}"

        # Verify content
        with open(plan_path, 'r') as f:
            content = f.read()

            # Check for required sections
            required_sections = [
                "30-Day Action Plan",
                "Days 1-10",
                "Days 11-20",
                "Days 21-30",
                "Weekly Review"
            ]

            for section in required_sections:
                assert section in content, f"Action plan missing section: {section}"

        print("✅ Action plan file created with all required sections")
        return True
    except Exception as e:
        print(f"❌ Action plan file generation test failed: {e}")
        return False


def test_notebook_exists():
    """Test that the Jupyter notebook exists and has content"""
    print("\nTesting Notebook existence...")
    try:
        notebook_path = "M4_4_Wrap_Up_and_Next_Steps.ipynb"
        assert os.path.exists(notebook_path), f"Notebook not found at {notebook_path}"

        # Check file size (should be substantial)
        file_size = os.path.getsize(notebook_path)
        assert file_size > 10000, f"Notebook seems too small: {file_size} bytes"

        print(f"✅ Notebook exists ({file_size:,} bytes)")
        return True
    except Exception as e:
        print(f"❌ Notebook test failed: {e}")
        return False


def test_templates_directory():
    """Test that templates directory exists"""
    print("\nTesting Templates directory...")
    try:
        templates_dir = "templates"
        assert os.path.exists(templates_dir), f"Templates directory not found"
        assert os.path.isdir(templates_dir), f"{templates_dir} is not a directory"

        # Check for expected files
        expected_files = ["action_plan.md", "skills_matrix.csv"]

        for file_name in expected_files:
            file_path = os.path.join(templates_dir, file_name)
            if os.path.exists(file_path):
                print(f"✅ Found template: {file_name}")

        return True
    except Exception as e:
        print(f"❌ Templates directory test failed: {e}")
        return False


def run_all_tests():
    """Run all smoke tests"""
    print("="*70)
    print("RUNNING M4.4 PLANNING TOOLS SMOKE TESTS")
    print("="*70)

    tests = [
        ("Import Test", test_imports),
        ("Skills Checklist", test_skills_checklist),
        ("Knowledge Gaps", test_knowledge_gaps),
        ("Learning Paths", test_learning_paths),
        ("Action Plan Template", test_action_plan_template),
        ("Post-Course Mistakes", test_post_course_mistakes),
        ("Skills CSV Generation", test_skills_csv_generation),
        ("Action Plan File Generation", test_action_plan_file_generation),
        ("Notebook Exists", test_notebook_exists),
        ("Templates Directory", test_templates_directory)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} - {test_name}")

    print("="*70)
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Everything is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
