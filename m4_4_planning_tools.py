#!/usr/bin/env python3
"""
M4.4 Planning Tools - Course Wrap-up & Next Steps
Simple CLI tool for skills assessment, gap analysis, and learning path planning
"""

import json
from typing import List, Dict
from datetime import datetime


class SkillsChecklist:
    """Skills assessment checklist from the course"""

    TECHNICAL_SKILLS = [
        "Implement vector search from scratch",
        "Build and deploy RAG systems",
        "Design chunking strategies for different content types",
        "Evaluate retrieval quality with metrics",
        "Implement hybrid search (sparse + dense)",
        "Work with multiple vector databases",
        "Build production APIs with FastAPI",
        "Handle conversation memory and context",
        "Engineer prompts for better LLM outputs",
        "Deploy containerized applications"
    ]

    SOFT_SKILLS = [
        "Break down complex problems",
        "Make architecture decisions",
        "Debug production issues",
        "Document work professionally",
        "Explain technical concepts clearly"
    ]

    @classmethod
    def display(cls):
        """Display the skills checklist"""
        print("\n" + "="*70)
        print("SKILLS YOU'VE GAINED FROM THIS COURSE")
        print("="*70)

        print("\n📚 TECHNICAL SKILLS:")
        for i, skill in enumerate(cls.TECHNICAL_SKILLS, 1):
            print(f"  {i:2d}. [ ] {skill}")

        print("\n🧠 SOFT SKILLS:")
        for i, skill in enumerate(cls.SOFT_SKILLS, 1):
            print(f"  {i:2d}. [ ] {skill}")

        print("\n" + "="*70)
        print("💡 Mark the skills you feel confident in!")
        print("="*70 + "\n")


class KnowledgeGaps:
    """What the course didn't teach - honest assessment"""

    GAPS = {
        "Model Training & Fine-tuning": [
            "Training models from scratch",
            "Fine-tuning for domain-specific tasks",
            "Understanding transformer architectures deeply",
            "Distributed training infrastructure"
        ],
        "MLOps & Production ML": [
            "ML experiment tracking (MLflow, W&B)",
            "Model versioning and registry",
            "A/B testing for ML models",
            "Feature stores and data pipelines",
            "Advanced monitoring and observability",
            "Model drift detection"
        ],
        "Advanced RAG Architectures": [
            "Graph RAG for complex relationships",
            "Multi-modal RAG (images, audio, video)",
            "Agentic RAG with tool use",
            "Self-improving RAG systems",
            "Advanced reranking strategies"
        ],
        "Evaluation at Scale": [
            "Large-scale evaluation frameworks",
            "Human evaluation pipelines",
            "Statistical significance testing",
            "Cost-quality tradeoff optimization"
        ],
        "Research-Level Topics": [
            "Novel architecture design",
            "Publishing papers",
            "Benchmarking methodologies",
            "Contributing to foundational models"
        ]
    }

    @classmethod
    def display(cls):
        """Display knowledge gaps"""
        print("\n" + "="*70)
        print("WHAT THIS COURSE DIDN'T TEACH")
        print("="*70)
        print("\n⚠️  These gaps guide your continued learning!\n")

        for category, items in cls.GAPS.items():
            print(f"\n❌ {category}:")
            for item in items:
                print(f"   • {item}")

        print("\n" + "="*70)
        print("💡 You're at the BEGINNING of your AI journey, not the end.")
        print("="*70 + "\n")


class LearningPaths:
    """Three main learning paths with decision criteria"""

    PATHS = {
        "Path 1: Framework Mastery (LangChain/LlamaIndex)": {
            "duration": "4-6 weeks (15-20 hrs/week)",
            "cost": "$50-200 for courses",
            "use_when": [
                "Goal is to ship products quickly",
                "Comfortable with abstraction layers",
                "Targeting AI engineer roles",
                "Have 15+ hours/week available",
                "Enjoy rapid prototyping"
            ],
            "avoid_when": [
                "Goal is ML research or PhD",
                "Need to optimize costs/performance at scale",
                "Prefer stability over cutting-edge",
                "Limited time (<10 hours/week)"
            ]
        },
        "Path 2: RAG Specialization (Deep Expertise)": {
            "duration": "6-12 months (15-20 hrs/week)",
            "cost": "$0-500 (papers free, optional courses)",
            "use_when": [
                "Fascinated by retrieval and search",
                "Enjoy research and experimentation",
                "Targeting ML engineer roles",
                "Comfortable with academic papers",
                "Want to publish or contribute to research"
            ],
            "avoid_when": [
                "Need job quickly (<3 months)",
                "Prefer building products over research",
                "Uncomfortable with math/papers",
                "Limited time (<10 hours/week)"
            ]
        },
        "Path 3: Full-Stack AI Engineer": {
            "duration": "9-12 months (20-25 hrs/week)",
            "cost": "$200-800 (courses, hosting, domains)",
            "use_when": [
                "Want to build complete products",
                "Targeting startup roles",
                "Enjoy variety over deep specialization",
                "Have 20+ hours/week available",
                "Entrepreneurial mindset"
            ],
            "avoid_when": [
                "Want to specialize in ML/AI deeply",
                "Need job in <6 months",
                "Prefer depth over breadth",
                "Limited time (<15 hours/week)",
                "Targeting FAANG ML roles"
            ]
        }
    }

    @classmethod
    def display(cls):
        """Display learning paths"""
        print("\n" + "="*70)
        print("YOUR NEXT LEARNING PATHS")
        print("="*70)

        for i, (path, details) in enumerate(cls.PATHS.items(), 1):
            print(f"\n{'='*70}")
            print(f"{i}. {path}")
            print(f"{'='*70}")
            print(f"⏱️  Duration: {details['duration']}")
            print(f"💰 Cost: {details['cost']}")

            print(f"\n✅ USE WHEN:")
            for criteria in details['use_when']:
                print(f"   • {criteria}")

            print(f"\n🚫 AVOID WHEN:")
            for criteria in details['avoid_when']:
                print(f"   • {criteria}")

        print("\n" + "="*70)
        print("💡 Pick based on career goals, time, and learning style!")
        print("="*70 + "\n")


class ActionPlanTemplate:
    """30-day action plan structure"""

    @staticmethod
    def generate(path_choice: str = "General") -> str:
        """Generate a 30-day action plan"""
        plan = f"""
{'='*70}
30-DAY ACTION PLAN - {path_choice.upper()}
{'='*70}

📅 DAYS 1-10: Foundation & Polish
---------------------------------
[ ] Polish one portfolio project from the course
[ ] Deploy it with proper documentation
[ ] Write a case study / blog post about it
[ ] Share on LinkedIn with demo video/screenshots
[ ] Join 2-3 AI communities (Discord/forums)

📅 DAYS 11-20: Build & Learn
----------------------------
[ ] Start a new project applying advanced techniques
[ ] Document your learning publicly (weekly posts)
[ ] Engage with community (answer questions, help others)
[ ] Read 2-3 technical articles/papers in your chosen path
[ ] Update GitHub with clean, documented code

📅 DAYS 21-30: Network & Give Back
----------------------------------
[ ] Contribute to open source (docs, code, or issues)
[ ] Write a tutorial or record a video
[ ] Connect with 5 practitioners in the field
[ ] Review your progress and set next 30-day goals
[ ] Celebrate wins and share learnings

{'='*70}
💡 Weekly Check: Did I build, share, engage, learn, and connect?
{'='*70}
"""
        return plan

    @classmethod
    def display(cls, path: str = "General"):
        """Display action plan"""
        print(cls.generate(path))


class PostCourseMistakes:
    """Common mistakes to avoid"""

    MISTAKES = {
        "Tutorial Hell": {
            "symptom": "Taking more courses instead of building",
            "solution": "70% building, 30% learning. Build for 3-4 weeks before next course."
        },
        "Building in Private": {
            "symptom": "Zero portfolio visibility, hidden GitHub repos",
            "solution": "Make repos public, post on LinkedIn, share weekly learnings."
        },
        "Perfectionism Paralysis": {
            "symptom": "Never shipping, endless refining",
            "solution": "2-3 week deadline per project. Ship with 'Known Limitations'."
        },
        "Jumping to Advanced Topics": {
            "symptom": "Weak fundamentals, can't debug issues",
            "solution": "Master basics deeply first. Can you explain embeddings to a 10-year-old?"
        },
        "Learning in Isolation": {
            "symptom": "No network, no referrals, struggling alone",
            "solution": "Join communities, comment daily, ask questions, help others."
        }
    }

    @classmethod
    def display(cls):
        """Display common mistakes"""
        print("\n" + "="*70)
        print("COMMON POST-COURSE MISTAKES TO AVOID")
        print("="*70)

        for i, (mistake, details) in enumerate(cls.MISTAKES.items(), 1):
            print(f"\n🚫 Mistake #{i}: {mistake}")
            print(f"   Symptom: {details['symptom']}")
            print(f"   ✅ Solution: {details['solution']}")

        print("\n" + "="*70)
        print("💡 Weekly check: Building? Sharing? Engaging? Fundamentals? Network?")
        print("="*70 + "\n")


class JobSearchReality:
    """Realistic job timeline and expectations"""

    @staticmethod
    def display():
        """Display job search reality"""
        print("\n" + "="*70)
        print("JOB SEARCH REALITY CHECK")
        print("="*70)

        print("\n⏱️  TYPICAL TIMELINE:")
        print("   • Entry-level AI roles: 3-6 months from course completion to offer")
        print("   • Career switchers: 6-12 months typical")
        print("   • Junior roles receive: 200-300 applications each")
        print("   • 50-100 applications typical before offers")

        print("\n📅 REALISTIC BREAKDOWN:")
        print("   Weeks 1-4:   Portfolio polish, build 1 new project, network")
        print("   Weeks 5-12:  Active applications (10-20/week), interview prep")
        print("   Weeks 13-20: Second rounds, technical challenges")
        print("   Weeks 21-24: Offers and decision-making")

        print("\n✅ RIGHT-FIT ROLES:")
        print("   • AI Engineer (RAG focus)")
        print("   • Applied ML Engineer")
        print("   • RAG Solutions Engineer")
        print("   • Full-Stack AI Developer")
        print("   • AI Integration Specialist")

        print("\n❌ NEEDS MORE PREPARATION:")
        print("   • ML Research Scientist (needs PhD/papers)")
        print("   • ML Infrastructure Engineer (needs K8s/MLOps)")
        print("   • AI Safety / Alignment (different specialization)")
        print("   • AI Product Manager (needs PM experience)")

        print("\n" + "="*70)
        print("💡 Don't give up at week 8 when you've only heard 'no' - that's normal!")
        print("="*70 + "\n")


def main_menu():
    """Display main menu and handle user interaction"""

    while True:
        print("\n" + "="*70)
        print("M4.4 COURSE WRAP-UP PLANNING TOOLS")
        print("="*70)
        print("\nWhat would you like to explore?")
        print("\n1. 📋 Skills Checklist - What you've gained")
        print("2. ⚠️  Knowledge Gaps - What we didn't teach")
        print("3. 🛤️  Learning Paths - Where to go next")
        print("4. 📅 30-Day Action Plan - Get started")
        print("5. 🚫 Common Mistakes - What to avoid")
        print("6. 💼 Job Search Reality - Honest timelines")
        print("7. 📊 Generate Skills Matrix CSV")
        print("8. 📝 Generate Action Plan Template")
        print("9. ❌ Exit")

        choice = input("\nEnter your choice (1-9): ").strip()

        if choice == "1":
            SkillsChecklist.display()
        elif choice == "2":
            KnowledgeGaps.display()
        elif choice == "3":
            LearningPaths.display()
        elif choice == "4":
            print("\nWhich path are you considering?")
            print("1. Framework Mastery")
            print("2. RAG Specialization")
            print("3. Full-Stack AI")
            print("4. General (undecided)")
            path_choice = input("\nEnter choice (1-4): ").strip()
            paths = {
                "1": "Framework Mastery",
                "2": "RAG Specialization",
                "3": "Full-Stack AI",
                "4": "General"
            }
            ActionPlanTemplate.display(paths.get(path_choice, "General"))
        elif choice == "5":
            PostCourseMistakes.display()
        elif choice == "6":
            JobSearchReality.display()
        elif choice == "7":
            generate_skills_csv()
        elif choice == "8":
            generate_action_plan_file()
        elif choice == "9":
            print("\n🚀 Keep building! Keep learning! Keep sharing!\n")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


def generate_skills_csv():
    """Generate skills matrix CSV file for tracking course competencies.

    Creates a CSV file at templates/skills_matrix.csv containing all technical
    and soft skills from the course with columns for self-assessment (1-5 scale)
    and notes. The templates/ directory is created if it doesn't exist.

    Side Effects:
        Creates or overwrites templates/skills_matrix.csv
        Creates templates/ directory if missing
        Prints success message to stdout

    Example:
        >>> generate_skills_csv()
        ✅ Skills matrix CSV generated: templates/skills_matrix.csv
        💡 Fill in your confidence levels (1-5) and notes!
    """
    import csv
    import os

    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    filename = "templates/skills_matrix.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Skill", "Confidence (1-5)", "Notes"])

        # Technical skills
        for skill in SkillsChecklist.TECHNICAL_SKILLS:
            writer.writerow(["Technical", skill, "", ""])

        # Soft skills
        for skill in SkillsChecklist.SOFT_SKILLS:
            writer.writerow(["Soft Skills", skill, "", ""])

    print(f"\n✅ Skills matrix CSV generated: {filename}")
    print("💡 Fill in your confidence levels (1-5) and notes!")


def generate_action_plan_file():
    """Generate 30-day action plan markdown template for post-course career strategy.

    Creates a markdown file at templates/action_plan.md with a structured 30-day
    plan including learning path selection, daily task breakdowns, weekly review
    checklists, project planning sections, and reflection prompts. The templates/
    directory is created if it doesn't exist.

    Side Effects:
        Creates or overwrites templates/action_plan.md
        Creates templates/ directory if missing
        Prints success message to stdout with customization tip

    Example:
        >>> generate_action_plan_file()
        ✅ Action plan template generated: templates/action_plan.md
        💡 Customize it with your goals and track your progress!
    """
    import os

    os.makedirs("templates", exist_ok=True)

    filename = "templates/action_plan.md"

    content = f"""# 30-Day Action Plan - Course Wrap-up

**Generated:** {datetime.now().strftime("%Y-%m-%d")}

## My Learning Path Choice

- [ ] Path 1: Framework Mastery (LangChain/LlamaIndex)
- [ ] Path 2: RAG Specialization (Deep Expertise)
- [ ] Path 3: Full-Stack AI Engineer

## Days 1-10: Foundation & Polish

**Goal:** Polish existing work and establish online presence

- [ ] Polish one portfolio project from the course
- [ ] Deploy it with proper documentation
- [ ] Write a case study / blog post about it
- [ ] Share on LinkedIn with demo video/screenshots
- [ ] Join 2-3 AI communities (Discord/forums)

**Notes:**

## Days 11-20: Build & Learn

**Goal:** Apply advanced techniques and learn publicly

- [ ] Start a new project applying advanced techniques
- [ ] Document your learning publicly (weekly posts)
- [ ] Engage with community (answer questions, help others)
- [ ] Read 2-3 technical articles/papers in your chosen path
- [ ] Update GitHub with clean, documented code

**Notes:**

## Days 21-30: Network & Give Back

**Goal:** Contribute and grow your network

- [ ] Contribute to open source (docs, code, or issues)
- [ ] Write a tutorial or record a video
- [ ] Connect with 5 practitioners in the field
- [ ] Review your progress and set next 30-day goals
- [ ] Celebrate wins and share learnings

**Notes:**

## Weekly Review Checklist

**Week 1:**
- [ ] Built or shipped something this week
- [ ] Shared something publicly
- [ ] Engaged with the community
- [ ] Worked on fundamentals
- [ ] Made one connection

**Week 2:**
- [ ] Built or shipped something this week
- [ ] Shared something publicly
- [ ] Engaged with the community
- [ ] Worked on fundamentals
- [ ] Made one connection

**Week 3:**
- [ ] Built or shipped something this week
- [ ] Shared something publicly
- [ ] Engaged with the community
- [ ] Worked on fundamentals
- [ ] Made one connection

**Week 4:**
- [ ] Built or shipped something this week
- [ ] Shared something publicly
- [ ] Engaged with the community
- [ ] Worked on fundamentals
- [ ] Made one connection

## Projects to Build

1. **Project Idea:**
   - Description:
   - Tech Stack:
   - Timeline:
   - Learning Goals:

2. **Project Idea:**
   - Description:
   - Tech Stack:
   - Timeline:
   - Learning Goals:

## Resources to Explore

- Course/Tutorial:
- Book/Paper:
- Community:
- Person to follow:

## Reflection

**What went well:**

**What to improve:**

**Key learnings:**

**Next 30-day goals:**
"""

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\n✅ Action plan template generated: {filename}")
    print("💡 Customize it with your goals and track your progress!")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments for non-interactive automation
    parser = argparse.ArgumentParser(
        description="M4.4 Course Wrap-up Planning Tools - Interactive CLI or automated template generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python m4_4_planning_tools.py                      # Interactive menu
  python m4_4_planning_tools.py --generate-skills-csv
  python m4_4_planning_tools.py --generate-action-plan
  python m4_4_planning_tools.py --print-learning-paths
        """
    )

    parser.add_argument(
        "--generate-skills-csv",
        action="store_true",
        help="Generate templates/skills_matrix.csv and exit"
    )
    parser.add_argument(
        "--generate-action-plan",
        action="store_true",
        help="Generate templates/action_plan.md and exit"
    )
    parser.add_argument(
        "--print-learning-paths",
        action="store_true",
        help="Display all three learning paths with decision cards and exit"
    )

    args = parser.parse_args()

    # Handle non-interactive flags with early exit
    if args.generate_skills_csv:
        generate_skills_csv()
        exit(0)

    if args.generate_action_plan:
        generate_action_plan_file()
        exit(0)

    if args.print_learning_paths:
        LearningPaths.display()
        exit(0)

    # Interactive mode (default when no flags provided)
    print("\n🎉 Welcome to M4.4 Course Wrap-up Planning Tools!")
    print("Your guide to next steps after completing the RAG course.\n")
    main_menu()
