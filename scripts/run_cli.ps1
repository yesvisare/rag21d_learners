# M4.4 CLI Wrapper (PowerShell)
# Non-interactive flags for automation

param(
    [switch]$Skills,
    [switch]$Plan
)

if ($Skills) {
    python m4_4_planning_tools.py --generate-skills-csv
    exit
}

if ($Plan) {
    python m4_4_planning_tools.py --generate-action-plan
    exit
}

# Interactive mode (no flags)
python m4_4_planning_tools.py
