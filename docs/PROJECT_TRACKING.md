# Project Tracking System

This project tracking system helps manage issues and project progress for the satellite detection and segmentation project. Issues are grouped by status: **open**, **in progress**, **under validation**, and **closed**.

## Features

- ✅ Create, update, and delete issues
- ✅ Group issues by status (open, in progress, under validation, closed)
- ✅ Set priorities (low, medium, high, critical)
- ✅ Assign issues to team members
- ✅ Add labels for categorization
- ✅ Rich CLI interface with colored output
- ✅ Project dashboard with status overview
- ✅ JSON-based persistence

## Quick Start

### Installation

The project tracker is included in the repository. Install dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Show project dashboard
python project_tracker.py dashboard

# Create a new issue
python project_tracker.py create "Implement YOLOv8 model" \
    --description "Add YOLOv8 for real-time detection" \
    --priority high \
    --assignee "ml-team"

# List all issues
python project_tracker.py list

# List issues by status
python project_tracker.py list --status open
python project_tracker.py list --status "in_progress"

# Show issue details
python project_tracker.py show <issue_id>

# Update issue status
python project_tracker.py update <issue_id> --status "in_progress"
python project_tracker.py update <issue_id> --status "under_validation"
python project_tracker.py update <issue_id> --status closed

# Delete an issue
python project_tracker.py delete <issue_id>
```

## Status Workflow

Issues follow this typical workflow:

1. **open** - New issues that need to be addressed
2. **in progress** - Issues currently being worked on
3. **under validation** - Issues completed but awaiting review/testing
4. **closed** - Issues that are fully completed and verified

## Priority Levels

- **low** - Nice to have features or minor improvements
- **medium** - Standard features and improvements  
- **high** - Important features or bug fixes
- **critical** - Urgent issues that block progress

## Project Structure

```
src/project_tracking/
├── __init__.py          # Module initialization
├── models.py            # Issue and Project data models
├── tracker.py           # Core tracking logic
└── cli.py              # Command line interface

configs/
└── project_tracking.yaml  # Configuration settings

project_tracker.py       # Main CLI entry point
project_state.json       # Persistent project data (auto-created)
```

## Dashboard Example

```
📊 satellite-detection-segmentation - Project Dashboard

╭── Open ──╮ ╭── In Progress ──╮ ╭── Under Validation ──╮ ╭── Closed ──╮
│          │ │                 │ │                      │ │            │
│  2       │ │  1              │ │  1                   │ │  0         │
│          │ │                 │ │                      │ │            │
╰──────────╯ ╰─────────────────╯ ╰──────────────────────╯ ╰────────────╯

📋 Issues by Priority
                    
  Priority   Count  
 ────────────────── 
  Low            1  
  Medium         1  
  High           2  
  Critical       0  

🕒 Recent Activity
                                                                                  
  ID         Title                                Status             Updated      
 ──────────────────────────────────────────────────────────────────────────────── 
  3cbd271e   Set up MLflow experiment tracking    under validation   08-21 08:28  
  1eb50753   Set up data preprocessing pipeline   in progress        08-21 08:28  
  e2897bf0   Implement DETR model                 open               08-21 08:28  
  67debf1a   Implement U-Net model                open               08-21 08:28
```

## Integration with Development Workflow

The project tracker is designed to integrate with your ML development workflow:

1. **Planning Phase**: Create issues for features, experiments, and improvements
2. **Development**: Move issues to "in progress" when starting work
3. **Review**: Move to "under validation" for peer review, testing, or validation
4. **Completion**: Close issues when fully implemented and verified

## Advanced Usage

### Filtering and Searching

```bash
# Filter by assignee
python project_tracker.py list --assignee "ml-team"

# Combine filters
python project_tracker.py list --status open --assignee "data-team"
```

### Labels and Organization

```bash
# Create issue with labels
python project_tracker.py create "Optimize model inference" \
    --labels "performance,optimization,inference"
```

### Custom Project File

```bash
# Use custom project file location
python project_tracker.py --project-file "/path/to/custom.json" dashboard
```

## Contributing

To extend the project tracker:

1. Add new fields to `models.py` 
2. Update serialization in `tracker.py`
3. Extend CLI commands in `cli.py`
4. Update configuration in `configs/project_tracking.yaml`

The system is designed to be extensible and can be integrated with external tools like GitHub Issues, Jira, or MLflow experiments.