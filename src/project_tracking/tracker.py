"""Project tracker for managing issues and project state."""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Issue, Project, IssueStatus, IssuePriority


class ProjectTracker:
    """Main class for managing project issues and state."""
    
    def __init__(self, project_file: str = "project_state.json"):
        """Initialize the project tracker."""
        self.project_file = Path(project_file)
        self.project = self._load_project()
    
    def _load_project(self) -> Project:
        """Load project from file or create a new one."""
        if self.project_file.exists():
            try:
                with open(self.project_file, 'r') as f:
                    data = json.load(f)
                return self._deserialize_project(data)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Could not load project file ({e}). Creating new project.")
        
        return Project(
            name="satellite-detection-segmentation",
            description="Satellite Detection and Segmentation using CNN and Transformer models"
        )
    
    def _deserialize_project(self, data: dict) -> Project:
        """Deserialize project data from dictionary."""
        issues = []
        for issue_data in data.get('issues', []):
            issue = Issue(
                title=issue_data['title'],
                description=issue_data.get('description', ''),
                status=IssueStatus(issue_data.get('status', 'open')),
                priority=IssuePriority(issue_data.get('priority', 'medium')),
                assignee=issue_data.get('assignee', ''),
                labels=issue_data.get('labels', []),
                created_at=datetime.fromisoformat(issue_data['created_at']),
                updated_at=datetime.fromisoformat(issue_data['updated_at']),
                closed_at=datetime.fromisoformat(issue_data['closed_at']) if issue_data.get('closed_at') else None,
                issue_id=issue_data['issue_id']
            )
            issues.append(issue)
        
        return Project(
            name=data.get('name', 'satellite-detection-segmentation'),
            description=data.get('description', ''),
            issues=issues,
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
    
    def _serialize_project(self) -> dict:
        """Serialize project to dictionary."""
        return {
            'name': self.project.name,
            'description': self.project.description,
            'created_at': self.project.created_at.isoformat(),
            'updated_at': self.project.updated_at.isoformat(),
            'issues': [
                {
                    'issue_id': issue.issue_id,
                    'title': issue.title,
                    'description': issue.description,
                    'status': issue.status.value,
                    'priority': issue.priority.value,
                    'assignee': issue.assignee,
                    'labels': issue.labels,
                    'created_at': issue.created_at.isoformat(),
                    'updated_at': issue.updated_at.isoformat(),
                    'closed_at': issue.closed_at.isoformat() if issue.closed_at else None
                }
                for issue in self.project.issues
            ]
        }
    
    def save_project(self):
        """Save project to file."""
        self.project.updated_at = datetime.now()
        data = self._serialize_project()
        
        with open(self.project_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_issue(self, title: str, description: str = "", priority: IssuePriority = IssuePriority.MEDIUM,
                    assignee: str = "", labels: List[str] = None) -> Issue:
        """Create a new issue."""
        if labels is None:
            labels = []
        
        issue = Issue(
            title=title,
            description=description,
            priority=priority,
            assignee=assignee,
            labels=labels
        )
        
        self.project.add_issue(issue)
        self.save_project()
        return issue
    
    def update_issue_status(self, issue_id: str, status: IssueStatus) -> bool:
        """Update issue status."""
        issue = self.project.get_issue(issue_id)
        if issue:
            issue.update_status(status)
            self.save_project()
            return True
        return False
    
    def update_issue(self, issue_id: str, **kwargs) -> bool:
        """Update issue fields."""
        issue = self.project.get_issue(issue_id)
        if not issue:
            return False
        
        for key, value in kwargs.items():
            if hasattr(issue, key):
                if key == 'status' and isinstance(value, str):
                    value = IssueStatus(value)
                elif key == 'priority' and isinstance(value, str):
                    value = IssuePriority(value)
                setattr(issue, key, value)
        
        issue.updated_at = datetime.now()
        self.save_project()
        return True
    
    def delete_issue(self, issue_id: str) -> bool:
        """Delete an issue."""
        if self.project.remove_issue(issue_id):
            self.save_project()
            return True
        return False
    
    def list_issues(self, status: Optional[IssueStatus] = None, 
                   assignee: Optional[str] = None) -> List[Issue]:
        """List issues with optional filtering."""
        issues = self.project.issues
        
        if status:
            issues = [issue for issue in issues if issue.status == status]
        
        if assignee:
            issues = [issue for issue in issues if issue.assignee == assignee]
        
        return sorted(issues, key=lambda x: x.updated_at, reverse=True)
    
    def get_dashboard_data(self) -> dict:
        """Get data for dashboard display."""
        status_summary = self.project.get_status_summary()
        
        # Convert enum keys to strings for serialization
        status_summary_str = {status.value: count for status, count in status_summary.items()}
        
        recent_issues = sorted(self.project.issues, key=lambda x: x.updated_at, reverse=True)[:5]
        
        priority_summary = {}
        for priority in IssuePriority:
            priority_summary[priority.value] = len(self.project.get_issues_by_priority(priority))
        
        return {
            'project_name': self.project.name,
            'total_issues': len(self.project.issues),
            'status_summary': status_summary_str,
            'priority_summary': priority_summary,
            'recent_activity': [
                {
                    'id': issue.issue_id,
                    'title': issue.title,
                    'status': issue.status.value,
                    'updated_at': issue.updated_at.isoformat()
                }
                for issue in recent_issues
            ]
        }