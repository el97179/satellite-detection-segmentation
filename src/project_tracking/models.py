"""Data models for project tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import uuid


class IssueStatus(Enum):
    """Enumeration of possible issue statuses."""
    OPEN = "open"
    IN_PROGRESS = "in progress"
    UNDER_VALIDATION = "under validation"
    CLOSED = "closed"


class IssuePriority(Enum):
    """Enumeration of issue priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Issue:
    """Represents a project issue or task."""
    
    title: str
    description: str = ""
    status: IssueStatus = IssueStatus.OPEN
    priority: IssuePriority = IssuePriority.MEDIUM
    assignee: str = ""
    labels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    issue_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        """Ensure status is IssueStatus enum."""
        if isinstance(self.status, str):
            self.status = IssueStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = IssuePriority(self.priority)
    
    def update_status(self, new_status: IssueStatus):
        """Update issue status and timestamps."""
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status == IssueStatus.CLOSED and self.closed_at is None:
            self.closed_at = datetime.now()
        elif new_status != IssueStatus.CLOSED:
            self.closed_at = None
    
    def add_label(self, label: str):
        """Add a label to the issue."""
        if label not in self.labels:
            self.labels.append(label)
            self.updated_at = datetime.now()
    
    def remove_label(self, label: str):
        """Remove a label from the issue."""
        if label in self.labels:
            self.labels.remove(label)
            self.updated_at = datetime.now()


@dataclass
class Project:
    """Represents the overall project with its issues."""
    
    name: str
    description: str = ""
    issues: List[Issue] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_issue(self, issue: Issue):
        """Add an issue to the project."""
        self.issues.append(issue)
        self.updated_at = datetime.now()
    
    def remove_issue(self, issue_id: str) -> bool:
        """Remove an issue by ID. Returns True if found and removed."""
        for i, issue in enumerate(self.issues):
            if issue.issue_id == issue_id:
                self.issues.pop(i)
                self.updated_at = datetime.now()
                return True
        return False
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by ID."""
        for issue in self.issues:
            if issue.issue_id == issue_id:
                return issue
        return None
    
    def get_issues_by_status(self, status: IssueStatus) -> List[Issue]:
        """Get all issues with a specific status."""
        return [issue for issue in self.issues if issue.status == status]
    
    def get_issues_by_priority(self, priority: IssuePriority) -> List[Issue]:
        """Get all issues with a specific priority."""
        return [issue for issue in self.issues if issue.priority == priority]
    
    def get_issues_by_assignee(self, assignee: str) -> List[Issue]:
        """Get all issues assigned to a specific person."""
        return [issue for issue in self.issues if issue.assignee == assignee]
    
    def get_status_summary(self) -> dict:
        """Get a summary of issues by status."""
        summary = {status: 0 for status in IssueStatus}
        for issue in self.issues:
            summary[issue.status] += 1
        return summary