#!/usr/bin/env python3
"""
Simple test script for project tracking functionality.
Tests core functionality without external test frameworks.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.project_tracking.models import Issue, Project, IssueStatus, IssuePriority
from src.project_tracking.tracker import ProjectTracker


def test_issue_creation():
    """Test issue creation and status updates."""
    print("Testing issue creation...")
    
    issue = Issue(
        title="Test Issue",
        description="Test description",
        priority=IssuePriority.HIGH
    )
    
    assert issue.title == "Test Issue"
    assert issue.status == IssueStatus.OPEN
    assert issue.priority == IssuePriority.HIGH
    assert len(issue.issue_id) == 8
    
    print("✓ Issue creation works")


def test_status_updates():
    """Test issue status transitions."""
    print("Testing status updates...")
    
    issue = Issue(title="Test Issue")
    original_updated = issue.updated_at
    
    # Test status update
    issue.update_status(IssueStatus.IN_PROGRESS)
    assert issue.status == IssueStatus.IN_PROGRESS
    assert issue.updated_at > original_updated
    assert issue.closed_at is None
    
    # Test closing
    issue.update_status(IssueStatus.CLOSED)
    assert issue.status == IssueStatus.CLOSED
    assert issue.closed_at is not None
    
    # Test reopening
    issue.update_status(IssueStatus.OPEN)
    assert issue.status == IssueStatus.OPEN
    assert issue.closed_at is None
    
    print("✓ Status updates work")


def test_project_management():
    """Test project-level operations."""
    print("Testing project management...")
    
    project = Project(name="Test Project")
    
    # Add issues
    issue1 = Issue(title="Issue 1", priority=IssuePriority.HIGH)
    issue2 = Issue(title="Issue 2", status=IssueStatus.IN_PROGRESS)
    
    project.add_issue(issue1)
    project.add_issue(issue2)
    
    assert len(project.issues) == 2
    
    # Test filtering
    open_issues = project.get_issues_by_status(IssueStatus.OPEN)
    assert len(open_issues) == 1
    assert open_issues[0].title == "Issue 1"
    
    in_progress_issues = project.get_issues_by_status(IssueStatus.IN_PROGRESS)
    assert len(in_progress_issues) == 1
    assert in_progress_issues[0].title == "Issue 2"
    
    # Test summary
    summary = project.get_status_summary()
    assert summary[IssueStatus.OPEN] == 1
    assert summary[IssueStatus.IN_PROGRESS] == 1
    assert summary[IssueStatus.CLOSED] == 0
    
    print("✓ Project management works")


def test_tracker_persistence():
    """Test tracker persistence functionality."""
    print("Testing tracker persistence...")
    
    # Use temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # Create tracker and add issues
        tracker1 = ProjectTracker(temp_file)
        issue = tracker1.create_issue(
            title="Persistent Issue",
            description="Test persistence",
            priority=IssuePriority.MEDIUM
        )
        issue_id = issue.issue_id
        
        # Update status
        tracker1.update_issue_status(issue_id, IssueStatus.IN_PROGRESS)
        
        # Create new tracker instance (simulates reload)
        tracker2 = ProjectTracker(temp_file)
        
        # Verify data persisted
        assert len(tracker2.project.issues) == 1
        loaded_issue = tracker2.project.get_issue(issue_id)
        assert loaded_issue is not None
        assert loaded_issue.title == "Persistent Issue"
        assert loaded_issue.status == IssueStatus.IN_PROGRESS
        assert loaded_issue.priority == IssuePriority.MEDIUM
        
        print("✓ Tracker persistence works")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_dashboard_data():
    """Test dashboard data generation."""
    print("Testing dashboard data...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        tracker = ProjectTracker(temp_file)
        
        # Create test issues
        tracker.create_issue("Issue 1", priority=IssuePriority.HIGH)
        issue2 = tracker.create_issue("Issue 2", priority=IssuePriority.LOW)
        tracker.update_issue_status(issue2.issue_id, IssueStatus.CLOSED)
        
        # Get dashboard data
        data = tracker.get_dashboard_data()
        
        assert data['project_name'] == "satellite-detection-segmentation"
        assert data['total_issues'] == 2
        assert data['status_summary']['open'] == 1
        assert data['status_summary']['closed'] == 1
        assert data['priority_summary']['high'] == 1
        assert data['priority_summary']['low'] == 1
        assert len(data['recent_activity']) == 2
        
        print("✓ Dashboard data works")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def run_tests():
    """Run all tests."""
    print("🧪 Running project tracking tests...\n")
    
    try:
        test_issue_creation()
        test_status_updates()
        test_project_management()
        test_tracker_persistence()
        test_dashboard_data()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)