"""Command line interface for project tracking."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from datetime import datetime

from .models import IssueStatus, IssuePriority
from .tracker import ProjectTracker


console = Console()


def format_datetime(dt):
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M")


def get_status_color(status: IssueStatus) -> str:
    """Get color for status display."""
    colors = {
        IssueStatus.OPEN: "red",
        IssueStatus.IN_PROGRESS: "yellow", 
        IssueStatus.UNDER_VALIDATION: "blue",
        IssueStatus.CLOSED: "green"
    }
    return colors.get(status, "white")


def get_priority_color(priority: IssuePriority) -> str:
    """Get color for priority display."""
    colors = {
        IssuePriority.LOW: "dim cyan",
        IssuePriority.MEDIUM: "cyan",
        IssuePriority.HIGH: "yellow",
        IssuePriority.CRITICAL: "red bold"
    }
    return colors.get(priority, "white")


class CLI:
    """Command line interface for project tracking."""
    
    def __init__(self, project_file: str = "project_state.json"):
        self.tracker = ProjectTracker(project_file)
    
    def create_issue(self, title: str, description: str = "", priority: str = "medium",
                    assignee: str = "", labels: str = ""):
        """Create a new issue."""
        try:
            priority_enum = IssuePriority(priority.lower())
        except ValueError:
            console.print(f"[red]Invalid priority: {priority}. Use: low, medium, high, critical[/red]")
            return
        
        labels_list = [label.strip() for label in labels.split(",")] if labels else []
        
        issue = self.tracker.create_issue(
            title=title,
            description=description,
            priority=priority_enum,
            assignee=assignee,
            labels=labels_list
        )
        
        console.print(f"[green]✓ Created issue #{issue.issue_id}: {title}[/green]")
    
    def list_issues(self, status: str = None, assignee: str = None):
        """List issues with optional filtering."""
        status_enum = None
        if status:
            try:
                status_enum = IssueStatus(status.lower().replace("_", " "))
            except ValueError:
                console.print(f"[red]Invalid status: {status}[/red]")
                return
        
        issues = self.tracker.list_issues(status=status_enum, assignee=assignee)
        
        if not issues:
            console.print("[yellow]No issues found.[/yellow]")
            return
        
        table = Table(title="Project Issues", box=box.ROUNDED)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Status", no_wrap=True)
        table.add_column("Priority", no_wrap=True)
        table.add_column("Assignee", style="green")
        table.add_column("Updated", style="dim")
        
        for issue in issues:
            status_color = get_status_color(issue.status)
            priority_color = get_priority_color(issue.priority)
            
            table.add_row(
                issue.issue_id,
                issue.title[:50] + ("..." if len(issue.title) > 50 else ""),
                f"[{status_color}]{issue.status.value}[/{status_color}]",
                f"[{priority_color}]{issue.priority.value}[/{priority_color}]",
                issue.assignee or "-",
                format_datetime(issue.updated_at)
            )
        
        console.print(table)
    
    def update_status(self, issue_id: str, status: str):
        """Update issue status."""
        try:
            status_enum = IssueStatus(status.lower().replace("_", " "))
        except ValueError:
            console.print(f"[red]Invalid status: {status}. Use: open, in_progress, under_validation, closed[/red]")
            return
        
        if self.tracker.update_issue_status(issue_id, status_enum):
            console.print(f"[green]✓ Updated issue #{issue_id} status to {status_enum.value}[/green]")
        else:
            console.print(f"[red]Issue #{issue_id} not found[/red]")
    
    def show_issue(self, issue_id: str):
        """Show detailed issue information."""
        issue = self.tracker.project.get_issue(issue_id)
        if not issue:
            console.print(f"[red]Issue #{issue_id} not found[/red]")
            return
        
        status_color = get_status_color(issue.status)
        priority_color = get_priority_color(issue.priority)
        
        content = f"""
[bold]Title:[/bold] {issue.title}
[bold]ID:[/bold] {issue.issue_id}
[bold]Status:[/bold] [{status_color}]{issue.status.value}[/{status_color}]
[bold]Priority:[/bold] [{priority_color}]{issue.priority.value}[/{priority_color}]
[bold]Assignee:[/bold] {issue.assignee or "Unassigned"}
[bold]Labels:[/bold] {", ".join(issue.labels) if issue.labels else "None"}
[bold]Created:[/bold] {format_datetime(issue.created_at)}
[bold]Updated:[/bold] {format_datetime(issue.updated_at)}
[bold]Closed:[/bold] {format_datetime(issue.closed_at)}

[bold]Description:[/bold]
{issue.description or "No description provided"}
"""
        
        panel = Panel(content.strip(), title=f"Issue #{issue.issue_id}", border_style="blue")
        console.print(panel)
    
    def dashboard(self):
        """Show project dashboard."""
        data = self.tracker.get_dashboard_data()
        
        # Title
        console.print(f"\n[bold blue]📊 {data['project_name']} - Project Dashboard[/bold blue]\n")
        
        # Status summary
        status_panels = []
        for status, count in data['status_summary'].items():
            try:
                status_enum = IssueStatus(status)
                color = get_status_color(status_enum)
                panel = Panel(
                    f"[bold {color}]{count}[/bold {color}]",
                    title=status.title(),
                    border_style=color,
                    padding=(1, 2)
                )
                status_panels.append(panel)
            except ValueError:
                continue
        
        console.print(Columns(status_panels, equal=True))
        console.print()
        
        # Priority summary
        console.print("[bold]📋 Issues by Priority[/bold]")
        priority_table = Table(box=box.SIMPLE)
        priority_table.add_column("Priority", style="bold")
        priority_table.add_column("Count", justify="right")
        
        for priority, count in data['priority_summary'].items():
            try:
                priority_enum = IssuePriority(priority)
                color = get_priority_color(priority_enum)
                priority_table.add_row(
                    f"[{color}]{priority.title()}[/{color}]",
                    f"[{color}]{count}[/{color}]"
                )
            except ValueError:
                continue
        
        console.print(priority_table)
        console.print()
        
        # Recent activity
        if data['recent_activity']:
            console.print("[bold]🕒 Recent Activity[/bold]")
            activity_table = Table(box=box.SIMPLE)
            activity_table.add_column("ID", style="cyan")
            activity_table.add_column("Title")
            activity_table.add_column("Status")
            activity_table.add_column("Updated")
            
            for item in data['recent_activity']:
                try:
                    status_enum = IssueStatus(item['status'])
                    color = get_status_color(status_enum)
                    activity_table.add_row(
                        item['id'],
                        item['title'][:40] + ("..." if len(item['title']) > 40 else ""),
                        f"[{color}]{item['status']}[/{color}]",
                        datetime.fromisoformat(item['updated_at']).strftime("%m-%d %H:%M")
                    )
                except ValueError:
                    continue
            
            console.print(activity_table)
    
    def delete_issue(self, issue_id: str, force: bool = False):
        """Delete an issue."""
        issue = self.tracker.project.get_issue(issue_id)
        if not issue:
            console.print(f"[red]Issue #{issue_id} not found[/red]")
            return
        
        if not force:
            console.print(f"[yellow]Are you sure you want to delete issue #{issue_id}: {issue.title}?[/yellow]")
            confirm = click.confirm("This action cannot be undone")
            if not confirm:
                console.print("[yellow]Deletion cancelled[/yellow]")
                return
        
        if self.tracker.delete_issue(issue_id):
            console.print(f"[green]✓ Deleted issue #{issue_id}[/green]")
        else:
            console.print(f"[red]Failed to delete issue #{issue_id}[/red]")