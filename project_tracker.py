#!/usr/bin/env python3
"""
Project Tracker CLI for Satellite Detection and Segmentation Project

This tool helps track issues and project progress with status grouping:
- open
- in progress  
- under validation
- closed

Usage:
    python project_tracker.py dashboard              # Show project dashboard
    python project_tracker.py create "Issue title"   # Create new issue
    python project_tracker.py list                   # List all issues
    python project_tracker.py list --status open     # List issues by status
    python project_tracker.py show <issue_id>        # Show issue details
    python project_tracker.py update <issue_id> --status "in progress"  # Update issue
"""

import click
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.project_tracking.cli import CLI
from src.project_tracking.models import IssueStatus, IssuePriority


@click.group()
@click.option('--project-file', default='project_state.json', 
              help='Project state file location')
@click.pass_context
def cli(ctx, project_file):
    """Project tracker for satellite detection and segmentation."""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = CLI(project_file)


@cli.command()
@click.pass_context
def dashboard(ctx):
    """Show project dashboard with status overview."""
    ctx.obj['cli'].dashboard()


@cli.command()
@click.argument('title')
@click.option('--description', '-d', default='', help='Issue description')
@click.option('--priority', '-p', default='medium', 
              type=click.Choice(['low', 'medium', 'high', 'critical']),
              help='Issue priority')
@click.option('--assignee', '-a', default='', help='Assigned person')
@click.option('--labels', '-l', default='', help='Comma-separated labels')
@click.pass_context
def create(ctx, title, description, priority, assignee, labels):
    """Create a new issue."""
    ctx.obj['cli'].create_issue(title, description, priority, assignee, labels)


@cli.command()
@click.option('--status', '-s', 
              type=click.Choice(['open', 'in_progress', 'under_validation', 'closed']),
              help='Filter by status')
@click.option('--assignee', '-a', help='Filter by assignee')
@click.pass_context
def list_issues(ctx, status, assignee):
    """List issues with optional filtering."""
    status_formatted = status.replace('_', ' ') if status else None
    ctx.obj['cli'].list_issues(status_formatted, assignee)


@cli.command()
@click.argument('issue_id')
@click.pass_context
def show(ctx, issue_id):
    """Show detailed information for an issue."""
    ctx.obj['cli'].show_issue(issue_id)


@cli.command()
@click.argument('issue_id')
@click.option('--status', '-s', 
              type=click.Choice(['open', 'in_progress', 'under_validation', 'closed']),
              help='New status')
@click.pass_context
def update(ctx, issue_id, status):
    """Update an issue's status."""
    if status:
        status_formatted = status.replace('_', ' ')
        ctx.obj['cli'].update_status(issue_id, status_formatted)
    else:
        click.echo("Please specify --status to update")


@cli.command()
@click.argument('issue_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_context
def delete(ctx, issue_id, force):
    """Delete an issue."""
    ctx.obj['cli'].delete_issue(issue_id, force)


# Alias for list command
cli.add_command(list_issues, name='list')


if __name__ == '__main__':
    cli()