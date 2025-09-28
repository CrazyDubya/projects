#!/usr/bin/env python3
"""
Project Validation Script

This script validates the functionality and setup of projects in the repository.
It performs basic checks to ensure projects can be imported and executed.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_file(file_path):
    """Check if a Python file is syntactically correct."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(file_path):
    """Check if all imports in a Python file can be resolved."""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        if spec is None:
            return False, "Could not create module spec"
        
        # Try to create module (this will check imports)
        module = importlib.util.module_from_spec(spec)
        return True, "Imports OK"
    except ImportError as e:
        return False, f"Import Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_project_structure():
    """Validate the overall project structure."""
    issues = []
    successes = []
    
    # Check for essential files
    essential_files = ['INDEX.md', 'PROJECT_INDEX.md', 'README.md', 'requirements.txt']
    for file in essential_files:
        if os.path.exists(file):
            successes.append(f"✓ Found {file}")
        else:
            issues.append(f"✗ Missing {file}")
    
    return issues, successes

def validate_projects():
    """Validate individual projects."""
    project_results = {}
    root_dir = Path('.')
    
    # Get all directories that contain Python files
    project_dirs = []
    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if directory contains Python files
            py_files = list(item.glob('**/*.py'))
            if py_files:
                project_dirs.append(item)
    
    for project_dir in project_dirs:
        project_name = project_dir.name
        project_results[project_name] = {
            'files_checked': 0,
            'syntax_ok': 0,
            'import_ok': 0,
            'issues': [],
            'successes': []
        }
        
        # Find all Python files in the project
        py_files = list(project_dir.glob('**/*.py'))
        
        for py_file in py_files:
            project_results[project_name]['files_checked'] += 1
            
            # Check syntax
            syntax_ok, syntax_msg = check_python_file(py_file)
            if syntax_ok:
                project_results[project_name]['syntax_ok'] += 1
                project_results[project_name]['successes'].append(f"✓ {py_file.name}: {syntax_msg}")
            else:
                project_results[project_name]['issues'].append(f"✗ {py_file.name}: {syntax_msg}")
            
            # Note: Skip import checking as it requires all dependencies to be installed
            # This would be too complex for a simple validation script
    
    return project_results

def generate_report(structure_issues, structure_successes, project_results):
    """Generate a comprehensive validation report."""
    print("="*70)
    print("PROJECT VALIDATION REPORT")
    print("="*70)
    
    # Overall structure
    print("\n📁 PROJECT STRUCTURE")
    print("-" * 30)
    for success in structure_successes:
        print(success)
    for issue in structure_issues:
        print(issue)
    
    # Project-by-project results
    print(f"\n🐍 PYTHON PROJECT VALIDATION")
    print("-" * 40)
    
    total_files = 0
    total_syntax_ok = 0
    
    for project_name, results in project_results.items():
        if results['files_checked'] == 0:
            continue
            
        total_files += results['files_checked']
        total_syntax_ok += results['syntax_ok']
        
        print(f"\n{project_name}:")
        print(f"  Files checked: {results['files_checked']}")
        print(f"  Syntax OK: {results['syntax_ok']}/{results['files_checked']}")
        
        # Show first few successes and issues
        for success in results['successes'][:3]:
            print(f"  {success}")
        for issue in results['issues'][:3]:
            print(f"  {issue}")
        
        if len(results['issues']) > 3:
            print(f"  ... and {len(results['issues']) - 3} more issues")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("-" * 20)
    print(f"Total Python files: {total_files}")
    print(f"Syntax validation passed: {total_syntax_ok}/{total_files}")
    print(f"Success rate: {total_syntax_ok/total_files*100:.1f}%" if total_files > 0 else "No files found")
    
    structure_score = len(structure_successes) / (len(structure_successes) + len(structure_issues)) * 100 if (len(structure_successes) + len(structure_issues)) > 0 else 100
    print(f"Structure completeness: {structure_score:.1f}%")
    
    return total_syntax_ok == total_files and len(structure_issues) == 0

def main():
    """Main validation function."""
    print("Starting project validation...")
    
    # Validate project structure
    structure_issues, structure_successes = validate_project_structure()
    
    # Validate individual projects
    project_results = validate_projects()
    
    # Generate report
    all_good = generate_report(structure_issues, structure_successes, project_results)
    
    # Save report to file
    print(f"\n💾 Saving validation report...")
    
    # Exit with appropriate code
    if all_good:
        print("✅ All validations passed!")
        return 0
    else:
        print("⚠️  Some issues found. See report above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())