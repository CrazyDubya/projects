#!/usr/bin/env python3
"""
Project Evaluator - Comprehensive analysis and expansion tool
Evaluates all projects in the repository across multiple dimensions
"""

import os
import sys
import json
import subprocess
import ast
import re
from collections import defaultdict
from pathlib import Path
import importlib.util

class ProjectEvaluator:
    def __init__(self, repo_root='.'):
        self.repo_root = Path(repo_root).resolve()
        self.projects = {}
        self.metrics = {}
        
    def discover_projects(self):
        """Discover all projects in the repository"""
        projects = []
        
        # Skip certain directories
        skip_dirs = {'.git', '.github', 'node_modules', '__pycache__', 
                    'packages', '.venv', 'venv', 'env'}
        
        for item in self.repo_root.iterdir():
            if item.is_dir() and item.name not in skip_dirs:
                # Check if directory contains Python files
                py_files = list(item.glob('*.py'))
                if py_files:
                    projects.append({
                        'name': item.name,
                        'path': item,
                        'python_files': py_files,
                        'readme_path': item / 'README.md' if (item / 'README.md').exists() else None
                    })
        
        return projects
    
    def analyze_code_quality(self, project):
        """Analyze code quality metrics for a project"""
        metrics = {
            'total_lines': 0,
            'total_files': len(project['python_files']),
            'avg_lines_per_file': 0,
            'complexity_score': 0,
            'documentation_score': 0,
            'import_dependencies': set(),
            'syntax_errors': [],
            'maintainability_issues': [],
            'security_concerns': [],
        }
        
        for py_file in project['python_files']:
            try:
                file_metrics = self.analyze_file(py_file)
                metrics['total_lines'] += file_metrics['lines']
                metrics['complexity_score'] += file_metrics['complexity']
                metrics['documentation_score'] += file_metrics['documentation']
                metrics['import_dependencies'].update(file_metrics['imports'])
                metrics['syntax_errors'].extend(file_metrics['syntax_errors'])
                metrics['maintainability_issues'].extend(file_metrics['maintainability'])
                metrics['security_concerns'].extend(file_metrics['security'])
            except Exception as e:
                metrics['syntax_errors'].append(f"Error analyzing {py_file}: {e}")
        
        if metrics['total_files'] > 0:
            metrics['avg_lines_per_file'] = metrics['total_lines'] / metrics['total_files']
            metrics['complexity_score'] = metrics['complexity_score'] / metrics['total_files']
            metrics['documentation_score'] = metrics['documentation_score'] / metrics['total_files']
        
        # Convert set to list for JSON serialization
        metrics['import_dependencies'] = list(metrics['import_dependencies'])
        
        return metrics
    
    def analyze_file(self, file_path):
        """Analyze a single Python file"""
        metrics = {
            'lines': 0,
            'complexity': 0,
            'documentation': 0,
            'imports': set(),
            'syntax_errors': [],
            'maintainability': [],
            'security': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                metrics['lines'] = len(lines)
                
                # Parse AST for analysis
                try:
                    tree = ast.parse(content)
                    metrics.update(self.analyze_ast(tree, content, lines))
                except SyntaxError as e:
                    metrics['syntax_errors'].append(f"Syntax error in {file_path}: {e}")
                
        except Exception as e:
            metrics['syntax_errors'].append(f"Cannot read {file_path}: {e}")
            
        return metrics
    
    def analyze_ast(self, tree, content, lines):
        """Analyze AST for various metrics"""
        metrics = {
            'complexity': 0,
            'documentation': 0,
            'imports': set(),
            'maintainability': [],
            'security': []
        }
        
        # Count docstrings and comments
        docstring_count = 0
        comment_count = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Analyze AST nodes
        for node in ast.walk(tree):
            # Complexity analysis (cyclomatic complexity approximation)
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                metrics['complexity'] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                metrics['complexity'] += 1
                # Check for docstrings
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring_count += 1
            
            # Import analysis
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    metrics['imports'].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    metrics['imports'].add(node.module)
            
            # Security concerns
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and 
                    node.func.id in ['eval', 'exec', 'compile']):
                    metrics['security'].append(f"Potentially dangerous function: {node.func.id}")
                elif (isinstance(node.func, ast.Attribute) and 
                      node.func.attr in ['system', 'popen', 'spawn']):
                    metrics['security'].append(f"System call detected: {node.func.attr}")
        
        # Documentation score
        total_definitions = sum(1 for node in ast.walk(tree) 
                              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        if total_definitions > 0:
            metrics['documentation'] = (docstring_count / total_definitions) * 100
        
        # Maintainability issues
        if metrics['complexity'] > 20:
            metrics['maintainability'].append("High complexity detected")
        if len(lines) > 500:
            metrics['maintainability'].append("Large file (>500 lines)")
        if total_definitions > 20:
            metrics['maintainability'].append("Many definitions in single file")
            
        return metrics
    
    def analyze_dependencies(self, project):
        """Analyze project dependencies"""
        deps = {
            'external_libs': set(),
            'stdlib_libs': set(),
            'local_imports': set(),
            'dependency_health': 'unknown'
        }
        
        # Standard library modules (partial list)
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'subprocess', 'threading',
            'collections', 'pathlib', 're', 'ast', 'importlib', 'argparse',
            'logging', 'unittest', 'tkinter', 'sqlite3', 'http', 'urllib'
        }
        
        for py_file in project['python_files']:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Find import statements
                import_pattern = r'(?:from\s+(\w+(?:\.\w+)*)|import\s+(\w+(?:\.\w+)*))'
                matches = re.findall(import_pattern, content)
                
                for match in matches:
                    module = match[0] or match[1]
                    if module:
                        base_module = module.split('.')[0]
                        if base_module in stdlib_modules:
                            deps['stdlib_libs'].add(base_module)
                        elif base_module.startswith('.') or base_module in [p.stem for p in project['python_files']]:
                            deps['local_imports'].add(base_module)
                        else:
                            deps['external_libs'].add(base_module)
                            
            except Exception:
                continue
        
        # Convert sets to lists for JSON serialization
        deps['external_libs'] = list(deps['external_libs'])
        deps['stdlib_libs'] = list(deps['stdlib_libs'])
        deps['local_imports'] = list(deps['local_imports'])
        
        # Assess dependency health
        if len(deps['external_libs']) == 0:
            deps['dependency_health'] = 'excellent'
        elif len(deps['external_libs']) <= 3:
            deps['dependency_health'] = 'good'
        elif len(deps['external_libs']) <= 6:
            deps['dependency_health'] = 'moderate'
        else:
            deps['dependency_health'] = 'concerning'
            
        return deps
    
    def analyze_documentation(self, project):
        """Analyze project documentation quality"""
        doc_metrics = {
            'has_readme': project['readme_path'] is not None,
            'readme_quality': 0,
            'readme_sections': [],
            'installation_docs': False,
            'usage_examples': False,
            'api_docs': False
        }
        
        if project['readme_path'] and project['readme_path'].exists():
            try:
                with open(project['readme_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read().lower()
                    
                # Check for common sections
                sections = ['overview', 'installation', 'usage', 'example', 'api', 
                           'configuration', 'contributing', 'license']
                for section in sections:
                    if section in readme_content:
                        doc_metrics['readme_sections'].append(section)
                
                # Specific checks
                doc_metrics['installation_docs'] = any(word in readme_content for word in 
                                                     ['install', 'pip', 'setup', 'requirements'])
                doc_metrics['usage_examples'] = any(word in readme_content for word in 
                                                  ['example', 'usage', '```', 'python'])
                doc_metrics['api_docs'] = any(word in readme_content for word in 
                                            ['api', 'function', 'class', 'method'])
                
                # Quality score based on completeness
                doc_metrics['readme_quality'] = len(doc_metrics['readme_sections']) * 10
                if doc_metrics['installation_docs']:
                    doc_metrics['readme_quality'] += 15
                if doc_metrics['usage_examples']:
                    doc_metrics['readme_quality'] += 15
                if doc_metrics['api_docs']:
                    doc_metrics['readme_quality'] += 10
                    
            except Exception:
                pass
        
        return doc_metrics
    
    def calculate_overall_scores(self, project_metrics):
        """Calculate overall scores for the project"""
        scores = {}
        
        # Code Quality Score (0-100)
        quality_score = 100
        if project_metrics['code_quality']['syntax_errors']:
            quality_score -= 30
        if project_metrics['code_quality']['complexity_score'] > 15:
            quality_score -= 20
        if project_metrics['code_quality']['maintainability_issues']:
            quality_score -= len(project_metrics['code_quality']['maintainability_issues']) * 10
        if project_metrics['code_quality']['security_concerns']:
            quality_score -= len(project_metrics['code_quality']['security_concerns']) * 15
        
        scores['code_quality'] = max(0, quality_score)
        
        # Documentation Score (0-100)
        scores['documentation'] = project_metrics['documentation']['readme_quality']
        
        # Maintainability Score (0-100)
        maintainability = 100
        if project_metrics['code_quality']['avg_lines_per_file'] > 300:
            maintainability -= 20
        if project_metrics['dependencies']['dependency_health'] == 'concerning':
            maintainability -= 30
        elif project_metrics['dependencies']['dependency_health'] == 'moderate':
            maintainability -= 15
        
        scores['maintainability'] = max(0, maintainability)
        
        # Security Score (0-100)
        security = 100 - len(project_metrics['code_quality']['security_concerns']) * 20
        scores['security'] = max(0, security)
        
        # Overall Score
        scores['overall'] = (scores['code_quality'] + scores['documentation'] + 
                           scores['maintainability'] + scores['security']) / 4
        
        return scores
    
    def evaluate_all_projects(self):
        """Evaluate all discovered projects"""
        projects = self.discover_projects()
        results = {}
        
        for project in projects:
            print(f"Evaluating project: {project['name']}")
            
            project_metrics = {
                'basic_info': {
                    'name': project['name'],
                    'path': str(project['path']),
                    'file_count': len(project['python_files']),
                    'has_readme': project['readme_path'] is not None
                },
                'code_quality': self.analyze_code_quality(project),
                'dependencies': self.analyze_dependencies(project),
                'documentation': self.analyze_documentation(project)
            }
            
            project_metrics['scores'] = self.calculate_overall_scores(project_metrics)
            results[project['name']] = project_metrics
        
        return results
    
    def generate_report(self, results, output_file='EVALUATION_REPORT.md'):
        """Generate a comprehensive markdown report"""
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Project Evaluation Report\n\n")
            f.write("*Generated automatically by project_evaluator.py*\n\n")
            
            # Summary statistics
            f.write("## Executive Summary\n\n")
            total_projects = len(results)
            avg_overall_score = sum(r['scores']['overall'] for r in results.values()) / total_projects
            
            f.write(f"- **Total Projects Analyzed**: {total_projects}\n")
            f.write(f"- **Average Overall Score**: {avg_overall_score:.1f}/100\n")
            f.write(f"- **Report Generated**: {os.path.basename(output_file)}\n\n")
            
            # Top performers
            sorted_projects = sorted(results.items(), key=lambda x: x[1]['scores']['overall'], reverse=True)
            f.write("### Top Performing Projects\n\n")
            for i, (name, data) in enumerate(sorted_projects[:5], 1):
                f.write(f"{i}. **{name}** - {data['scores']['overall']:.1f}/100\n")
            f.write("\n")
            
            # Detailed analysis
            f.write("## Detailed Project Analysis\n\n")
            
            for project_name, data in sorted_projects:
                f.write(f"### {project_name}\n\n")
                
                # Basic info
                f.write("**Basic Information:**\n")
                f.write(f"- Files: {data['basic_info']['file_count']} Python files\n")
                f.write(f"- Lines of Code: {data['code_quality']['total_lines']}\n")
                f.write(f"- Average File Size: {data['code_quality']['avg_lines_per_file']:.1f} lines\n")
                f.write(f"- Has README: {'✅' if data['basic_info']['has_readme'] else '❌'}\n\n")
                
                # Scores
                f.write("**Quality Scores:**\n")
                f.write(f"- Overall: {data['scores']['overall']:.1f}/100\n")
                f.write(f"- Code Quality: {data['scores']['code_quality']:.1f}/100\n")
                f.write(f"- Documentation: {data['scores']['documentation']:.1f}/100\n")
                f.write(f"- Maintainability: {data['scores']['maintainability']:.1f}/100\n")
                f.write(f"- Security: {data['scores']['security']:.1f}/100\n\n")
                
                # Dependencies
                f.write("**Dependencies:**\n")
                f.write(f"- External Libraries: {', '.join(data['dependencies']['external_libs']) if data['dependencies']['external_libs'] else 'None'}\n")
                f.write(f"- Dependency Health: {data['dependencies']['dependency_health'].title()}\n\n")
                
                # Issues
                if data['code_quality']['syntax_errors']:
                    f.write("**Issues Found:**\n")
                    for error in data['code_quality']['syntax_errors']:
                        f.write(f"- ⚠️ {error}\n")
                
                if data['code_quality']['security_concerns']:
                    for concern in data['code_quality']['security_concerns']:
                        f.write(f"- 🔒 {concern}\n")
                
                if data['code_quality']['maintainability_issues']:
                    for issue in data['code_quality']['maintainability_issues']:
                        f.write(f"- 🔧 {issue}\n")
                
                f.write("\n---\n\n")

if __name__ == "__main__":
    evaluator = ProjectEvaluator()
    results = evaluator.evaluate_all_projects()
    evaluator.generate_report(results)
    print("Evaluation complete! Report generated as EVALUATION_REPORT.md")