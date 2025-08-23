#!/usr/bin/env python3
"""
Repository Analyzer - Comprehensive analysis tool for the projects repository
Evaluates and expands on existing project documentation with detailed metrics.
"""

import os
import json
import ast
import re
from collections import defaultdict, Counter
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple, Any

class ProjectAnalyzer:
    """Analyzes individual projects for various metrics and characteristics."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_name = self.project_path.name
        self.metrics = {}
        
    def analyze_code_metrics(self) -> Dict[str, Any]:
        """Analyze code-related metrics for the project."""
        metrics = {
            'total_files': 0,
            'python_files': 0,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'complexity_score': 0,
            'functions': 0,
            'classes': 0,
            'imports': set(),
            'external_dependencies': set()
        }
        
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file():
                metrics['total_files'] += 1
                
                if file_path.suffix == '.py':
                    metrics['python_files'] += 1
                    file_metrics = self._analyze_python_file(file_path)
                    
                    metrics['total_lines'] += file_metrics['total_lines']
                    metrics['code_lines'] += file_metrics['code_lines']
                    metrics['comment_lines'] += file_metrics['comment_lines']
                    metrics['blank_lines'] += file_metrics['blank_lines']
                    metrics['complexity_score'] += file_metrics['complexity']
                    metrics['functions'] += file_metrics['functions']
                    metrics['classes'] += file_metrics['classes']
                    metrics['imports'].update(file_metrics['imports'])
                    metrics['external_dependencies'].update(file_metrics['external_deps'])
        
        # Convert sets to lists for JSON serialization
        metrics['imports'] = list(metrics['imports'])
        metrics['external_dependencies'] = list(metrics['external_dependencies'])
        
        return metrics
    
    def _analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for metrics."""
        metrics = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'complexity': 0,
            'functions': 0,
            'classes': 0,
            'imports': set(),
            'external_deps': set()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            metrics['total_lines'] = len(lines)
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    metrics['blank_lines'] += 1
                elif stripped.startswith('#'):
                    metrics['comment_lines'] += 1
                else:
                    metrics['code_lines'] += 1
            
            # AST analysis for complexity and structure
            try:
                tree = ast.parse(content)
                metrics.update(self._analyze_ast(tree))
            except SyntaxError:
                pass  # Skip files with syntax errors
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return metrics
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST for structural metrics."""
        metrics = {
            'complexity': 0,
            'functions': 0,
            'classes': 0,
            'imports': set(),
            'external_deps': set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                metrics['functions'] += 1
                metrics['complexity'] += self._calculate_complexity(node)
            elif isinstance(node, ast.ClassDef):
                metrics['classes'] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    metrics['imports'].add(module_name)
                    if self._is_external_dependency(module_name):
                        metrics['external_deps'].add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    metrics['imports'].add(module_name)
                    if self._is_external_dependency(module_name):
                        metrics['external_deps'].add(module_name)
        
        return metrics
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def _is_external_dependency(self, module_name: str) -> bool:
        """Check if a module is an external dependency."""
        standard_libs = {
            'os', 'sys', 'json', 'ast', 're', 'collections', 'pathlib',
            'subprocess', 'typing', 'datetime', 'time', 'random', 'math',
            'itertools', 'functools', 'operator', 'copy', 'pickle', 'sqlite3',
            'csv', 'xml', 'html', 'urllib', 'http', 'email', 'hashlib',
            'base64', 'uuid', 'logging', 'argparse', 'configparser', 'io',
            'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache', 'textwrap'
        }
        return module_name not in standard_libs and module_name != self.project_name
    
    def analyze_documentation(self) -> Dict[str, Any]:
        """Analyze documentation quality and completeness."""
        docs = {
            'has_readme': False,
            'readme_length': 0,
            'has_docstrings': False,
            'docstring_coverage': 0.0,
            'documentation_files': []
        }
        
        # Check for README
        readme_files = list(self.project_path.glob('README*'))
        if readme_files:
            docs['has_readme'] = True
            readme_path = readme_files[0]
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    docs['readme_length'] = len(f.read().split())
            except Exception:
                pass
        
        # Check for other documentation files
        doc_extensions = {'.md', '.rst', '.txt', '.doc'}
        for file_path in self.project_path.rglob('*'):
            if file_path.suffix.lower() in doc_extensions:
                docs['documentation_files'].append(str(file_path.relative_to(self.project_path)))
        
        # Analyze docstring coverage
        docs.update(self._analyze_docstring_coverage())
        
        return docs
    
    def _analyze_docstring_coverage(self) -> Dict[str, Any]:
        """Analyze docstring coverage in Python files."""
        total_functions = 0
        documented_functions = 0
        
        for py_file in self.project_path.glob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                            
            except Exception:
                continue
        
        coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
        
        return {
            'has_docstrings': documented_functions > 0,
            'docstring_coverage': round(coverage, 2),
            'total_documentable': total_functions,
            'documented_items': documented_functions
        }
    
    def analyze_project_health(self) -> Dict[str, Any]:
        """Analyze overall project health indicators."""
        health = {
            'maintainability_score': 0,
            'complexity_rating': 'Low',
            'documentation_rating': 'Poor',
            'size_category': 'Small',
            'estimated_effort_hours': 0
        }
        
        code_metrics = self.analyze_code_metrics()
        doc_metrics = self.analyze_documentation()
        
        # Calculate maintainability score (0-100)
        complexity_factor = min(code_metrics['complexity_score'] / max(code_metrics['functions'], 1), 10) / 10
        doc_factor = doc_metrics['docstring_coverage'] / 100
        size_factor = min(code_metrics['code_lines'] / 1000, 1)
        
        health['maintainability_score'] = round((1 - complexity_factor) * 40 + doc_factor * 30 + (1 - size_factor) * 30)
        
        # Complexity rating
        avg_complexity = code_metrics['complexity_score'] / max(code_metrics['functions'], 1)
        if avg_complexity < 2:
            health['complexity_rating'] = 'Low'
        elif avg_complexity < 4:
            health['complexity_rating'] = 'Medium'
        else:
            health['complexity_rating'] = 'High'
        
        # Documentation rating
        if doc_metrics['docstring_coverage'] > 80:
            health['documentation_rating'] = 'Excellent'
        elif doc_metrics['docstring_coverage'] > 60:
            health['documentation_rating'] = 'Good'
        elif doc_metrics['docstring_coverage'] > 30:
            health['documentation_rating'] = 'Fair'
        else:
            health['documentation_rating'] = 'Poor'
        
        # Size category
        if code_metrics['code_lines'] < 100:
            health['size_category'] = 'Small'
        elif code_metrics['code_lines'] < 500:
            health['size_category'] = 'Medium'
        elif code_metrics['code_lines'] < 1000:
            health['size_category'] = 'Large'
        else:
            health['size_category'] = 'Very Large'
        
        # Estimated effort in hours (rough calculation)
        health['estimated_effort_hours'] = round(code_metrics['code_lines'] / 20 + code_metrics['complexity_score'] * 0.5)
        
        return health

class RepositoryAnalyzer:
    """Analyzes the entire repository for comprehensive insights."""
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path)
        self.projects = []
        self.analysis_results = {}
        
    def discover_projects(self) -> List[str]:
        """Discover all projects in the repository."""
        projects = []
        
        for item in self.repo_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it's a project directory (contains Python files or README)
                has_python = any(item.glob('*.py'))
                has_readme = any(item.glob('README*'))
                
                if has_python or has_readme:
                    projects.append(item.name)
        
        return sorted(projects)
    
    def analyze_all_projects(self) -> Dict[str, Any]:
        """Analyze all projects in the repository."""
        projects = self.discover_projects()
        results = {}
        
        for project_name in projects:
            print(f"Analyzing project: {project_name}")
            analyzer = ProjectAnalyzer(self.repo_path / project_name)
            
            results[project_name] = {
                'code_metrics': analyzer.analyze_code_metrics(),
                'documentation': analyzer.analyze_documentation(),
                'health': analyzer.analyze_project_health()
            }
        
        self.analysis_results = results
        return results
    
    def generate_repository_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive repository summary."""
        if not self.analysis_results:
            self.analyze_all_projects()
        
        summary = {
            'total_projects': len(self.analysis_results),
            'total_python_files': 0,
            'total_lines_of_code': 0,
            'total_functions': 0,
            'total_classes': 0,
            'average_complexity': 0,
            'documentation_coverage': 0,
            'external_dependencies': set(),
            'project_categories': defaultdict(list),
            'complexity_distribution': defaultdict(int),
            'size_distribution': defaultdict(int),
            'top_projects_by_size': [],
            'top_projects_by_complexity': [],
            'most_documented_projects': [],
            'technology_stack': Counter()
        }
        
        total_complexity = 0
        total_documented = 0
        total_documentable = 0
        
        for project_name, analysis in self.analysis_results.items():
            code = analysis['code_metrics']
            docs = analysis['documentation']
            health = analysis['health']
            
            summary['total_python_files'] += code['python_files']
            summary['total_lines_of_code'] += code['code_lines']
            summary['total_functions'] += code['functions']
            summary['total_classes'] += code['classes']
            total_complexity += code['complexity_score']
            total_documented += docs['documented_items']
            total_documentable += docs['total_documentable']
            
            summary['external_dependencies'].update(code['external_dependencies'])
            summary['complexity_distribution'][health['complexity_rating']] += 1
            summary['size_distribution'][health['size_category']] += 1
            summary['technology_stack'].update(code['external_dependencies'])
        
        # Calculate averages and top projects
        if self.analysis_results:
            summary['average_complexity'] = round(total_complexity / summary['total_functions'] if summary['total_functions'] > 0 else 0, 2)
            summary['documentation_coverage'] = round(total_documented / total_documentable * 100 if total_documentable > 0 else 0, 2)
        
        # Convert sets to lists
        summary['external_dependencies'] = list(summary['external_dependencies'])
        summary['project_categories'] = dict(summary['project_categories'])
        summary['complexity_distribution'] = dict(summary['complexity_distribution'])
        summary['size_distribution'] = dict(summary['size_distribution'])
        
        # Top projects
        projects_by_size = sorted(self.analysis_results.items(), 
                                key=lambda x: x[1]['code_metrics']['code_lines'], reverse=True)
        summary['top_projects_by_size'] = [(name, data['code_metrics']['code_lines']) for name, data in projects_by_size[:5]]
        
        projects_by_complexity = sorted(self.analysis_results.items(), 
                                      key=lambda x: x[1]['code_metrics']['complexity_score'], reverse=True)
        summary['top_projects_by_complexity'] = [(name, data['code_metrics']['complexity_score']) for name, data in projects_by_complexity[:5]]
        
        projects_by_docs = sorted(self.analysis_results.items(), 
                                key=lambda x: x[1]['documentation']['docstring_coverage'], reverse=True)
        summary['most_documented_projects'] = [(name, data['documentation']['docstring_coverage']) for name, data in projects_by_docs[:5]]
        
        return summary
    
    def save_analysis(self, output_file: str = 'repository_analysis.json'):
        """Save analysis results to a JSON file."""
        if not self.analysis_results:
            self.analyze_all_projects()
        
        output_data = {
            'repository_summary': self.generate_repository_summary(),
            'project_details': self.analysis_results,
            'analysis_timestamp': subprocess.check_output(['date']).decode().strip()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Analysis saved to {output_file}")

def main():
    """Main function to run repository analysis."""
    analyzer = RepositoryAnalyzer()
    
    print("🔍 Analyzing repository structure and projects...")
    results = analyzer.analyze_all_projects()
    
    print("📊 Generating repository summary...")
    summary = analyzer.generate_repository_summary()
    
    print("💾 Saving analysis results...")
    analyzer.save_analysis()
    
    print("\n📋 Repository Analysis Summary:")
    print(f"Total Projects: {summary['total_projects']}")
    print(f"Total Python Files: {summary['total_python_files']}")
    print(f"Total Lines of Code: {summary['total_lines_of_code']:,}")
    print(f"Total Functions: {summary['total_functions']}")
    print(f"Total Classes: {summary['total_classes']}")
    print(f"Average Function Complexity: {summary['average_complexity']}")
    print(f"Documentation Coverage: {summary['documentation_coverage']}%")
    print(f"External Dependencies: {len(summary['external_dependencies'])}")
    
    print("\n🏆 Top Projects by Size:")
    for name, loc in summary['top_projects_by_size']:
        print(f"  {name}: {loc:,} lines")
    
    print("\n🔧 Technology Stack:")
    for tech, count in summary['technology_stack'].most_common(10):
        print(f"  {tech}: {count} projects")

if __name__ == "__main__":
    main()