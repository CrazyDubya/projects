#!/usr/bin/env python3
"""
Project Integration Example: Repository Analysis Pipeline

This script demonstrates how multiple projects in the repository can work together
to create a comprehensive analysis pipeline. It combines:
- allseeingeye: Directory structure analysis
- jsonreader: Processing configuration and results
- shared_utils: Common functionality

This serves as an example of project integration and demonstrates the value
of standardized interfaces and shared utilities.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from shared_utils import (
    ProjectConfig, setup_logging, safe_file_operation, 
    ensure_directory, timestamp_filename
)

# Configuration
CONFIG = ProjectConfig()
LOGGER = setup_logging(level=CONFIG.get('log_level', 'INFO'))


class RepositoryAnalyzer:
    """
    Comprehensive repository analysis using multiple integrated tools.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize the analyzer.
        
        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path)
        self.output_dir = ensure_directory(CONFIG.get('output_dir', './integration_output'))
        self.analysis_data = {}
        
        LOGGER.info(f"Initialized RepositoryAnalyzer for: {self.repo_path}")
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Analyze repository structure using allseeingeye functionality.
        
        Returns:
            Dictionary containing structure analysis results
        """
        LOGGER.info("Analyzing repository structure...")
        
        structure_data = {
            'timestamp': datetime.now().isoformat(),
            'repo_path': str(self.repo_path),
            'directories': [],
            'files': [],
            'file_types': {},
            'total_size': 0
        }
        
        try:
            # Walk through repository structure
            for item in self.repo_path.rglob('*'):
                if item.is_file():
                    # Analyze file
                    file_info = {
                        'path': str(item.relative_to(self.repo_path)),
                        'name': item.name,
                        'extension': item.suffix.lower(),
                        'size': item.stat().st_size,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    structure_data['files'].append(file_info)
                    structure_data['total_size'] += file_info['size']
                    
                    # Count file types
                    ext = file_info['extension'] or 'no_extension'
                    structure_data['file_types'][ext] = structure_data['file_types'].get(ext, 0) + 1
                    
                elif item.is_dir():
                    # Analyze directory
                    dir_info = {
                        'path': str(item.relative_to(self.repo_path)),
                        'name': item.name,
                        'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    }
                    structure_data['directories'].append(dir_info)
            
            # Generate summary statistics
            structure_data['summary'] = {
                'total_files': len(structure_data['files']),
                'total_directories': len(structure_data['directories']),
                'total_size_mb': round(structure_data['total_size'] / (1024 * 1024), 2),
                'most_common_file_type': max(structure_data['file_types'], 
                                           key=structure_data['file_types'].get) if structure_data['file_types'] else None
            }
            
            LOGGER.info(f"Structure analysis completed: {structure_data['summary']['total_files']} files, "
                       f"{structure_data['summary']['total_directories']} directories")
            
            return structure_data
            
        except Exception as e:
            LOGGER.error(f"Structure analysis failed: {e}")
            return structure_data
    
    def analyze_projects(self) -> Dict[str, Any]:
        """
        Analyze individual projects in the repository.
        
        Returns:
            Dictionary containing project analysis results
        """
        LOGGER.info("Analyzing individual projects...")
        
        project_data = {
            'timestamp': datetime.now().isoformat(),
            'projects': [],
            'categories': {},
            'technologies': set(),
            'total_loc': 0
        }
        
        try:
            # Look for project directories (containing Python files or README)
            potential_projects = []
            for item in self.repo_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if directory contains Python files or README
                    python_files = list(item.glob('*.py'))
                    readme_files = list(item.glob('README.md')) + list(item.glob('readme.md'))
                    
                    if python_files or readme_files:
                        potential_projects.append(item)
            
            # Analyze each project
            for project_dir in potential_projects:
                project_info = self._analyze_single_project(project_dir)
                if project_info:
                    project_data['projects'].append(project_info)
                    
                    # Update technologies
                    project_data['technologies'].update(project_info.get('technologies', []))
                    
                    # Update categories
                    category = project_info.get('category', 'unknown')
                    project_data['categories'][category] = project_data['categories'].get(category, 0) + 1
                    
                    # Update total lines of code
                    project_data['total_loc'] += project_info.get('lines_of_code', 0)
            
            # Convert set to list for JSON serialization
            project_data['technologies'] = list(project_data['technologies'])
            
            LOGGER.info(f"Project analysis completed: {len(project_data['projects'])} projects found")
            
            return project_data
            
        except Exception as e:
            LOGGER.error(f"Project analysis failed: {e}")
            return project_data
    
    def _analyze_single_project(self, project_dir: Path) -> Dict[str, Any]:
        """
        Analyze a single project directory.
        
        Args:
            project_dir: Path to the project directory
            
        Returns:
            Dictionary containing project analysis
        """
        try:
            project_info = {
                'name': project_dir.name,
                'path': str(project_dir.relative_to(self.repo_path)),
                'python_files': [],
                'lines_of_code': 0,
                'technologies': [],
                'category': 'utility'  # default category
            }
            
            # Analyze Python files
            for py_file in project_dir.glob('*.py'):
                file_content = safe_file_operation('read', py_file)
                if file_content:
                    lines = len(file_content.splitlines())
                    project_info['python_files'].append({
                        'name': py_file.name,
                        'lines': lines
                    })
                    project_info['lines_of_code'] += lines
                    
                    # Detect technologies from imports
                    self._detect_technologies(file_content, project_info['technologies'])
            
            # Categorize project based on patterns
            project_info['category'] = self._categorize_project(project_info)
            
            # Read README if available
            readme_content = None
            for readme_name in ['README.md', 'readme.md']:
                readme_path = project_dir / readme_name
                if readme_path.exists():
                    readme_content = safe_file_operation('read', readme_path)
                    break
            
            if readme_content:
                project_info['has_readme'] = True
                project_info['readme_length'] = len(readme_content)
            else:
                project_info['has_readme'] = False
            
            return project_info
            
        except Exception as e:
            LOGGER.error(f"Failed to analyze project {project_dir}: {e}")
            return None
    
    def _detect_technologies(self, content: str, technologies: List[str]) -> None:
        """
        Detect technologies used based on import statements.
        
        Args:
            content: File content to analyze
            technologies: List to append detected technologies
        """
        import_patterns = {
            'flask': ['flask'],
            'anthropic': ['anthropic'],
            'openai': ['openai'],
            'pytorch': ['torch'],
            'transformers': ['transformers'],
            'gui': ['tkinter', 'PyQt5'],
            'web': ['requests', 'urllib'],
            'data': ['pandas', 'numpy'],
            'nltk': ['nltk'],
            'rich': ['rich'],
            'bluetooth': ['bleak'],
            'monitoring': ['prometheus']
        }
        
        for tech, patterns in import_patterns.items():
            if any(pattern in content for pattern in patterns):
                if tech not in technologies:
                    technologies.append(tech)
    
    def _categorize_project(self, project_info: Dict[str, Any]) -> str:
        """
        Categorize project based on its characteristics.
        
        Args:
            project_info: Project information dictionary
            
        Returns:
            Category string
        """
        name = project_info['name'].lower()
        technologies = project_info.get('technologies', [])
        
        # AI/LLM projects
        if any(tech in technologies for tech in ['anthropic', 'openai', 'transformers']):
            return 'ai_llm'
        
        # Web projects
        if 'flask' in technologies or 'web' in technologies:
            return 'web'
        
        # GUI projects
        if 'gui' in technologies:
            return 'gui'
        
        # Games
        if any(keyword in name for keyword in ['game', 'chess', '4x', 'nomic']):
            return 'game'
        
        # File processing
        if any(keyword in name for keyword in ['reader', 'merge', 'convert', 'file']):
            return 'file_processing'
        
        # System automation
        if any(keyword in name for keyword in ['mover', 'auto', 'watch', 'bluetooth']):
            return 'automation'
        
        return 'utility'
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Complete analysis report
        """
        LOGGER.info("Generating comprehensive analysis report...")
        
        # Run all analyses
        structure_analysis = self.analyze_structure()
        project_analysis = self.analyze_projects()
        
        # Combine results
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'repository_path': str(self.repo_path)
            },
            'structure_analysis': structure_analysis,
            'project_analysis': project_analysis,
            'summary': {
                'total_projects': len(project_analysis['projects']),
                'total_files': structure_analysis['summary']['total_files'],
                'total_size_mb': structure_analysis['summary']['total_size_mb'],
                'total_lines_of_code': project_analysis['total_loc'],
                'technologies_used': project_analysis['technologies'],
                'project_categories': project_analysis['categories']
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """
        Save analysis report to file.
        
        Args:
            report: Report data to save
            
        Returns:
            Path to saved report file
        """
        report_filename = timestamp_filename('repository_analysis', '.json')
        report_path = self.output_dir / report_filename
        
        # Save JSON report
        success = safe_file_operation('write', report_path, report)
        if success:
            LOGGER.info(f"Report saved to: {report_path}")
            
            # Also save a human-readable summary
            summary_path = self.output_dir / timestamp_filename('repository_summary', '.txt')
            self._save_human_readable_summary(report, summary_path)
            
            return report_path
        else:
            LOGGER.error("Failed to save report")
            return None
    
    def _save_human_readable_summary(self, report: Dict[str, Any], summary_path: Path) -> None:
        """
        Save human-readable summary of the analysis.
        
        Args:
            report: Report data
            summary_path: Path to save summary
        """
        try:
            summary_lines = [
                "# Repository Analysis Summary",
                f"Generated: {report['analysis_metadata']['timestamp']}",
                f"Repository: {report['analysis_metadata']['repository_path']}",
                "",
                "## Overview",
                f"- Total Projects: {report['summary']['total_projects']}",
                f"- Total Files: {report['summary']['total_files']}",
                f"- Repository Size: {report['summary']['total_size_mb']} MB",
                f"- Total Lines of Code: {report['summary']['total_lines_of_code']}",
                "",
                "## Technologies Used",
            ]
            
            for tech in sorted(report['summary']['technologies_used']):
                summary_lines.append(f"- {tech}")
            
            summary_lines.extend([
                "",
                "## Project Categories",
            ])
            
            for category, count in report['summary']['project_categories'].items():
                summary_lines.append(f"- {category}: {count} projects")
            
            summary_lines.extend([
                "",
                "## Top Projects by Size",
            ])
            
            # Sort projects by lines of code
            projects = report['project_analysis']['projects']
            top_projects = sorted(projects, key=lambda x: x.get('lines_of_code', 0), reverse=True)[:5]
            
            for project in top_projects:
                summary_lines.append(f"- {project['name']}: {project.get('lines_of_code', 0)} LOC")
            
            summary_content = "\n".join(summary_lines)
            safe_file_operation('write', summary_path, summary_content)
            LOGGER.info(f"Human-readable summary saved to: {summary_path}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save human-readable summary: {e}")


def main():
    """
    Main function demonstrating integrated repository analysis.
    """
    LOGGER.info("Starting integrated repository analysis pipeline")
    
    # Initialize analyzer for current repository
    repo_path = Path(__file__).parent
    analyzer = RepositoryAnalyzer(repo_path)
    
    try:
        # Generate comprehensive report
        report = analyzer.generate_report()
        
        # Save report
        report_path = analyzer.save_report(report)
        
        if report_path:
            print("\n" + "="*60)
            print("REPOSITORY ANALYSIS COMPLETED")
            print("="*60)
            print(f"Total Projects: {report['summary']['total_projects']}")
            print(f"Total Files: {report['summary']['total_files']}")
            print(f"Repository Size: {report['summary']['total_size_mb']} MB")
            print(f"Total Lines of Code: {report['summary']['total_lines_of_code']}")
            print(f"Technologies: {', '.join(sorted(report['summary']['technologies_used']))}")
            print(f"\nDetailed report saved to: {report_path}")
            print(f"Output directory: {analyzer.output_dir}")
            print("="*60)
        else:
            print("Analysis completed but failed to save report")
            
    except Exception as e:
        LOGGER.error(f"Analysis pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()