#!/usr/bin/env python3
"""
Enhanced Documentation Generator
Expands existing repository documentation with comprehensive analysis and insights.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class DocumentationExpander:
    """Expands existing documentation with comprehensive analysis data."""
    
    def __init__(self, analysis_file: str = 'repository_analysis.json'):
        with open(analysis_file, 'r') as f:
            self.analysis_data = json.load(f)
        self.repo_summary = self.analysis_data['repository_summary']
        self.project_details = self.analysis_data['project_details']
    
    def generate_enhanced_index(self) -> str:
        """Generate an enhanced INDEX.md with comprehensive metrics."""
        content = f"""# Projects Portfolio Index - Enhanced Analysis

## Repository Overview
This repository contains a diverse collection of **{self.repo_summary['total_projects']} Python-based projects** ranging from AI/LLM tools to automation utilities, games, and file management systems. Each project is designed as a standalone utility or application with specific capabilities and use cases.

### 📊 Repository Metrics
- **Total Projects**: {self.repo_summary['total_projects']}
- **Python Files**: {self.repo_summary['total_python_files']}
- **Lines of Code**: {self.repo_summary['total_lines_of_code']:,}
- **Functions**: {self.repo_summary['total_functions']}
- **Classes**: {self.repo_summary['total_classes']}
- **Average Function Complexity**: {self.repo_summary['average_complexity']}
- **Documentation Coverage**: {self.repo_summary['documentation_coverage']:.1f}%
- **External Dependencies**: {len(self.repo_summary['external_dependencies'])}

### 🎯 Project Distribution by Complexity
"""
        
        for complexity, count in self.repo_summary['complexity_distribution'].items():
            percentage = (count / self.repo_summary['total_projects']) * 100
            content += f"- **{complexity}**: {count} projects ({percentage:.1f}%)\n"
        
        content += "\n### 📏 Project Distribution by Size\n"
        for size, count in self.repo_summary['size_distribution'].items():
            percentage = (count / self.repo_summary['total_projects']) * 100
            content += f"- **{size}**: {count} projects ({percentage:.1f}%)\n"
        
        content += f"""

## Project Categories

{self._generate_categorized_project_tables()}

## 🏆 Top Performing Projects

### 🚀 Largest Projects (Lines of Code)
"""
        
        for i, (project, lines) in enumerate(self.repo_summary['top_projects_by_size'], 1):
            health = self.project_details[project]['health']
            content += f"{i}. **{project}** - {lines:,} lines (Complexity: {health['complexity_rating']}, Maintainability: {health['maintainability_score']}/100)\n"
        
        content += "\n### 🧠 Most Complex Projects\n"
        for i, (project, complexity) in enumerate(self.repo_summary['top_projects_by_complexity'], 1):
            health = self.project_details[project]['health']
            content += f"{i}. **{project}** - Complexity Score: {complexity} (Size: {health['size_category']})\n"
        
        content += "\n### 📚 Best Documented Projects\n"
        for i, (project, coverage) in enumerate(self.repo_summary['most_documented_projects'], 1):
            if coverage > 0:
                content += f"{i}. **{project}** - {coverage:.1f}% documentation coverage\n"
        
        content += f"""

## 🔧 Technology Stack Analysis

### Core Dependencies
The repository leverages {len(self.repo_summary['external_dependencies'])} external dependencies:

#### AI/ML Libraries
- **anthropic**: Used in {self._count_dependency_usage('anthropic')} projects - Claude API integration
- **openai**: Used in {self._count_dependency_usage('openai')} projects - GPT model integration  
- **transformers**: Advanced NLP capabilities
- **torch**: Deep learning framework
- **sklearn**: Machine learning utilities

#### GUI/Interface
- **tkinter**: Desktop GUI applications
- **PyQt5**: Advanced desktop applications
- **rich**: Enhanced console output

#### Data Processing
- **lxml**: XML processing
- **networkx**: Graph analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization

#### Specialized Tools
- **bleak**: Bluetooth communication
- **watchdog**: File system monitoring
- **wordcloud**: Text visualization
- **prometheus_client**: Metrics collection

## 📈 Repository Health Assessment

### Overall Health Score: {self._calculate_overall_health()}/100

#### Strengths:
- Diverse project portfolio covering multiple domains
- Good distribution of project complexities
- Strong use of modern Python libraries
- Comprehensive project categorization

#### Areas for Improvement:
- Documentation coverage is low at {self.repo_summary['documentation_coverage']:.1f}%
- {self._count_high_complexity_projects()} projects have high complexity
- Missing centralized dependency management
- No automated testing infrastructure visible

## 🎯 Commercial Viability Analysis

{self._generate_commercial_analysis()}

## 🔗 Project Interconnections

{self._generate_interconnection_analysis()}

## 📋 Detailed Project Metrics

For comprehensive project-by-project analysis including code metrics, documentation quality, and health indicators, see the generated `repository_analysis.json` file.

---

*This enhanced analysis was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using automated repository analysis tools.*
"""
        
        return content
    
    def _generate_categorized_project_tables(self) -> str:
        """Generate project tables organized by categories."""
        # Categorize projects based on their characteristics
        categories = {
            'AI & Language Model Tools': [],
            'Games & Simulations': [],
            'File Management & Processing': [],
            'Automation & Utilities': [],
            'Communication Tools': [],
            'Data & Analytics': []
        }
        
        # AI/ML projects (based on dependencies and naming)
        ai_indicators = ['anthropic', 'openai', 'transformers', 'torch', 'sklearn']
        
        # Game projects
        game_names = ['4x', 'Quantum_Chess', 'nomic']
        
        # File processing projects
        file_processing = ['allseeingeye', 'mover', 'HeaderPy', 'MDtoHTML', 'MakeMarkdown', 'xmlmerge', 'jsonreader']
        
        for project_name, details in self.project_details.items():
            deps = details['code_metrics']['external_dependencies']
            code_metrics = details['code_metrics']
            health = details['health']
            docs = details['documentation']
            
            # Determine category
            if any(ai_dep in deps for ai_dep in ai_indicators) or 'chat' in project_name.lower() or 'llm' in project_name.lower():
                categories['AI & Language Model Tools'].append(project_name)
            elif project_name in game_names:
                categories['Games & Simulations'].append(project_name)
            elif project_name in file_processing:
                categories['File Management & Processing'].append(project_name)
            elif 'communication' in project_name.lower() or 'chat' in project_name.lower() or 'bluetooth' in project_name.lower():
                categories['Communication Tools'].append(project_name)
            elif project_name in ['movelog']:
                categories['Data & Analytics'].append(project_name)
            else:
                categories['Automation & Utilities'].append(project_name)
        
        content = ""
        for category, projects in categories.items():
            if projects:
                content += f"### 🔧 {category}\n\n"
                content += "| Project | Size | Complexity | Documentation | Maintainability | Key Dependencies |\n"
                content += "|---------|------|------------|---------------|-----------------|------------------|\n"
                
                for project in projects:
                    details = self.project_details[project]
                    code = details['code_metrics']
                    health = details['health']
                    docs = details['documentation']
                    
                    size_info = f"{code['code_lines']} LOC"
                    complexity = health['complexity_rating']
                    doc_score = f"{docs['docstring_coverage']:.1f}%"
                    maintainability = f"{health['maintainability_score']}/100"
                    key_deps = ', '.join(code['external_dependencies'][:3]) if code['external_dependencies'] else 'None'
                    
                    content += f"| **{project}** | {size_info} | {complexity} | {doc_score} | {maintainability} | {key_deps} |\n"
                
                content += "\n"
        
        return content
    
    def _count_dependency_usage(self, dependency: str) -> int:
        """Count how many projects use a specific dependency."""
        count = 0
        for project_details in self.project_details.values():
            if dependency in project_details['code_metrics']['external_dependencies']:
                count += 1
        return count
    
    def _calculate_overall_health(self) -> int:
        """Calculate overall repository health score."""
        total_score = 0
        for project_details in self.project_details.values():
            total_score += project_details['health']['maintainability_score']
        
        return round(total_score / len(self.project_details))
    
    def _count_high_complexity_projects(self) -> int:
        """Count projects with high complexity."""
        return self.repo_summary['complexity_distribution'].get('High', 0)
    
    def _generate_commercial_analysis(self) -> str:
        """Generate commercial viability analysis."""
        content = """### Market-Ready Projects
Based on code quality, documentation, and complexity analysis:

#### 🟢 Ready for Commercial Use
"""
        
        ready_projects = []
        potential_projects = []
        needs_work_projects = []
        
        for project_name, details in self.project_details.items():
            health = details['health']
            docs = details['documentation']
            code = details['code_metrics']
            
            score = health['maintainability_score']
            doc_coverage = docs['docstring_coverage']
            complexity = health['complexity_rating']
            
            if score >= 70 and doc_coverage >= 50:
                ready_projects.append((project_name, score, doc_coverage))
            elif score >= 50 or code['code_lines'] > 200:
                potential_projects.append((project_name, score, doc_coverage))
            else:
                needs_work_projects.append((project_name, score, doc_coverage))
        
        for project, score, doc_coverage in sorted(ready_projects, key=lambda x: x[1], reverse=True):
            content += f"- **{project}**: Maintainability {score}/100, Documentation {doc_coverage:.1f}%\n"
        
        content += "\n#### 🟡 Commercial Potential (Needs Enhancement)\n"
        for project, score, doc_coverage in sorted(potential_projects, key=lambda x: x[1], reverse=True):
            content += f"- **{project}**: Maintainability {score}/100, Documentation {doc_coverage:.1f}%\n"
        
        content += "\n#### 🔴 Requires Significant Development\n"
        for project, score, doc_coverage in sorted(needs_work_projects, key=lambda x: x[1], reverse=True):
            content += f"- **{project}**: Maintainability {score}/100, Documentation {doc_coverage:.1f}%\n"
        
        return content
    
    def _generate_interconnection_analysis(self) -> str:
        """Analyze project interconnections and dependencies."""
        content = """### Shared Dependencies Analysis

Projects sharing common dependencies suggest potential for integration or code reuse:

"""
        
        # Find shared dependencies
        dependency_projects = {}
        for project_name, details in self.project_details.items():
            for dep in details['code_metrics']['external_dependencies']:
                if dep not in dependency_projects:
                    dependency_projects[dep] = []
                dependency_projects[dep].append(project_name)
        
        # Show dependencies used by multiple projects
        shared_deps = {dep: projects for dep, projects in dependency_projects.items() if len(projects) > 1}
        
        for dep, projects in sorted(shared_deps.items(), key=lambda x: len(x[1]), reverse=True):
            if len(projects) > 2:  # Only show dependencies used by 3+ projects
                content += f"#### {dep}\nUsed by **{len(projects)} projects**: {', '.join(projects)}\n\n"
        
        content += """### Integration Opportunities

Based on shared dependencies and functionality overlap:

1. **AI/ML Integration Hub**: Projects using anthropic/openai could share API wrappers
2. **File Processing Pipeline**: Directory and file utilities could be combined
3. **GUI Framework**: Projects with tkinter/PyQt5 could share UI components
4. **Data Analysis Suite**: Projects with visualization capabilities could be integrated

"""
        
        return content
    
    def generate_project_health_dashboard(self) -> str:
        """Generate a comprehensive project health dashboard."""
        content = f"""# Repository Health Dashboard

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Repository Health: {self._calculate_overall_health()}/100

### 🎯 Key Metrics Summary
- **Total Projects**: {self.repo_summary['total_projects']}
- **Codebase Size**: {self.repo_summary['total_lines_of_code']:,} lines across {self.repo_summary['total_python_files']} files
- **Functional Complexity**: {self.repo_summary['total_functions']} functions, {self.repo_summary['total_classes']} classes
- **Average Complexity**: {self.repo_summary['average_complexity']}/function
- **Documentation Coverage**: {self.repo_summary['documentation_coverage']:.1f}%
- **Technology Diversity**: {len(self.repo_summary['external_dependencies'])} external dependencies

## 📈 Project Health Breakdown

| Project | Size | Complexity | Maintainability | Documentation | Health Grade |
|---------|------|------------|-----------------|---------------|--------------|
"""
        
        # Sort projects by health score
        projects_by_health = sorted(
            self.project_details.items(),
            key=lambda x: x[1]['health']['maintainability_score'],
            reverse=True
        )
        
        for project_name, details in projects_by_health:
            health = details['health']
            docs = details['documentation']
            code = details['code_metrics']
            
            # Assign health grade
            score = health['maintainability_score']
            if score >= 80:
                grade = "A"
            elif score >= 70:
                grade = "B"
            elif score >= 60:
                grade = "C"
            elif score >= 50:
                grade = "D"
            else:
                grade = "F"
            
            content += f"| {project_name} | {code['code_lines']} LOC | {health['complexity_rating']} | {score}/100 | {docs['docstring_coverage']:.1f}% | {grade} |\n"
        
        content += f"""

## 🔧 Improvement Recommendations

### High Priority
1. **Documentation Enhancement**: Increase documentation coverage from {self.repo_summary['documentation_coverage']:.1f}% to >60%
2. **Dependency Management**: Create centralized requirements.txt or pyproject.toml
3. **Testing Infrastructure**: Add automated testing for critical projects

### Medium Priority  
1. **Code Complexity Reduction**: Refactor {self._count_high_complexity_projects()} high-complexity projects
2. **API Standardization**: Harmonize similar functionality across projects
3. **Performance Optimization**: Profile and optimize largest projects

### Low Priority
1. **Code Style Consistency**: Implement automated formatting
2. **Monitoring Setup**: Add health monitoring for production-ready projects
3. **Integration Planning**: Develop cross-project integration strategy

## 🏆 Success Stories

### Top Performers by Category:
"""
        
        # Show top performer in each category
        categories = ['Large', 'Medium', 'Small']
        for category in categories:
            category_projects = {name: details for name, details in self.project_details.items() 
                               if details['health']['size_category'] == category}
            
            if category_projects:
                best_project = max(category_projects.items(), 
                                 key=lambda x: x[1]['health']['maintainability_score'])
                project_name, details = best_project
                score = details['health']['maintainability_score']
                content += f"- **{category} Project Champion**: {project_name} (Maintainability: {score}/100)\n"
        
        content += f"""

---
*Dashboard generated by automated repository analysis. For detailed metrics, see `repository_analysis.json`*
"""
        
        return content
    
    def save_enhanced_documentation(self):
        """Save all enhanced documentation files."""
        # Generate enhanced INDEX.md
        enhanced_index = self.generate_enhanced_index()
        with open('INDEX_ENHANCED.md', 'w') as f:
            f.write(enhanced_index)
        
        # Generate health dashboard
        health_dashboard = self.generate_project_health_dashboard()
        with open('REPOSITORY_HEALTH.md', 'w') as f:
            f.write(health_dashboard)
        
        print("📝 Enhanced documentation generated:")
        print("  - INDEX_ENHANCED.md: Comprehensive repository overview")
        print("  - REPOSITORY_HEALTH.md: Project health dashboard")

def main():
    """Main function to generate enhanced documentation."""
    if not os.path.exists('repository_analysis.json'):
        print("❌ repository_analysis.json not found. Please run repository_analyzer.py first.")
        return
    
    print("📚 Generating enhanced documentation...")
    expander = DocumentationExpander()
    expander.save_enhanced_documentation()
    
    print("✅ Documentation expansion complete!")

if __name__ == "__main__":
    main()