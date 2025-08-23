#!/usr/bin/env python3
"""
Cross-Project Integration Analyzer
Identifies integration opportunities and dependencies between projects.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

class IntegrationAnalyzer:
    """Analyzes potential integrations between projects."""
    
    def __init__(self, analysis_file: str = 'repository_analysis.json'):
        with open(analysis_file, 'r') as f:
            self.analysis_data = json.load(f)
        self.project_details = self.analysis_data['project_details']
    
    def find_shared_dependencies(self) -> Dict[str, List[str]]:
        """Find dependencies shared between multiple projects."""
        dependency_projects = defaultdict(list)
        
        for project_name, details in self.project_details.items():
            deps = details['code_metrics']['external_dependencies']
            for dep in deps:
                dependency_projects[dep].append(project_name)
        
        # Only return dependencies used by multiple projects
        return {dep: projects for dep, projects in dependency_projects.items() 
                if len(projects) > 1}
    
    def analyze_functional_similarity(self) -> Dict[str, List[str]]:
        """Group projects by functional similarity."""
        categories = {
            'AI/ML APIs': [],
            'File Processing': [],
            'Game Logic': [],
            'GUI Applications': [],
            'Data Analysis': [],
            'System Automation': [],
            'Communication': []
        }
        
        for project_name, details in self.project_details.items():
            deps = set(details['code_metrics']['external_dependencies'])
            
            # Categorize based on dependencies and naming patterns
            if any(ai_dep in deps for ai_dep in ['anthropic', 'openai', 'transformers', 'torch']):
                categories['AI/ML APIs'].append(project_name)
            elif any(gui_dep in deps for gui_dep in ['tkinter', 'PyQt5']):
                categories['GUI Applications'].append(project_name)
            elif any(data_dep in deps for data_dep in ['numpy', 'matplotlib', 'textstat', 'wordcloud']):
                categories['Data Analysis'].append(project_name)
            elif 'file' in project_name.lower() or project_name in ['allseeingeye', 'mover', 'xmlmerge', 'jsonreader']:
                categories['File Processing'].append(project_name)
            elif project_name in ['4x', 'Quantum_Chess', 'nomic']:
                categories['Game Logic'].append(project_name)
            elif 'chat' in project_name.lower() or 'bluetooth' in project_name.lower():
                categories['Communication'].append(project_name)
            else:
                categories['System Automation'].append(project_name)
        
        return {cat: projects for cat, projects in categories.items() if projects}
    
    def identify_integration_opportunities(self) -> List[Dict[str, any]]:
        """Identify specific integration opportunities."""
        opportunities = []
        
        shared_deps = self.find_shared_dependencies()
        functional_groups = self.analyze_functional_similarity()
        
        # Opportunity 1: API Wrapper Consolidation
        api_projects = []
        for project_name, details in self.project_details.items():
            deps = details['code_metrics']['external_dependencies']
            if any(api in deps for api in ['anthropic', 'openai', 'harmonized_api_wrappers']):
                api_projects.append(project_name)
        
        if len(api_projects) > 2:
            opportunities.append({
                'type': 'API Consolidation',
                'projects': api_projects,
                'description': 'Multiple projects use AI APIs. Consider creating a unified API wrapper.',
                'complexity': 'Medium',
                'impact': 'High',
                'effort_hours': 20
            })
        
        # Opportunity 2: File Processing Pipeline
        file_projects = functional_groups.get('File Processing', [])
        if len(file_projects) > 2:
            opportunities.append({
                'type': 'File Processing Pipeline',
                'projects': file_projects,
                'description': 'File processing projects could be combined into a unified pipeline.',
                'complexity': 'Low',
                'impact': 'Medium',
                'effort_hours': 15
            })
        
        # Opportunity 3: GUI Framework Standardization
        gui_projects = functional_groups.get('GUI Applications', [])
        if len(gui_projects) > 1:
            opportunities.append({
                'type': 'GUI Framework',
                'projects': gui_projects,
                'description': 'GUI projects could share common interface components.',
                'complexity': 'High',
                'impact': 'Medium',
                'effort_hours': 40
            })
        
        # Opportunity 4: Game Engine Components
        game_projects = functional_groups.get('Game Logic', [])
        if len(game_projects) > 1:
            opportunities.append({
                'type': 'Game Engine',
                'projects': game_projects,
                'description': 'Game projects could share common engine components.',
                'complexity': 'High',
                'impact': 'High',
                'effort_hours': 60
            })
        
        # Opportunity 5: Data Analysis Suite
        data_projects = functional_groups.get('Data Analysis', [])
        if len(data_projects) > 1:
            opportunities.append({
                'type': 'Data Analysis Suite',
                'projects': data_projects,
                'description': 'Data analysis tools could be integrated into a comprehensive suite.',
                'complexity': 'Medium',
                'impact': 'High',
                'effort_hours': 30
            })
        
        return opportunities
    
    def calculate_integration_complexity(self, projects: List[str]) -> Dict[str, any]:
        """Calculate the complexity of integrating specific projects."""
        total_loc = 0
        total_functions = 0
        total_classes = 0
        complexity_scores = []
        dependencies = set()
        
        for project in projects:
            if project in self.project_details:
                details = self.project_details[project]
                code = details['code_metrics']
                
                total_loc += code['code_lines']
                total_functions += code['functions']
                total_classes += code['classes']
                complexity_scores.append(code['complexity_score'])
                dependencies.update(code['external_dependencies'])
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        # Calculate integration difficulty score (0-100)
        size_factor = min(total_loc / 1000, 1) * 30
        complexity_factor = min(avg_complexity / 5, 1) * 40
        dependency_factor = min(len(dependencies) / 10, 1) * 30
        
        difficulty_score = size_factor + complexity_factor + dependency_factor
        
        return {
            'total_lines': total_loc,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'unique_dependencies': len(dependencies),
            'average_complexity': round(avg_complexity, 2),
            'integration_difficulty': round(difficulty_score),
            'estimated_effort_weeks': round(total_loc / 500 + len(dependencies) * 0.5, 1)
        }
    
    def generate_integration_roadmap(self) -> str:
        """Generate a comprehensive integration roadmap."""
        opportunities = self.identify_integration_opportunities()
        shared_deps = self.find_shared_dependencies()
        functional_groups = self.analyze_functional_similarity()
        
        content = f"""# Cross-Project Integration Roadmap

## 🎯 Integration Opportunities Analysis

Based on dependency analysis and functional similarity, we've identified {len(opportunities)} major integration opportunities.

### 📊 Shared Dependencies Analysis

Projects sharing dependencies suggest natural integration points:

"""
        
        # Show top shared dependencies
        sorted_shared = sorted(shared_deps.items(), key=lambda x: len(x[1]), reverse=True)
        for dep, projects in sorted_shared[:10]:
            if len(projects) > 2:
                content += f"#### {dep}\n"
                content += f"**Used by {len(projects)} projects**: {', '.join(projects)}\n"
                content += f"**Integration potential**: {'High' if len(projects) > 3 else 'Medium'}\n\n"
        
        content += "### 🔗 Functional Groups\n\n"
        
        for category, projects in functional_groups.items():
            if len(projects) > 1:
                content += f"#### {category}\n"
                content += f"**Projects**: {', '.join(projects)}\n"
                
                complexity = self.calculate_integration_complexity(projects)
                content += f"**Integration complexity**: {complexity['integration_difficulty']}/100\n"
                content += f"**Estimated effort**: {complexity['estimated_effort_weeks']} weeks\n\n"
        
        content += "## 🚀 Recommended Integration Projects\n\n"
        
        # Sort opportunities by impact and complexity
        sorted_opportunities = sorted(opportunities, 
                                    key=lambda x: (x['impact'] == 'High', x['complexity'] != 'High'))
        
        for i, opp in enumerate(sorted_opportunities, 1):
            content += f"### {i}. {opp['type']}\n"
            content += f"**Projects involved**: {', '.join(opp['projects'])}\n"
            content += f"**Description**: {opp['description']}\n"
            content += f"**Complexity**: {opp['complexity']}\n"
            content += f"**Impact**: {opp['impact']}\n"
            content += f"**Estimated effort**: {opp['effort_hours']} hours\n"
            
            # Add detailed analysis
            complexity = self.calculate_integration_complexity(opp['projects'])
            content += f"**Technical details**:\n"
            content += f"  - Total codebase: {complexity['total_lines']:,} lines\n"
            content += f"  - Functions to integrate: {complexity['total_functions']}\n"
            content += f"  - Classes to merge: {complexity['total_classes']}\n"
            content += f"  - Unique dependencies: {complexity['unique_dependencies']}\n\n"
        
        content += """## 📈 Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
1. **API Wrapper Consolidation**: Unify AI API interfaces
2. **Dependency Standardization**: Create centralized requirements management
3. **Common Utilities**: Extract shared functionality

### Phase 2: Domain Integration (Weeks 3-6)
1. **File Processing Pipeline**: Integrate file utilities
2. **Data Analysis Suite**: Combine analysis tools
3. **Communication Framework**: Unify chat and messaging tools

### Phase 3: Advanced Integration (Weeks 7-12)
1. **Game Engine Components**: Extract common game logic
2. **GUI Framework**: Standardize interface components
3. **Cross-Domain Integration**: Connect different functional areas

## 🔧 Technical Considerations

### Architecture Patterns
- **Plugin Architecture**: Allow projects to remain independent while sharing core functionality
- **Microservices**: Split large projects into smaller, composable services
- **API Gateway**: Provide unified access to different project capabilities

### Code Organization
```
projects/
├── core/                 # Shared utilities and frameworks
│   ├── api_wrappers/    # Unified AI API interfaces
│   ├── file_processing/ # Common file operations
│   └── gui_components/  # Shared UI elements
├── integrations/        # Cross-project integrations
│   ├── ai_suite/       # Combined AI tools
│   ├── file_pipeline/  # File processing workflow
│   └── game_engine/    # Game development framework
└── standalone/         # Independent projects
```

### Migration Strategy
1. **Extract Common Code**: Identify and extract shared functionality
2. **Create Adapters**: Build compatibility layers for existing projects
3. **Gradual Migration**: Move projects to new architecture incrementally
4. **Maintain Compatibility**: Ensure existing functionality continues to work

## 📊 Success Metrics

- **Code Reuse**: Increase shared code percentage from 0% to 30%
- **Dependency Reduction**: Reduce unique dependencies by 40%
- **Development Speed**: Decrease new feature development time by 50%
- **Maintenance Cost**: Reduce bug fix time by 60%

---

*Integration roadmap generated based on comprehensive repository analysis*
"""
        
        return content
    
    def save_integration_analysis(self):
        """Save integration analysis to file."""
        roadmap = self.generate_integration_roadmap()
        
        with open('INTEGRATION_ROADMAP.md', 'w') as f:
            f.write(roadmap)
        
        # Also save raw analysis data
        analysis_data = {
            'shared_dependencies': self.find_shared_dependencies(),
            'functional_groups': self.analyze_functional_similarity(),
            'integration_opportunities': self.identify_integration_opportunities()
        }
        
        with open('integration_analysis.json', 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print("🔗 Integration analysis saved:")
        print("  - INTEGRATION_ROADMAP.md: Comprehensive integration roadmap")
        print("  - integration_analysis.json: Raw analysis data")

def main():
    """Main function to run integration analysis."""
    if not os.path.exists('repository_analysis.json'):
        print("❌ repository_analysis.json not found. Please run repository_analyzer.py first.")
        return
    
    print("🔍 Analyzing cross-project integration opportunities...")
    analyzer = IntegrationAnalyzer()
    
    print("📋 Generating integration roadmap...")
    analyzer.save_integration_analysis()
    
    print("✅ Integration analysis complete!")

if __name__ == "__main__":
    main()