# Contributing to Projects Repository

First off, thank you for considering contributing to this repository! This collection of projects represents a diverse ecosystem of tools, utilities, and experiments, and we welcome contributions of all kinds.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Style Guidelines](#style-guidelines)
- [Project-Specific Guidelines](#project-specific-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the repository maintainers.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, commands, etc.)
- **Describe the behavior you observed and what you expected**
- **Include screenshots** if applicable
- **Specify your environment** (OS, Python version, dependencies)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed enhancement
- **Explain why this enhancement would be useful**
- **List any alternative solutions** you've considered
- **Include mockups or examples** if applicable

### Pull Requests

We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure your code follows the existing style
4. Update documentation as needed
5. Issue the pull request with a clear description

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Basic understanding of Python and GitHub workflow

### Initial Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/projects.git
cd projects

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install common dependencies
pip install requests rich lxml

# For AI/LLM projects
pip install anthropic torch transformers

# Install development dependencies (if applicable)
pip install pytest black flake8
```

### Running Projects

Each project is standalone. Navigate to the specific project directory and follow its README:

```bash
cd ChatGPTArchive
python chatgptarchive.py
```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clear, commented code
- Follow the existing code style
- Add documentation for new features
- Test your changes thoroughly

### 3. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of your changes"
```

### 4. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

## 🎨 Style Guidelines

### Python Code Style

We follow PEP 8 with some flexibility:

```python
# Good: Clear variable names, proper spacing
def process_conversation_data(input_file, output_dir):
    """Process ChatGPT conversation data and generate reports.
    
    Args:
        input_file (str): Path to input JSON file
        output_dir (str): Directory for output files
        
    Returns:
        dict: Processing statistics
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return process_data(data)

# Bad: Unclear names, poor formatting
def pcd(if,od):
    f=open(if,'r')
    d=json.load(f)
    return pd(d)
```

### General Guidelines

- **Use descriptive variable names** - `conversation_count` not `cc`
- **Add docstrings** to functions and classes
- **Comment complex logic** but avoid obvious comments
- **Keep functions focused** - one function, one purpose
- **Limit line length** to 100 characters when practical
- **Use type hints** when it improves clarity

### Documentation Style

- Use Markdown for all documentation
- Include code examples where applicable
- Keep explanations clear and concise
- Update relevant documentation when changing code

## 📁 Project-Specific Guidelines

### AI/LLM Projects

- Always use environment variables for API keys
- Include example `.env.example` files
- Document API rate limits and costs
- Handle API errors gracefully
- Log API interactions for debugging

### File Processing Projects

- Validate input files before processing
- Handle missing files and directories gracefully
- Provide clear error messages
- Support batch processing where applicable
- Don't modify original files unless explicitly intended

### Game Projects

- Keep game logic separate from display logic
- Document game rules clearly
- Include example gameplay in README
- Provide save/load functionality where applicable
- Handle edge cases in game state

### Automation Projects

- Include safety checks (confirm before destructive operations)
- Log all automated actions
- Provide dry-run mode
- Document required permissions
- Handle interruptions gracefully

## 💬 Commit Messages

### Format

```
type(scope): Short description (50 chars or less)

More detailed explanatory text, if necessary. Wrap at 72 characters.
Explain the problem that this commit is solving, and why you chose
this approach.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: "Fixes #123" or "Relates to #456"
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(chatgpt): Add sentiment analysis to conversation processor

Implement sentiment analysis using VADER lexicon to analyze
conversation tone and emotional content.

- Add sentiment scoring to each message
- Generate sentiment trends over time
- Update documentation with usage examples

Fixes #42
```

```
fix(mover): Prevent duplicate file processing

Check file hash before moving to prevent processing the same
file multiple times when monitoring directories.

Fixes #15
```

## 🔍 Pull Request Process

### Before Submitting

1. ✅ Test your changes thoroughly
2. ✅ Update documentation (README, docstrings, etc.)
3. ✅ Ensure code follows style guidelines
4. ✅ Add or update tests if applicable
5. ✅ Verify all tests pass
6. ✅ Update CHANGELOG if the project has one

### PR Description Template

```markdown
## Description
Brief description of what this PR does and why.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Changes Made
- List specific changes
- Use bullet points
- Be clear and concise

## Testing
Describe how you tested your changes:
- Test cases run
- Manual testing performed
- Edge cases considered

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Fixes #(issue number)
Related to #(issue number)
```

### Review Process

1. At least one maintainer will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be acknowledged in the commit

## 🏷️ Versioning

For projects that use versioning, we follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- 🐛 Bug fixes and stability improvements
- 📚 Documentation improvements
- ✅ Adding tests to existing projects
- ♿ Accessibility improvements

### Medium Priority
- 🌟 New features for existing projects
- 🔧 Performance optimizations
- 🎨 UI/UX improvements for GUI projects
- 🔐 Security enhancements

### Nice to Have
- 🆕 New standalone projects
- 🔄 Refactoring for better code organization
- 🌍 Internationalization
- 📊 Analytics and metrics

## 💡 Need Help?

- 📖 Check the [README](README.md) and project-specific documentation
- 🔍 Search existing [issues](https://github.com/CrazyDubya/projects/issues)
- 💬 Start a [discussion](https://github.com/CrazyDubya/projects/discussions)
- 📧 Contact the maintainers

## 🙏 Recognition

Contributors are recognized in several ways:

- Listed in the project's contributors
- Mentioned in release notes for significant contributions
- GitHub's automatic contribution tracking

Thank you for contributing to this project! 🎉

---

*These guidelines are adapted from best practices and may evolve as the project grows. Suggestions for improving these guidelines are always welcome!*