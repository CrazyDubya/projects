#!/usr/bin/env python3
"""
AllSeeingEye - Directory Analysis Tool

A comprehensive directory analysis tool that builds visual trees and processes 
files by type with detailed content analysis and reporting.

Enhanced with shared utilities for better error handling, logging, and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, TextIO, Set

# Add parent directory to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_utils import setup_logging, safe_file_operation, ProjectConfig, ensure_directory

# Configuration
CONFIG = ProjectConfig()
LOGGER = setup_logging(level=CONFIG.get('log_level', 'INFO'))

# File type configurations - can be customized via config
CODE_FILES = CONFIG.get('code_files', ['.py', '.js', '.html', '.css', '.csv'])
DATA_FILES = CONFIG.get('data_files', ['.txt'])
EXCLUDED_DIRS = CONFIG.get('excluded_dirs', ['chatter', 'data', '__pycache__', '.git', '.venv'])
EXCLUDED_FILES = CONFIG.get('excluded_files', ['chat.html', 'conversations.json'])

def list_directory(directory: Path) -> list[str]:
    """
    List directory contents including hidden files.
    
    This function is Unix/Linux/MacOS specific and uses the 'ls' command.
    
    Args:
        directory: Path to the directory to list
        
    Returns:
        List of filenames in the directory
        
    Raises:
        subprocess.CalledProcessError: If ls command fails
    """
    try:
        cmd = ['ls', '-1a', str(directory)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to list directory {directory}: {e}")
        return []
    except FileNotFoundError:
        LOGGER.error("'ls' command not found. This tool requires a Unix-like system.")
        return []


def process_file(file_path: Path, output_file: TextIO, include_text: bool = True) -> None:
    """
    Process individual files based on their extension and configuration.
    
    Args:
        file_path: Path to the file to process
        output_file: Output file handle to write results
        include_text: Whether to include text content for data files
    """
    try:
        ext = file_path.suffix.lower()
        
        if ext in CODE_FILES or (ext in DATA_FILES and include_text):
            # For code and data files, include the content
            content = safe_file_operation('read', file_path)
            if content is not None:
                header = f"## {file_path.name} ##\n" if ext in CODE_FILES else ""
                output_file.write(f"{header}{content}\n\n")
                LOGGER.debug(f"Processed file with content: {file_path}")
            else:
                output_file.write(f"Error reading file: {file_path}\n")
                LOGGER.warning(f"Failed to read file: {file_path}")
        else:
            # For other files, just list the file name and location
            output_file.write(f"{file_path}\n")
            LOGGER.debug(f"Listed file without content: {file_path}")
            
    except Exception as e:
        error_msg = f"Error processing file {file_path}: {e}"
        output_file.write(f"{error_msg}\n")
        LOGGER.error(error_msg)


def should_exclude_path(path: Path) -> bool:
    """
    Check if a path should be excluded from processing.
    
    Args:
        path: Path to check
        
    Returns:
        True if path should be excluded, False otherwise
    """
    # Check if directory should be excluded
    if path.is_dir() and path.name in EXCLUDED_DIRS:
        return True
    
    # Check if file should be excluded
    if path.is_file() and path.name in EXCLUDED_FILES:
        return True
    
    # Check for specific path patterns
    excluded_patterns = [
        'chatter/data/chat.html',
        'chatter/data/conversations.json'
    ]
    
    for pattern in excluded_patterns:
        if str(path).endswith(pattern):
            return True
    
    return False

def build_tree(directory: Path, output_file: TextIO, include_text: bool, prefix: str = '') -> None:
    """
    Recursively build a visual tree of the directory structure and process files.
    
    Args:
        directory: Path to the directory to analyze
        output_file: Output file handle to write results
        include_text: Whether to include text content for data files
        prefix: Prefix for tree visualization
    """
    try:
        entries = list_directory(directory)
        file_count = 0
        dir_count = 0
        
        for entry in entries:
            if entry in ['.', '..']:
                continue  # Skip current and parent directory markers
                
            path = directory / entry
            
            if should_exclude_path(path):
                if path.is_dir():
                    output_file.write(f"{prefix}+-- {entry}/ # Excluded directory\n")
                    LOGGER.debug(f"Excluded directory: {path}")
                continue
            
            if path.is_dir():
                # Directory: write its name and explore recursively
                output_file.write(f"{prefix}+-- {entry}/\n")
                dir_count += 1
                build_tree(path, output_file, include_text, prefix=prefix + "    ")
            else:
                # File: process based on configuration
                output_file.write(f"{prefix}+-- {entry}\n")
                file_count += 1
                process_file(path, output_file, include_text)
        
        LOGGER.info(f"Processed directory {directory}: {dir_count} subdirs, {file_count} files")
        
    except Exception as e:
        error_msg = f"Error processing directory {directory}: {e}"
        output_file.write(f"{error_msg}\n")
        LOGGER.error(error_msg)


def run_tree_command(directory: Path, output_file: Path) -> bool:
    """
    Run the system 'tree' command to generate directory structure.
    
    Args:
        directory: Directory to analyze
        output_file: File to write tree output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, "w") as f:
            result = subprocess.run(
                ["tree", "-a", str(directory)], 
                stdout=f, 
                text=True, 
                check=True
            )
        LOGGER.info(f"Generated tree structure for {directory}")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Tree command failed: {e}")
        return False
    except FileNotFoundError:
        LOGGER.warning("'tree' command not found, skipping tree visualization")
        return False


def main() -> None:
    """
    Main function to orchestrate directory analysis.
    
    This function generates comprehensive directory analysis reports with
    optional text content inclusion.
    """
    LOGGER.info("Starting AllSeeingEye directory analysis")
    
    # Configuration
    current_directory = Path.cwd()
    output_dir = ensure_directory(CONFIG.get('output_dir', './output'))
    
    # Output filenames
    tree_output_filename = output_dir / "_directory_tree.txt"
    final_output_filename = output_dir / "directory_structure_and_files.txt"
    final_output_filename_no_text = output_dir / "directory_structure_and_files_no_text.txt"
    
    LOGGER.info(f"Analyzing directory: {current_directory}")
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Generate tree structure if available
    tree_available = run_tree_command(current_directory, tree_output_filename)
    
    # Generate detailed analysis reports
    reports = [
        (True, final_output_filename, "with text content"),
        (False, final_output_filename_no_text, "without text content")
    ]
    
    for include_text, output_filename, description in reports:
        LOGGER.info(f"Generating report {description}: {output_filename}")
        
        try:
            with open(output_filename, "w", encoding='utf-8') as output_file:
                # Write header
                output_file.write(f"# Directory Analysis Report - {description.title()}\n")
                output_file.write(f"# Generated: {CONFIG.get('timestamp', 'Unknown')}\n")
                output_file.write(f"# Directory: {current_directory}\n\n")
                
                # Include tree output if available
                if tree_available and safe_file_operation('exists', tree_output_filename):
                    tree_content = safe_file_operation('read', tree_output_filename)
                    if tree_content:
                        output_file.write("## Tree Structure ##\n")
                        output_file.write(tree_content)
                        output_file.write("\n## Detailed Analysis ##\n")
                
                # Generate detailed structure and file content
                build_tree(current_directory, output_file, include_text)
            
            LOGGER.info(f"Successfully generated: {output_filename}")
            
        except Exception as e:
            LOGGER.error(f"Failed to generate report {output_filename}: {e}")
    
    # Cleanup temporary tree file
    if tree_available and safe_file_operation('exists', tree_output_filename):
        try:
            tree_output_filename.unlink()
            LOGGER.debug("Cleaned up temporary tree file")
        except Exception as e:
            LOGGER.warning(f"Failed to cleanup tree file: {e}")
    
    LOGGER.info("AllSeeingEye analysis completed")
    print(f"Reports generated in: {output_dir}")
    print(f"- Full report: {final_output_filename.name}")
    print(f"- Summary report: {final_output_filename_no_text.name}")

if __name__ == "__main__":
    # Set timestamp for this run
    from datetime import datetime
    CONFIG.set('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    main()
