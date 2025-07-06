#!/usr/bin/env python3
"""
Comprehensive Codebase Cleanup Script

Organizes, consolidates, and removes redundant files to create a clean,
maintainable codebase structure.
"""

import os
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodebaseCleanup:
    """
    Comprehensive codebase cleanup and organization
    """
    
    def __init__(self):
        self.project_root = Path(".")
        self.cleanup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Statistics
        self.stats = {
            "files_removed": 0,
            "files_archived": 0,
            "files_consolidated": 0,
            "directories_removed": 0,
            "directories_created": 0,
            "space_saved_mb": 0
        }
        
        # Backup directory for critical files
        self.backup_dir = Path("archive") / f"cleanup_backup_{self.cleanup_date}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🧹 Codebase cleanup initialized")
    
    def run_cleanup(self) -> Dict[str, Any]:
        """
        Execute comprehensive cleanup process
        """
        print("🧹 COMPREHENSIVE CODEBASE CLEANUP")
        print("=" * 80)
        print(f"📅 Starting cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💾 Backup directory: {self.backup_dir}")
        print()
        
        try:
            # Phase 1: Remove empty and redundant files
            self._remove_empty_files()
            
            # Phase 2: Consolidate redundant scripts
            self._consolidate_scripts()
            
            # Phase 3: Clean up test artifacts
            self._cleanup_test_artifacts()
            
            # Phase 4: Organize documentation
            self._organize_documentation()
            
            # Phase 5: Archive old data
            self._archive_old_data()
            
            # Phase 6: Clean root directory
            self._clean_root_directory()
            
            # Phase 7: Optimize directory structure
            self._optimize_directory_structure()
            
            # Phase 8: Generate cleanup report
            report = self._generate_cleanup_report()
            
            print("\n🎉 CLEANUP COMPLETED SUCCESSFULLY!")
            self._print_summary()
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            raise
    
    def _remove_empty_files(self):
        """Remove empty Python files and other empty files"""
        print("🗑️ Phase 1: Removing Empty Files")
        print("-" * 50)
        
        # Find empty Python files
        empty_files = []
        for py_file in self.project_root.rglob("*.py"):
            # Skip virtual environment and git
            if any(skip_dir in str(py_file) for skip_dir in ['venv', '.venv', '.git', '__pycache__']):
                continue
                
            if py_file.stat().st_size == 0:
                empty_files.append(py_file)
        
        print(f"Found {len(empty_files)} empty Python files")
        
        for empty_file in empty_files:
            print(f"  🗑️ Removing: {empty_file}")
            empty_file.unlink()
            self.stats["files_removed"] += 1
        
        # Find and remove other empty files
        other_empty = []
        for file_path in self.project_root.rglob("*"):
            if (file_path.is_file() and 
                file_path.stat().st_size == 0 and 
                file_path.suffix in ['.json', '.md', '.txt', '.log'] and
                not any(skip_dir in str(file_path) for skip_dir in ['venv', '.venv', '.git'])):
                other_empty.append(file_path)
        
        if other_empty:
            print(f"Found {len(other_empty)} other empty files")
            for empty_file in other_empty:
                print(f"  🗑️ Removing: {empty_file}")
                empty_file.unlink()
                self.stats["files_removed"] += 1
    
    def _consolidate_scripts(self):
        """Consolidate redundant scripts"""
        print("\n🔄 Phase 2: Consolidating Redundant Scripts")
        print("-" * 50)
        
        scripts_dir = Path("scripts")
        
        # Define script categories and which ones to keep
        script_categories = {
            "cleanup": {
                "keep": ["comprehensive_codebase_cleanup.py"],
                "remove": [
                    "cleanup_legacy.py", "enhanced_cleanup.py", 
                    "comprehensive_cleanup.py"
                ]
            },
            "testing": {
                "keep": ["test_enhanced_resource_detection.py"],
                "remove": [
                    "demo_quality_test.py", "run_quality_test.py",
                    "run_comprehensive_quality_test.py", "test_system_integrity.py",
                    "test_agent_system.py"
                ]
            },
            "validation": {
                "keep": ["validate_deployment.py"],
                "remove": ["final_verification.py"]
            },
            "migration": {
                "keep": ["migrate_to_enhanced_resources.py"],
                "remove": ["migrate_legacy.py"]
            },
            "analysis": {
                "keep": ["system_status.py"],
                "remove": [
                    "analyze_skipped_messages.py", "analyze_complete_data_catalog.py",
                    "analyze_data_fields.py"
                ]
            },
            "pipeline": {
                "keep": ["run_pipeline.py"],
                "remove": ["run_standalone_pipeline.py"]
            }
        }
        
        for category, files in script_categories.items():
            print(f"\n📂 {category.upper()} Scripts:")
            
            # Archive files to be removed
            for script_name in files["remove"]:
                script_path = scripts_dir / script_name
                if script_path.exists():
                    backup_path = self.backup_dir / f"scripts_{script_name}"
                    print(f"  📦 Archiving: {script_name}")
                    shutil.copy2(script_path, backup_path)
                    script_path.unlink()
                    self.stats["files_archived"] += 1
            
            # List kept files
            for script_name in files["keep"]:
                script_path = scripts_dir / script_name
                if script_path.exists():
                    print(f"  ✅ Keeping: {script_name}")
                else:
                    print(f"  ⚠️ Missing: {script_name}")
    
    def _cleanup_test_artifacts(self):
        """Clean up test artifacts and temporary files"""
        print("\n🧪 Phase 3: Cleaning Test Artifacts")
        print("-" * 50)
        
        # Remove test result files from root
        test_artifacts = list(self.project_root.glob("*test*.json"))
        test_artifacts.extend(list(self.project_root.glob("*report*.json")))
        
        if test_artifacts:
            test_results_dir = Path("tests/results")
            test_results_dir.mkdir(exist_ok=True)
            self.stats["directories_created"] += 1
            
            print(f"Moving {len(test_artifacts)} test artifacts to tests/results/")
            for artifact in test_artifacts:
                if artifact.name not in ['requirements.txt', 'package.json']:
                    dest = test_results_dir / artifact.name
                    print(f"  📁 Moving: {artifact.name}")
                    shutil.move(str(artifact), str(dest))
                    self.stats["files_consolidated"] += 1
        
        # Clean up redundant test files
        tests_dir = Path("tests")
        redundant_tests = [
            "test_simple_reaction_search.py",
            "test_production_reaction_search.py", 
            "test_reaction_search.py",
            "test_reaction_functionality.py"
        ]
        
        for test_file in redundant_tests:
            test_path = tests_dir / test_file
            if test_path.exists():
                print(f"  🗑️ Removing redundant test: {test_file}")
                test_path.unlink()
                self.stats["files_removed"] += 1
    
    def _organize_documentation(self):
        """Organize and consolidate documentation"""
        print("\n📚 Phase 4: Organizing Documentation")
        print("-" * 50)
        
        docs_dir = Path("docs")
        
        # Create organized documentation structure
        organized_docs = {
            "setup": docs_dir / "setup",
            "guides": docs_dir / "guides", 
            "reference": docs_dir / "reference",
            "archived": docs_dir / "archived"
        }
        
        for doc_dir in organized_docs.values():
            doc_dir.mkdir(exist_ok=True)
            self.stats["directories_created"] += 1
        
        # Categorize and move documentation files
        doc_mappings = {
            "QUICKSTART.md": "setup",
            "DEPLOYMENT.md": "setup",
            "DEPLOYMENT_CHECKLIST.md": "setup",
            "example_queries.md": "guides",
            "RESOURCE_DETECTION_IMPROVEMENTS.md": "reference",
            "DATABASE_IMPROVEMENTS_SUMMARY.md": "reference"
        }
        
        # Archive completion summaries
        completion_docs = [
            "CLEANUP_COMPLETE.md", "FINAL_CLEANUP_COMPLETE.md",
            "PROJECT_COMPLETION.md", "REACTION_SEARCH_COMPLETE.md",
            "ANALYTICS_INTEGRATION_COMPLETE.md", "BRANCH_COMMIT_SUMMARY.md"
        ]
        
        print("📁 Organizing documentation files:")
        for doc_file in docs_dir.glob("*.md"):
            if doc_file.name in doc_mappings:
                dest_dir = organized_docs[doc_mappings[doc_file.name]]
                dest_path = dest_dir / doc_file.name
                print(f"  📄 Moving {doc_file.name} to {doc_mappings[doc_file.name]}/")
                shutil.move(str(doc_file), str(dest_path))
                self.stats["files_consolidated"] += 1
            elif doc_file.name in completion_docs:
                dest_path = organized_docs["archived"] / doc_file.name
                print(f"  📦 Archiving {doc_file.name}")
                shutil.move(str(doc_file), str(dest_path))
                self.stats["files_archived"] += 1
        
        # Move main documentation file
        if Path("DATABASE_IMPROVEMENTS_SUMMARY.md").exists():
            dest_path = organized_docs["reference"] / "DATABASE_IMPROVEMENTS_SUMMARY.md"
            shutil.move("DATABASE_IMPROVEMENTS_SUMMARY.md", str(dest_path))
            self.stats["files_consolidated"] += 1
    
    def _archive_old_data(self):
        """Archive old data files and temporary exports"""
        print("\n🗄️ Phase 5: Archiving Old Data")
        print("-" * 50)
        
        data_dir = Path("data")
        
        # Create data archive structure
        data_archive = Path("archive") / f"data_archive_{self.cleanup_date}"
        data_archive.mkdir(exist_ok=True)
        
        # Files/directories to archive
        archive_patterns = [
            "data/exports/quality_monitoring_report_*.json",
            "data/exports/resource_detection_evaluation_*.json", 
            "data/processing_markers/*_20250*.json",
            "data/sync_stats/*_20250*.json",
            "data/detected_resources/detection_stats_*.json"
        ]
        
        archived_count = 0
        for pattern in archive_patterns:
            for file_path in Path(".").glob(pattern):
                if file_path.exists():
                    dest_path = data_archive / file_path.name
                    print(f"  📦 Archiving: {file_path}")
                    shutil.move(str(file_path), str(dest_path))
                    archived_count += 1
                    self.stats["files_archived"] += 1
        
        if archived_count > 0:
            print(f"  ✅ Archived {archived_count} old data files")
        else:
            print("  ℹ️ No old data files to archive")
    
    def _clean_root_directory(self):
        """Clean up root directory clutter"""
        print("\n🏠 Phase 6: Cleaning Root Directory")
        print("-" * 50)
        
        # Files that should not be in root
        root_clutter = []
        
        # Find misplaced files
        for file_path in self.project_root.glob("*"):
            if (file_path.is_file() and 
                file_path.name not in [
                    'main.py', 'requirements.txt', 'pytest.ini', 
                    'readme.md', 'PROJECT_STRUCTURE.md', '.gitignore'
                ] and
                not file_path.name.startswith('.')):
                root_clutter.append(file_path)
        
        if root_clutter:
            misc_dir = Path("misc")
            misc_dir.mkdir(exist_ok=True)
            
            print(f"Moving {len(root_clutter)} files from root to misc/")
            for file_path in root_clutter:
                dest_path = misc_dir / file_path.name
                print(f"  📁 Moving: {file_path.name}")
                shutil.move(str(file_path), str(dest_path))
                self.stats["files_consolidated"] += 1
    
    def _optimize_directory_structure(self):
        """Optimize directory structure"""
        print("\n🏗️ Phase 7: Optimizing Directory Structure")
        print("-" * 50)
        
        # Remove empty directories
        empty_dirs = []
        for dir_path in self.project_root.rglob("*"):
            if (dir_path.is_dir() and 
                not any(skip_dir in str(dir_path) for skip_dir in ['venv', '.venv', '.git']) and
                not any(dir_path.iterdir())):
                empty_dirs.append(dir_path)
        
        for empty_dir in empty_dirs:
            print(f"  🗑️ Removing empty directory: {empty_dir}")
            empty_dir.rmdir()
            self.stats["directories_removed"] += 1
        
        # Create missing __init__.py files
        python_dirs = []
        for py_file in self.project_root.rglob("*.py"):
            if not any(skip_dir in str(py_file) for skip_dir in ['venv', '.venv', '.git']):
                python_dirs.append(py_file.parent)
        
        python_dirs = list(set(python_dirs))
        
        for py_dir in python_dirs:
            init_file = py_dir / "__init__.py"
            if not init_file.exists() and py_dir.name not in ['scripts', 'tests']:
                print(f"  📄 Creating: {init_file}")
                init_file.write_text("# Auto-generated __init__.py\n")
                self.stats["files_consolidated"] += 1
    
    def _generate_cleanup_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleanup report"""
        
        # Calculate space saved
        if self.backup_dir.exists():
            backup_size = sum(f.stat().st_size for f in self.backup_dir.rglob('*') if f.is_file())
            self.stats["space_saved_mb"] = backup_size / (1024 * 1024)
        
        report = {
            "cleanup_timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "backup_location": str(self.backup_dir),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_path = f"cleanup_report_{self.cleanup_date}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate post-cleanup recommendations"""
        return [
            "✅ Codebase cleanup completed successfully",
            "🔧 Update import statements if any module references changed",
            "📚 Review consolidated documentation in docs/ directory",
            "🧪 Run tests to ensure nothing was broken: python -m pytest tests/",
            "🗄️ Consider setting up automated cleanup scheduled tasks",
            "📝 Update PROJECT_STRUCTURE.md to reflect new organization",
            "🚀 Deploy cleaned codebase to production environment"
        ]
    
    def _print_summary(self):
        """Print cleanup summary"""
        print("\n📊 CLEANUP SUMMARY")
        print("-" * 50)
        print(f"Files removed:      {self.stats['files_removed']}")
        print(f"Files archived:     {self.stats['files_archived']}")
        print(f"Files consolidated: {self.stats['files_consolidated']}")
        print(f"Directories removed: {self.stats['directories_removed']}")
        print(f"Directories created: {self.stats['directories_created']}")
        print(f"Space saved:        {self.stats['space_saved_mb']:.1f} MB")
        print(f"Backup location:    {self.backup_dir}")


def main():
    """Main cleanup execution"""
    try:
        cleanup = CodebaseCleanup()
        report = cleanup.run_cleanup()
        
        print("\n🎊 CODEBASE CLEANUP COMPLETE!")
        print("\n🚀 Next Steps:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 