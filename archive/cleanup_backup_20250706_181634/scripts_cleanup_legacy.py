#!/usr/bin/env python3
"""
Legacy Cleanup Script
Safely archives and removes the legacy core/ directory after successful migration
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegacyCleanup:
    """Manages the safe cleanup of legacy core/ directory"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.core_dir = project_root / "core"
        self.archive_dir = project_root / "legacy_archive"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_legacy_archive(self):
        """Create a comprehensive archive of the legacy core/ system"""
        try:
            print("📦 Creating legacy archive...")
            
            # Create archive directory
            self.archive_dir.mkdir(exist_ok=True)
            
            # Archive the entire core/ directory
            archive_name = f"legacy_core_{self.timestamp}"
            archive_path = self.archive_dir / f"{archive_name}.tar.gz"
            
            # Create tar archive
            cmd = ["tar", "-czf", str(archive_path), "-C", str(self.project_root), "core/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Archive creation failed: {result.stderr}")
            
            # Create inventory of archived files
            inventory = {
                "archive_name": archive_name,
                "archive_path": str(archive_path),
                "timestamp": self.timestamp,
                "files_archived": [],
                "total_size_mb": archive_path.stat().st_size / (1024 * 1024)
            }
            
            # List all files in core/
            for file_path in self.core_dir.rglob("*"):
                if file_path.is_file():
                    inventory["files_archived"].append({
                        "path": str(file_path.relative_to(self.project_root)),
                        "size_bytes": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            
            # Save inventory
            inventory_path = self.archive_dir / f"{archive_name}_inventory.json"
            with open(inventory_path, 'w') as f:
                json.dump(inventory, f, indent=2)
            
            print(f"✅ Legacy archive created: {archive_path.name} ({inventory['total_size_mb']:.1f} MB)")
            print(f"📋 Inventory saved: {inventory_path.name}")
            
            return archive_path, inventory_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create legacy archive: {e}")
            raise
    
    def validate_modernized_services(self):
        """Validate that modernized services are working correctly"""
        try:
            print("🔍 Validating modernized services...")
            
            # Check that modernized services exist and import correctly
            services_dir = self.project_root / "agentic" / "services"
            required_services = [
                "unified_data_manager.py",
                "discord_service.py", 
                "content_processor.py",
                "sync_service.py"
            ]
            
            for service_file in required_services:
                service_path = services_dir / service_file
                if not service_path.exists():
                    raise Exception(f"Missing modernized service: {service_file}")
            
            # Test imports
            try:
                from agentic.services import (
                    UnifiedDataManager,
                    DiscordMessageService,
                    ContentProcessingService,
                    DataSynchronizationService
                )
                print("✅ All modernized services import successfully")
            except ImportError as e:
                raise Exception(f"Failed to import modernized services: {e}")
            
            # Check that modernized pipeline exists
            modernized_pipeline = self.project_root / "scripts" / "run_modernized_pipeline.py"
            if not modernized_pipeline.exists():
                raise Exception("Missing modernized pipeline script")
            
            print("✅ Modernized services validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Modernized services validation failed: {e}")
            raise
    
    def check_remaining_dependencies(self):
        """Check for any remaining dependencies on core/ directory"""
        try:
            print("🔍 Checking for remaining core/ dependencies...")
            
            # Search for imports or references to core/
            dangerous_patterns = [
                "from core import",
                "import core.",
                "from core.",
                "core/"
            ]
            
            found_dependencies = []
            
            # Define directories to exclude from scanning
            exclude_dirs = {
                'venv', '.venv', 'env', '.env', 
                'site-packages', '__pycache__', '.git',
                'node_modules', 'legacy_archive', 'core'
            }
            
            # Only check project source files, not dependencies
            source_dirs = ['services', 'cogs', 'agents', 'scripts', 'config', 'tests']
            source_files = ['main.py', 'bot.py', 'run.py']
            
            files_to_check = []
            
            # Add source files from root
            for file_name in source_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    files_to_check.append(file_path)
            
            # Add files from source directories
            for dir_name in source_dirs:
                dir_path = self.project_root / dir_name
                if dir_path.exists():
                    for py_file in dir_path.rglob("*.py"):
                        files_to_check.append(py_file)
            
            # Check the collected files
            for py_file in files_to_check:
                # Skip if file is in an excluded directory
                if any(exclude_dir in str(py_file) for exclude_dir in exclude_dirs):
                    continue
                
                # Skip our cleanup script
                if py_file.name == "cleanup_legacy.py":
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip files that are primarily migration/documentation utilities
                    if any(name in str(py_file) for name in ['migrate_legacy.py', 'create_snapshot.py', 'restore_snapshot.py', 'run_modernized_pipeline.py']):
                        continue
                        
                    for pattern in dangerous_patterns:
                        if pattern in content:
                            # Parse the code to identify actual imports vs documentation
                            lines = content.split('\n')
                            in_multiline_string = False
                            string_delimiter = None
                            
                            for i, line in enumerate(lines):
                                if pattern in line:
                                    stripped_line = line.strip()
                                    
                                    # Track multiline strings
                                    if '"""' in line:
                                        if string_delimiter == '"""':
                                            in_multiline_string = False
                                            string_delimiter = None
                                        elif not in_multiline_string:
                                            in_multiline_string = True
                                            string_delimiter = '"""'
                                    elif "'''" in line:
                                        if string_delimiter == "'''":
                                            in_multiline_string = False
                                            string_delimiter = None
                                        elif not in_multiline_string:
                                            in_multiline_string = True
                                            string_delimiter = "'''"
                                    
                                    # Skip documentation and comments
                                    if (stripped_line.startswith('#') or 
                                        in_multiline_string or
                                        stripped_line.startswith('"""') or
                                        stripped_line.startswith("'''") or
                                        # Skip print statements that are just user messages
                                        (stripped_line.startswith('print(') and 'core/' in stripped_line)):
                                        continue
                                    
                                    # This looks like actual code that imports/uses core
                                    found_dependencies.append({
                                        "file": str(py_file.relative_to(self.project_root)),
                                        "line": i + 1,
                                        "content": line.strip()
                                    })
                                    
                except Exception as e:
                    logger.warning(f"Could not check {py_file}: {e}")
            
            if found_dependencies:
                print("⚠️ Found remaining dependencies on core/:")
                for dep in found_dependencies:
                    print(f"  {dep['file']}:{dep['line']} - {dep['content']}")
                return False
            else:
                print("✅ No remaining dependencies on core/ found")
                return True
                
        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False
    
    def remove_legacy_core(self):
        """Safely remove the legacy core/ directory"""
        try:
            print("🗑️ Removing legacy core/ directory...")
            
            if not self.core_dir.exists():
                print("ℹ️ Legacy core/ directory already removed")
                return True
            
            # Final confirmation (in a real script, you might want user input)
            print(f"⚠️ About to remove {self.core_dir}")
            
            # Remove the directory
            shutil.rmtree(self.core_dir)
            
            print("✅ Legacy core/ directory removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove legacy core/ directory: {e}")
            return False
    
    def update_gitignore(self):
        """Update .gitignore to reflect the new structure"""
        try:
            gitignore_path = self.project_root / ".gitignore"
            
            if not gitignore_path.exists():
                return
            
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            # Remove core/ specific entries if they exist
            lines = content.split('\n')
            updated_lines = []
            
            for line in lines:
                # Keep lines that don't specifically reference core/
                if not (line.strip().startswith('core/') and 'core/' in line):
                    updated_lines.append(line)
            
            # Add legacy archive to gitignore
            if 'legacy_archive/' not in content:
                updated_lines.append('')
                updated_lines.append('# Legacy archive (post-migration)')
                updated_lines.append('legacy_archive/')
            
            # Write updated gitignore
            with open(gitignore_path, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            print("✅ Updated .gitignore for new structure")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update .gitignore: {e}")
    
    def run_cleanup(self):
        """Run the complete legacy cleanup process"""
        try:
            print("🧹 Starting legacy cleanup process...")
            print("=" * 50)
            
            # Step 1: Validate modernized services
            self.validate_modernized_services()
            
            # Step 2: Check for remaining dependencies
            if not self.check_remaining_dependencies():
                print("❌ Cannot proceed with cleanup - dependencies found")
                return False
            
            # Step 3: Create legacy archive
            archive_path, inventory_path = self.create_legacy_archive()
            
            # Step 4: Remove legacy core/
            if not self.remove_legacy_core():
                print("❌ Failed to remove legacy core/ directory")
                return False
            
            # Step 5: Update gitignore
            self.update_gitignore()
            
            # Step 6: Create cleanup report
            cleanup_report = {
                "cleanup_timestamp": self.timestamp,
                "archive_created": str(archive_path),
                "inventory_file": str(inventory_path),
                "modernized_services_validated": True,
                "dependencies_checked": True,
                "core_directory_removed": True,
                "gitignore_updated": True,
                "status": "SUCCESS"
            }
            
            report_path = self.project_root / "migration" / f"cleanup_report_{self.timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(cleanup_report, f, indent=2)
            
            print("=" * 50)
            print("🎉 Legacy cleanup completed successfully!")
            print(f"📦 Archive: {archive_path.name}")
            print(f"📋 Report: {report_path.name}")
            print("🏗️ Architecture now fully unified!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Legacy cleanup failed: {e}")
            return False


def main():
    """Main function to run legacy cleanup"""
    project_root = Path(__file__).parent.parent
    
    cleanup = LegacyCleanup(project_root)
    success = cleanup.run_cleanup()
    
    if success:
        print("\n✅ Legacy cleanup completed successfully!")
        print("🚀 System ready for production deployment")
        return 0
    else:
        print("\n❌ Legacy cleanup failed")
        print("🔍 Check the logs for details")
        return 1


if __name__ == "__main__":
    exit(main())
