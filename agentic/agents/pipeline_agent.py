"""
Pipeline Management Agent

Handles the data processing pipeline operations including Discord message fetching,
embedding, resource detection, and database management.
"""

import logging
import subprocess
import asyncio
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineAgent:
    """
    Agent responsible for managing the data processing pipeline.
    
    Integrates the legacy pipeline components with the agentic system:
    - Discord message fetching
    - Message embedding and storage
    - Resource detection and classification
    - Database synchronization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path(config.get("base_path", "."))
        self.log_path = self.base_path / "logs" / "pipeline.log"
        self.db_path = self.base_path / "data" / "discord_messages.db"
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Pipeline step configurations
        self.pipeline_steps = [
            {
                "name": "database_population",
                "command": "python3 scripts/database/populate_database.py",
                "description": "Run complete database population",
                "timeout": 1800
            }
        ]
        
        self.is_running = False
        self.current_step = None
        self.pipeline_history = []
        
        logger.info("PipelineAgent initialized")
    
    async def run_full_pipeline(self, user_id: str = "system") -> Dict[str, Any]:
        """
        Run the complete data processing pipeline.
        
        Args:
            user_id: User ID who initiated the pipeline
            
        Returns:
            Pipeline execution results
        """
        if self.is_running:
            return {
                "success": False,
                "error": "Pipeline is already running",
                "current_step": self.current_step
            }
        
        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.is_running = True
            self._reset_log()
            
            result = {
                "success": True,
                "pipeline_id": pipeline_id,
                "initiated_by": user_id,
                "start_time": datetime.utcnow().isoformat(),
                "steps": [],
                "stats": {}
            }
            
            logger.info(f"Starting pipeline {pipeline_id} initiated by {user_id}")
            
            # Execute each pipeline step
            for step_config in self.pipeline_steps:
                step_result = await self._execute_step(step_config)
                result["steps"].append(step_result)
                
                if not step_result["success"]:
                    result["success"] = False
                    result["failed_step"] = step_config["name"]
                    break
            
            # Get final database statistics
            result["stats"] = await self._get_db_stats()
            result["end_time"] = datetime.utcnow().isoformat()
            
            # Store pipeline history
            self.pipeline_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "success": False,
                "pipeline_id": pipeline_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        finally:
            self.is_running = False
            self.current_step = None
    
    async def run_single_step(self, step_name: str, user_id: str = "system") -> Dict[str, Any]:
        """
        Run a single pipeline step.
        
        Args:
            step_name: Name of the step to run
            user_id: User ID who initiated the step
            
        Returns:
            Step execution results
        """
        if self.is_running:
            return {
                "success": False,
                "error": "Pipeline is already running",
                "current_step": self.current_step
            }
        
        # Find step configuration
        step_config = None
        for config in self.pipeline_steps:
            if config["name"] == step_name:
                step_config = config
                break
        
        if not step_config:
            return {
                "success": False,
                "error": f"Unknown step: {step_name}",
                "available_steps": [s["name"] for s in self.pipeline_steps]
            }
        
        try:
            self.is_running = True
            self._reset_log()
            
            logger.info(f"Running single step {step_name} initiated by {user_id}")
            
            step_result = await self._execute_step(step_config)
            step_result["initiated_by"] = user_id
            step_result["stats"] = await self._get_db_stats()
            
            return step_result
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return {
                "success": False,
                "step": step_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        finally:
            self.is_running = False
            self.current_step = None
    
    async def _execute_step(self, step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step_name = step_config["name"]
        command = step_config["command"]
        timeout = step_config.get("timeout", 300)
        
        self.current_step = step_name
        
        logger.info(f"Executing step: {step_name}")
        
        try:
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.base_path
            )
            
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return_code = process.returncode
                output = stdout.decode('utf-8') if stdout else ""
                
                # Log output
                self._log_step_output(step_name, output)
                
                if return_code == 0:
                    logger.info(f"Step completed successfully: {step_name}")
                    return {
                        "success": True,
                        "step": step_name,
                        "return_code": return_code,
                        "output_length": len(output),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    logger.error(f"Step failed: {step_name} (exit code {return_code})")
                    return {
                        "success": False,
                        "step": step_name,
                        "return_code": return_code,
                        "error": f"Command failed with exit code {return_code}",
                        "output": output[-1000:],  # Last 1000 chars for debugging
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.error(f"Step timed out: {step_name}")
                return {
                    "success": False,
                    "step": step_name,
                    "error": f"Step timed out after {timeout} seconds",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error executing step {step_name}: {e}")
            return {
                "success": False,
                "step": step_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            if not self.db_path.exists():
                return {"error": "Database file not found"}
            
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            
            # Get message count
            cur.execute("SELECT COUNT(*) FROM messages")
            message_count = cur.fetchone()[0]
            
            # Get resource count
            cur.execute("SELECT COUNT(*) FROM resources") 
            resource_count = cur.fetchone()[0]
            
            # Get latest message timestamp
            cur.execute("SELECT MAX(timestamp) FROM messages")
            latest_message = cur.fetchone()[0]
            
            conn.close()
            
            return {
                "total_messages": message_count,
                "total_resources": resource_count,
                "latest_message": latest_message,
                "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting DB stats: {e}")
            return {"error": str(e)}
    
    def _reset_log(self):
        """Reset the pipeline log file."""
        try:
            with open(self.log_path, "w") as f:
                f.write(f"Pipeline log started at {datetime.utcnow().isoformat()}\n")
        except Exception as e:
            logger.error(f"Error resetting log: {e}")
    
    def _log_step_output(self, step_name: str, output: str):
        """Log step output to file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(f"\n=== {step_name} ===\n")
                f.write(output)
                f.write(f"\n=== End {step_name} ===\n\n")
        except Exception as e:
            logger.error(f"Error logging step output: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "is_running": self.is_running,
            "current_step": self.current_step,
            "available_steps": [s["name"] for s in self.pipeline_steps],
            "log_path": str(self.log_path),
            "db_path": str(self.db_path)
        }
    
    def get_pipeline_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent pipeline execution history."""
        return self.pipeline_history[-limit:]
    
    async def get_pipeline_logs(self, lines: int = 100) -> Dict[str, Any]:
        """Get recent pipeline logs."""
        try:
            if not self.log_path.exists():
                return {
                    "success": False,
                    "error": "Log file not found"
                }
            
            # Read last N lines from log file
            with open(self.log_path, "r") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            return {
                "success": True,
                "lines": recent_lines,
                "total_lines": len(all_lines),
                "log_path": str(self.log_path)
            }
            
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return {
                "success": False,
                "error": str(e)
            }
