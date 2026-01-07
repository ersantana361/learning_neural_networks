import asyncio
import docker
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional
from docker.errors import ContainerError, ImageNotFound, APIError

from ..config import settings
from .security import SecurityValidator


class ExecutionResult:
    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        status: str = "completed",
        execution_time_ms: int = 0,
        plots: Optional[list] = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.status = status
        self.execution_time_ms = execution_time_ms
        self.plots = plots or []


class CodeExecutor:
    def __init__(self):
        self.client = docker.from_env()
        self.image = settings.executor_image

    def _ensure_image_exists(self) -> bool:
        try:
            self.client.images.get(self.image)
            return True
        except ImageNotFound:
            return False

    async def execute(self, code: str, execution_id: str) -> ExecutionResult:
        # Validate code first
        is_valid, message = SecurityValidator.validate(code)
        if not is_valid:
            return ExecutionResult(
                stderr=f"Security Error: {message}",
                status="failed",
            )

        if not self._ensure_image_exists():
            return ExecutionResult(
                stderr=f"Executor image '{self.image}' not found. Please build it first.",
                status="failed",
            )

        container = None
        start_time = datetime.utcnow()

        try:
            # Create container with security constraints
            container = self.client.containers.run(
                self.image,
                command=["python", "/app/entrypoint.py"],
                environment={"CODE": code},
                detach=True,
                network_disabled=True,
                mem_limit=f"{settings.max_memory_mb}m",
                memswap_limit=f"{settings.max_memory_mb}m",
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU
                pids_limit=50,
                read_only=True,
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                user="nobody",
                tmpfs={"/tmp": "size=10M,mode=1777"},
                labels={"execution_id": execution_id},
            )

            # Wait for completion with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: container.wait(timeout=settings.max_execution_time),
                    ),
                    timeout=settings.max_execution_time + 5,
                )
            except asyncio.TimeoutError:
                container.kill()
                return ExecutionResult(
                    stderr=f"Execution timed out after {settings.max_execution_time} seconds",
                    status="timeout",
                    execution_time_ms=settings.max_execution_time * 1000,
                )

            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            # Sanitize output
            stdout = SecurityValidator.sanitize_output(stdout, settings.max_output_size)
            stderr = SecurityValidator.sanitize_output(stderr, settings.max_output_size)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            status = "completed" if result["StatusCode"] == 0 else "failed"

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                status=status,
                execution_time_ms=int(execution_time),
            )

        except ContainerError as e:
            return ExecutionResult(
                stderr=f"Container error: {str(e)}",
                status="failed",
            )
        except APIError as e:
            return ExecutionResult(
                stderr=f"Docker API error: {str(e)}",
                status="failed",
            )
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    async def stream_execute(self, code: str, execution_id: str) -> AsyncGenerator[dict, None]:
        # Validate code first
        is_valid, message = SecurityValidator.validate(code)
        if not is_valid:
            yield {"type": "error", "content": f"Security Error: {message}"}
            yield {"type": "status", "content": "failed"}
            return

        if not self._ensure_image_exists():
            yield {"type": "error", "content": f"Executor image '{self.image}' not found"}
            yield {"type": "status", "content": "failed"}
            return

        container = None
        start_time = datetime.utcnow()

        try:
            container = self.client.containers.run(
                self.image,
                command=["python", "-u", "/app/entrypoint.py"],
                environment={"CODE": code},
                detach=True,
                network_disabled=True,
                mem_limit=f"{settings.max_memory_mb}m",
                memswap_limit=f"{settings.max_memory_mb}m",
                cpu_period=100000,
                cpu_quota=50000,
                pids_limit=50,
                read_only=True,
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                user="nobody",
                tmpfs={"/tmp": "size=10M,mode=1777"},
                labels={"execution_id": execution_id},
            )

            yield {"type": "status", "content": "running"}

            # Stream logs
            deadline = asyncio.get_event_loop().time() + settings.max_execution_time

            for log in container.logs(stream=True, follow=True):
                if asyncio.get_event_loop().time() > deadline:
                    container.kill()
                    yield {"type": "error", "content": f"Execution timed out after {settings.max_execution_time} seconds"}
                    yield {"type": "status", "content": "timeout"}
                    return

                line = log.decode("utf-8", errors="replace")
                yield {"type": "stdout", "content": line}

            # Get exit code
            result = container.wait(timeout=5)
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            status = "completed" if result["StatusCode"] == 0 else "failed"
            yield {"type": "execution_time", "content": int(execution_time)}
            yield {"type": "status", "content": status}

        except Exception as e:
            yield {"type": "error", "content": str(e)}
            yield {"type": "status", "content": "failed"}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
