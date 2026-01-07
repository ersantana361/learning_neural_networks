from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime
import uuid
import json

from ..database import get_db
from ..models import Execution
from ..services import CodeExecutor

router = APIRouter(prefix="/api/execute", tags=["execute"])
executor = CodeExecutor()


class ExecuteRequest(BaseModel):
    code: str
    session_id: str
    lecture_id: int | None = None


class ExecuteResponse(BaseModel):
    execution_id: str
    status: str
    stdout: str
    stderr: str
    execution_time_ms: int


@router.post("")
async def execute_code(
    request: ExecuteRequest,
    db: AsyncSession = Depends(get_db),
) -> ExecuteResponse:
    execution_id = str(uuid.uuid4())

    # Create execution record
    execution = Execution(
        id=uuid.UUID(execution_id),
        session_id=request.session_id,
        lecture_id=request.lecture_id,
        code=request.code,
        status="running",
    )
    db.add(execution)
    await db.commit()

    # Execute code
    result = await executor.execute(request.code, execution_id)

    # Update execution record
    execution.status = result.status
    execution.stdout = result.stdout
    execution.stderr = result.stderr
    execution.execution_time_ms = result.execution_time_ms
    execution.completed_at = datetime.utcnow()
    await db.commit()

    return ExecuteResponse(
        execution_id=execution_id,
        status=result.status,
        stdout=result.stdout,
        stderr=result.stderr,
        execution_time_ms=result.execution_time_ms,
    )


@router.websocket("/ws")
async def websocket_execute(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            code = message.get("code", "")
            session_id = message.get("session_id", "anonymous")
            execution_id = str(uuid.uuid4())

            async for output in executor.stream_execute(code, execution_id):
                await websocket.send_json(output)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
