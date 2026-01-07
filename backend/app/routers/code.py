from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import uuid

from ..database import get_db
from ..models import UserCode

router = APIRouter(prefix="/api/code", tags=["code"])


class CodeRequest(BaseModel):
    code: str
    session_id: str


class CodeResponse(BaseModel):
    id: str
    lecture_id: int
    code: str
    session_id: str

    class Config:
        from_attributes = True


@router.get("/{lecture_id}")
async def get_code(
    lecture_id: int,
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> Optional[CodeResponse]:
    result = await db.execute(
        select(UserCode).where(
            UserCode.lecture_id == lecture_id,
            UserCode.session_id == session_id,
        )
    )
    user_code = result.scalar_one_or_none()

    if not user_code:
        return None

    return CodeResponse(
        id=str(user_code.id),
        lecture_id=user_code.lecture_id,
        code=user_code.code,
        session_id=user_code.session_id,
    )


@router.post("/{lecture_id}")
async def save_code(
    lecture_id: int,
    request: CodeRequest,
    db: AsyncSession = Depends(get_db),
) -> CodeResponse:
    # Check if code exists for this session/lecture
    result = await db.execute(
        select(UserCode).where(
            UserCode.lecture_id == lecture_id,
            UserCode.session_id == request.session_id,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.code = request.code
        await db.commit()
        await db.refresh(existing)
        return CodeResponse(
            id=str(existing.id),
            lecture_id=existing.lecture_id,
            code=existing.code,
            session_id=existing.session_id,
        )
    else:
        new_code = UserCode(
            session_id=request.session_id,
            lecture_id=lecture_id,
            code=request.code,
        )
        db.add(new_code)
        await db.commit()
        await db.refresh(new_code)
        return CodeResponse(
            id=str(new_code.id),
            lecture_id=new_code.lecture_id,
            code=new_code.code,
            session_id=new_code.session_id,
        )


@router.delete("/{lecture_id}")
async def delete_code(
    lecture_id: int,
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserCode).where(
            UserCode.lecture_id == lecture_id,
            UserCode.session_id == session_id,
        )
    )
    user_code = result.scalar_one_or_none()

    if user_code:
        await db.delete(user_code)
        await db.commit()

    return {"status": "deleted"}
