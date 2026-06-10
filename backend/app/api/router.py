from typing import Any
from fastapi import APIRouter,HTTPException,status
from app.schemas.chat import Model,ChatMessage
from fastapi import Request

router = APIRouter()


@router.post("/chat")
async def chat(body:ChatMessage, request:Request) -> Any:
    try:
        
        response = request.app.state.llm_model.chat(message=body.message)
        return {
            "response" : response
        } 
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    