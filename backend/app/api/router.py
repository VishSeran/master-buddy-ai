from typing import Any
from fastapi import APIRouter,HTTPException,status
from app.schemas.chat import Model,ChatMessage

router = APIRouter()

llm_model = Model()

@router.post("/chat")
async def chat(body:ChatMessage) -> Any:
    try:
        
        response = llm_model.chat(message=body.message)
        return {
            "response" : response
        } 
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    