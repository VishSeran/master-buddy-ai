from typing import Any
from fastapi import APIRouter,HTTPException,status

router = APIRouter()

@router.post("/chat")
async def chat(message:str) -> Any:
    
    