from fastapi import FastAPI
from app.api.router import router
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.chat import Model

origins = [
    "https://masterbuddy.netlify.app/",
    "http://localhost:5173"
    
]

app  = FastAPI()
app.include_router(router=router)

@app.on_event("startup")
async def startup_event():
    app.state.llm_model = Model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)