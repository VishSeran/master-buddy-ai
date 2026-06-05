from pydantic import BaseModel
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace


class chatMessage(BaseModel):
    message:str
    
class Model(BaseModel):
    
    def __init__(self, model_name = "SeranVishwa/master-buddy-v1.0",
                 task="text-generation", pipeline_kwargs={
                                                "max_new_tokens":2048,"max_length": None}):
        
        super().__init__()
        
        model = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task=task,
            pipeline_kwargs=pipeline_kwargs
        )
        
        self.llm_model = ChatHuggingFace(llm=model,temperature=1)