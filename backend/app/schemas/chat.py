from pydantic import BaseModel,ConfigDict
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re


class ChatMessage(BaseModel):
    message:str
    
class Model(BaseModel):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_model:object=None
    
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
    
    def chat(self, message:str):
        
        template = """You are a helpful study assistant. give accurate answers descriptively for user questions.
                    user: {question}
                    your response:
        """
        
        prompt = PromptTemplate(template=template,
                                input_variables=['question'])
        
        chain = prompt | self.llm_model | StrOutputParser()
        
        response = chain.invoke(input={"question":message})
        
        match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*)',response,re.DOTALL)
        
        return match.group(1).strip() if match else response.strip()