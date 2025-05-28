# build_runtime_llm_chain.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from prompt_update import SYS_MSG, PAT_TPL

def build_runtime_llm_chain(pattern_id: int, model_name="gpt-4o", temperature=0.5) -> LLMChain:
    assert 0 <= pattern_id < len(PAT_TPL), f"No template for pattern {pattern_id}"
    
    system_message = SystemMessage(content=SYS_MSG)
    human_message = HumanMessagePromptTemplate.from_template(PAT_TPL[pattern_id])

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    return LLMChain(llm=llm, prompt=chat_prompt)
