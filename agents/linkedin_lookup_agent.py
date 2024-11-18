import os
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub

load_dotenv()

def lookup(name: str) -> str:
    hugging_face_token=os.getenv("HUGGINGFACE_TOKEN")
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=1000,
        huggingfacehub_api_token=hugging_face_token
    )

    template = """Given the full name {name_of_person}, I want you to find a link to their LinkedIn profile page using the provided tool.  Cant be pub/dir, it has to be profile, not search page. Answer with pub/dir/ will not be accepted.
Your answer should contain only a URL."""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google for LinkedIn profile page",
            func=get_profile_url_tavily,
            description="Searches for a LinkedIn profile page using a full name. Provide a full name as input.",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format(name_of_person=name)}
    )
    linkedin_profile_url = result["output"]
    return linkedin_profile_url

if __name__ == "__main__":
    linkedin_url = lookup(name="Radoslaw Mical")
    print(linkedin_url)