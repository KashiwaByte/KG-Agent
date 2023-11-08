from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

 
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that offer {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

chain.run("debate competation")


text=chain.run("debate competation")
print(text)
 