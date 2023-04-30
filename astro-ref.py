"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.llms import AI21

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import os



# From here down is all the StreamLit UI.
st.set_page_config(page_title="career-advisor", page_icon="::")
st.header("career-advisor")

# Ask the user to enter their  API key
AI21_API_KEY = st.sidebar.text_input("AI21 API-KEY", type="password")

template = """The following is an tense conversation between a really evil urban planner talking and a citizen concerned about the future of their city and communities.
The urban planner does not want to talk about any other thing that urban planning and only provided bad advice that will end up destroying the city. He's somewhat mean and really wants to see the city be destroyed because of bad decisions.

Current conversation:
{history}
Citizen: {input}
Urban planner:"""
prompt_template = PromptTemplate(input_variables=["history", "input"], template=template)

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    #llm = AI21(model="j2-grande-instruct", stop=['Citizen:'])
    llm = AI21(model="j2-jumbo-instruct", stop=['Citizen:']) #TODO: Maybe play with temp?
    chain = ConversationChain(llm=llm,verbose=False, prompt=prompt_template, memory=ConversationBufferMemory(ai_prefix="Urban planner", human_prefix="Citizen"))
    
    
    return chain

if AI21_API_KEY:
    os.environ["AI21_API_KEY"] = AI21_API_KEY
    chain = load_chain()
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')


#chain = load_chain()
from PIL import Image

image = Image.open('images/grey_planner.png')

st.image(image, caption='XX century urban planner')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Citizen: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input and AI21_API_KEY:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="adventurer-neutral", seed="Boo")
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")