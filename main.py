from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)


class CAMELAgent:

    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        
        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message

import os





cv = """Francisco Gerardi
franciscogerardi@gmail.com
+37258975744
in/francisco-gerardi/
github.com/frnpnk
Tallinn, Estonia
Summary
Developer, maker, and technologist with hands-on experience in technology, education, and research projects. Deeply passionate about
using technology for social good and contributing to projects that have a positive impact on society and the environment. Skilled in front-
end development, devOps, design thinking, user-centered design, and database development. Recently relocated to Tallinn and is eager to
learn new skills and make a positive impact. Additionally, he have experience with digital fabrication (generative design), Arduino
development, and electronics.
PROFESSIONAL EXPERIENCE
Mad Unicorn Fabrication – Small business owner, December 2018 – Present
Successfully managed the production and sales of open source EFI for cars, generating revenue in Argentina.
Developed and designed a functional WordPress website to market and sell electronic devices, establishing an online presence for
Mad Unicorn Fabrication.
Led a successful pivot to physical interfaces like custom keyboards, leveraging previous experience with suppliers to increase
revenue and profitability.
Designed and developed a Chrome extension called MIDIMapper, which allows users to map web elements to MIDI messages
from a MIDI controller. Targeted the extension to specific use cases, such as controlling Gradio UI for Stable diffusion AI image
generation but that is also suitable for a wide range of other websites
EDUCATION
Front End specialist -Digital house - March 2022 – December 2022
React, redux, TDD, NextJs, Typescript, JUnit, Kubernetes, mongoDB, data analytics, agile methodologies, Jira, learning agility.
Certified tech developer- Digital house - March 2021 – December 2021
HTML, CSS, javascript, react, mySql, aws, docker, oop java, design thinking, UX/UI, Figma, git.
Product Manager - December 2020-February 2021
Design Thinking, Scrum, UX, Prototyping, Customer Development , KPI's and OKR's.
Foundations of Exponential Thinking - Singularity University/UdeSA - October 2020-December 2020
Exponential Leadership, Think Exponentially, Cognitive Biases, Moonshots
INTERESTS/MISCELLANEOUS
3d modeling and generative art/modeling: Solidworks Rhino, Grasshopper, Stable diffusion.
Digital fabrication and robotics
Native Spanish speaker; Fluent in English
Member of LAIA - Argentinian laboratory of artificial intelligence

"""

jobpost = """Mid Front End Engineer
betPawa  Estonia Remote 1 month ago  Over 200 applicants
Full-time · Mid-Senior level

We are looking for a proactive experienced front-end developer to help us conquer new growing markets. If you are bright, willing to face non-trivial challenges, and want to build great responsive UI - it might be you. Our environment is fast-paced, and ever-changing, and will suit someone who has the ability to adapt and think on their feet. In return, you will have the opportunity to work alongside a group of dedicated and smart individuals collaborating to achieve the same goal.

We are looking for a Front-End developer with:

The wish to learn new technologies and learn new frameworks and libraries
Excitement to join a developing company and love a fast-paced, ever-changing environment
Strong knowledge of JavaScript, ES6, HTML, CSS3, and corresponding ecosystems
Knowledge of good practices preferred design patterns and writing idiomatic JavaScript code
Experience with SASS/SCSS
Good English skills

Beneficial skills:

experience with Angular, React, Vue.js (including the main modules of the ecosystem like Vuex, Vue Router, Vuelidate)
previous work with Opera Mini (Presto/Extreme mode)
experience with project management systems (Jira)
experience in configuring assemblies (Webpack)
Agile
"""

os.environ["OPENAI_API_KEY"] = ""

assistant_role_name = "IT recruiter"
user_role_name = "HR consultant"
task = "Analize and define the percentage of fit and posible impovments between a cv and a jobpost"
word_limit = 50 # word limit for task brainstorming

task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
"""Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                             user_role_name=user_role_name,
                                                             task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content

assistant_inception_prompt = (
"""Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the cv:{cv}
Here is the jobpost: {jobpost}
Here is the task: {task}. Never forget our task!
I must instruct you based on your expertise and my needs to complete the task.

I must give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions you only answer questions.
You are never supposed to reply with a flake solution. Explain your solutions.
Your solution must be declarative sentences and simple present tense.
Unless I say the task is completed, you should always start with:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for task-solving.
Always end <YOUR_SOLUTION> with: Next request."""
)

user_inception_prompt = (
"""Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the cv:{cv}
Here is the jobpost: {jobpost}
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
)

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str, cv:str, jobpost:str):
    
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task, cv=cv, jobpost=jobpost)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task, cv=cv, jobpost=jobpost)[0]
    
    return assistant_sys_msg, user_sys_msg

assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task, cv, jobpost)

assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats 
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
                "Now start to give me introductions one by one. "
                "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)

print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

chat_turn_limit, n = 5, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    
    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break