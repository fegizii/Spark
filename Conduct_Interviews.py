import operator
from typing import Annotated
from langgraph.graph import MessagesState
from langchain_core.messages import get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from Generate_Analysts import *
import uuid
import markdown

# Load environment variables from the .env file
load_dotenv()

class InterviewState(MessagesState):
    interview_id: str  # ID of the interview
    max_num_questions: int  # Number turns of conversation
    num_responses: int  # Number answers so far
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    conduct_interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")



question_instructions = """You are {name}, an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.

2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


def generate_question(state: InterviewState, llm):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(name=analyst.name, goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Print the generated question
    print(f"\n[Analyst Question]: {question.content}")

    # Write messages to state
    return {"messages": [question]}




# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")


def search_web(state: InterviewState, llm):
    """ Retrieve docs from web search """

    # Web search tool
    tavily_search = TavilySearchResults(max_results=3)

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])

    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState, llm):
    """ Retrieve docs from wikipedia """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state['messages'])

    # Search
    search_docs = WikipediaLoader(query=search_query.search_query,
                                  load_max_docs=2).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 

You goal is to answer a question posed by the interviewer.

To answer question, use this context:

{context}

When answering questions, follow these guidelines:

1. Use only the information provided in the context. 

2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc

6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 

[1] assistant/docs/llama3_1.pdf, page 7 

And skip the addition of the brackets as well as the Document source preamble in your citation."""


def generate_answer(state: InterviewState, llm):
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Print the current context to check if it's changing with each question
    #print("\n[Context Update]:", context)

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Print the answer
    print(f"\n[Expert Answer]: {answer.content}")

    # Increment the number of responses
    state["num_responses"] += 1

    # Return updated state to persist num_responses and append the message
    return {
        "messages": [answer],
        "num_responses": state["num_responses"]
    }


def save_interview(state: InterviewState):
    """ Save interviews """

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"sections": [interview]}


def route_messages(state: InterviewState, name: str = "expert"):
    """ Route between question and answer """
    messages = state["messages"]
    max_num_questions = state.get('max_num_questions')

    # Get the number of responses directly from state instead of recalculating
    num_responses = state["num_responses"]

    print("-" * 100)
    print(f"[Route Decision]: Number of responses so far: {num_responses}")


    # Determine whether to end or continue the interview
    if num_responses >= max_num_questions:
        print("[Route Decision]: Maximum number of turns reached. Saving interview.")
        print("-" * 100)
        return 'save_interview'

    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        print("[Route Decision]: Interview concluded by analyst. Saving interview.")
        print("-" * 100)
        return 'save_interview'

    print("[Route Decision]: Continuing with the next question.")
    print("-" * 100)
    return "ask_question"


section_writer_instructions = """You are an expert technical writer. 

Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents

6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


def write_section(state: InterviewState, llm):
    """ Node to answer a question """

    # Get state
    interview = state["sections"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}, {interview}")])

    # Append it to state
    return {"sections": [section.content]}  # Adds the final report section


# Add nodes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", lambda state: generate_question(state, llm))
interview_builder.add_node("search_web", lambda state: search_web(state, llm))
interview_builder.add_node("search_wikipedia", lambda state: search_wikipedia(state, llm))
interview_builder.add_node("answer_question", lambda state: generate_answer(state, llm))
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", lambda state: write_section(state, llm))

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview
memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct interviews")


def conduct_interview_with_analyst(analyst, topic, max_questions, thread_id, llm_model, llm_temperature):

    interview_id = str(uuid.uuid4())  # Unique ID for each session

    # Print statement to indicate the start of the interview
    print(f"\n")
    print("-" * 100)
    print(f"Beginning interview. Interview ID: {interview_id}")

    # Initialize a fresh list of messages for each interview
    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    config = {"configurable": {"thread_id": thread_id, "session_id": interview_id}}

    global llm
    llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)

    # Start the interview with a fresh state keyed by session_id
    report = interview_graph.invoke({
        "interview_id": interview_id,
        "analyst": analyst,
        "messages": messages,
        "num_responses": 0,  # Initialize to 0 for each new interview
        "max_num_questions": max_questions,
        "context": [],
        "sections": []
    }, config)

    # Convert report to Markdown text
    markdown_content = markdown.markdown(report['sections'][0])

    # Save to Markdown file
    filename = f"interview_report_{analyst.name}.md"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    print(f"Report saved as '{filename}'")
