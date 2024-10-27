import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from the .env file
load_dotenv()

# Create analysts and review them using human-in-the-loop
class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


# Create structured output for LLM
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions


create_analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic: {topic}

2. Determine the most interesting themes based upon documents and / or feedback above.

3. Pick the top {max_analysts} themes.

4. Assign one analyst to each theme."""


def create_analysts(state: GenerateAnalystsState, structured_llm):
    """ Create analysts """

    topic = state['topic']
    max_analysts = state['max_analysts']

    # System message
    system_message = create_analyst_instructions.format(
        topic=topic,
        max_analysts=max_analysts
    )

    # Generate analysts
    response = structured_llm.invoke([SystemMessage(content=system_message), HumanMessage(content="Generate the set of analysts.")])

    # Write the list of analysis to state
    return {"analysts": response.analysts}


review_analyst_instructions = """You are tasked with reviewing a set of AI analyst personas that you have previously created. 

Follow these instructions carefully:

1. First, review the list of analysts you have previously created: 
{analysts_to_review}

2. Examine any feedback that has been provided to guide the review of the analysts: 
{human_analyst_feedback}

3. Update the list of analysts according to the feedback provided, making sure to keep the maximum number of analysts to {max_analysts}.

"""


def review_analysts(state: GenerateAnalystsState, structured_llm):
    """ Review analysts """
    analysts_to_review = "\n".join(analyst.persona for analyst in state.get('analysts', []))
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # System message
    system_message = review_analyst_instructions.format(
        analysts_to_review=analysts_to_review,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )

    # Generate analysts
    response = structured_llm.invoke([SystemMessage(content=system_message), HumanMessage(content="Review the list of analysts.")])

    # Write the list of analysis to state
    return {"analysts": response.analysts}


def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass


def should_continue(state: GenerateAnalystsState):
    """ Determines next node (whether to continue with revisions or end the process) """

    human_analyst_feedback = state.get('human_analyst_feedback', None)

    if human_analyst_feedback:
        return "review_analysts"
    return END


# Build the state graph with nodes and edges
builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", lambda state: create_analysts(state, structured_llm))
builder.add_node("review_analysts", lambda state: review_analysts(state, structured_llm))
builder.add_node("human_feedback", human_feedback)

# Define linear flow from start to create analysts, then to human feedback
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["review_analysts", END])
builder.add_edge("review_analysts", "human_feedback")

memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory).with_config(run_name="Generate analysts")




# Run the graph
def run_graph(topic: str, max_analysts: int, llm_model: str, llm_temperature: float, thr: str):

    global structured_llm
    # Initialize the language model with parameters provided by main.py
    llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # Common thread ID
    thread = {"configurable": {"thread_id": thr}}

    # Run the graph to generate the first draft
    for event in graph.stream({"topic": topic, "max_analysts": max_analysts, "human_analyst_feedback": ""}, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            print(f"\n>>>>>>>> FIRST DRAFT")
            for analyst in analysts:
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-" * 50)

    # Enter feedback loop with a maximum number of iterations
    max_feedback_loops = 10
    feedback_count = 0

    while feedback_count < max_feedback_loops:
        # Ask for human feedback
        human_feedback = input(f"Please provide feedback for the analysts (press Enter to end) [{feedback_count + 1}/{max_feedback_loops}]: ")

        if not human_feedback:
            # If no feedback is provided, end the process
            break

        # If feedback is provided, update the state and revise
        graph.update_state(thread, {"human_analyst_feedback": human_feedback}, as_node="human_feedback")

        # Run the graph execution with the updated feedback
        events = list(graph.stream(None, thread, stream_mode="values"))
        last_event = events[-1] if events else None
        if last_event:
            analysts = last_event.get('analysts', '')
            if analysts:
                print(f"\n>>>>>>>> REVISED DRAFT WITH HUMAN FEEDBACK")
                for analyst in analysts:
                    print(f"Name: {analyst.name}")
                    print(f"Affiliation: {analyst.affiliation}")
                    print(f"Role: {analyst.role}")
                    print(f"Description: {analyst.description}")
                    print("-" * 50)

        # Increment feedback count
        feedback_count += 1

    # Final state processing after feedback loop ends
    final_state = graph.get_state(thread)
    analysts = final_state.values.get('analysts')

    print("\n>>>>>>>> FINAL LIST")
    for analyst in analysts:
        print(f"Name: {analyst.name}")
        print(f"Affiliation: {analyst.affiliation}")
        print(f"Role: {analyst.role}")
        print(f"Description: {analyst.description}")
        print("-" * 50)

    # Save the final list of analysts to a JSON file
    analysts_data = [analyst.dict() for analyst in analysts]
    filename = f"Analysts_{thr}.json"

    # Write to a JSON file
    with open(filename, "w") as json_file:
        json.dump(analysts_data, json_file, indent=4)

    print(f"Analysts data saved to {filename}")

    return analysts
