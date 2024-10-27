from Conduct_Interviews import *
import uuid  #to generate random thread IDs

# create a random thread ID
thread_id = str(uuid.uuid4())

#LLM Inputs
llm_model = "gpt-4o-mini"
llm_temperature = 0.5

# INPUTS
topic = "Key success factors for ice cream business"
max_analysts = 5
max_questions = 5

# Step 1: Generate list of Analysts and save to a JSON file for replicability
analysts = run_graph(topic, max_analysts, llm_model, llm_temperature, thread_id)

# Step 2: The analysts will ask questions to the expert who has access to the web and documents
for analyst in analysts: conduct_interview_with_analyst(analyst, topic, max_questions, thread_id, llm_model, llm_temperature)

