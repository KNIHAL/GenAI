import sys
import warnings
from datetime import datetime
from research_agent.crew import ResearchAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew in a continuous loop for multiple queries.
    """
    print("Research Agent is ready! Type 'exit' to quit the program.")
    
    while True:
        # Get user input
        topic = input("\nWhat would you like to research? ").strip()
        
        # Check for exit condition
        if topic.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if not topic:
            print("Please enter a valid research topic.")
            continue
            
        inputs = {
            'topic': topic,
            'current_year': str(datetime.now().year)
        }

        try:
            print(f"\nResearching: {topic}...\n")
            ResearchAgent().crew().kickoff(inputs=inputs)
        except Exception as e:
            print(f"An error occurred while running the crew: {e}")
            print("Please try again with a different query.")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        ResearchAgent().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ResearchAgent().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        ResearchAgent().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
