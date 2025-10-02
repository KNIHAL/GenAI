#!/usr/bin/env python
import sys
import warnings
import os
from datetime import datetime
from crew import Leadgen

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """Run the lead generation crew."""
    
    # More comprehensive inputs
    inputs = {
        'topic': 'AI Agents',
        'role': 'AI Agent Development',
        'location': 'Remote',  # Add location filter
        'budget_min': 1000,    # Add budget filters
        'max_results': 50      # Limit results
    }
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    try:
        print(f"🚀 Starting Lead Generation for {inputs['topic']} - {inputs['role']}")
        result = Leadgen().crew().kickoff(inputs=inputs)
        print("✅ Lead generation completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
