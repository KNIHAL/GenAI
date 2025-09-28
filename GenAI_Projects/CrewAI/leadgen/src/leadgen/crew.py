import os
from datetime import datetime
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import ApifyActorsTool
from langchain_groq import ChatGroq

@CrewBase
class Leadgen:
    """Leadgen crew for finding and analyzing potential leads"""

    search_tool = ApifyActorsTool(
        actor_name="code_crafter/apollo-io-scraper"
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("MODEL", "llama3-8b-8192")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    self.groq_client = ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=0.1
    )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[self.search_tool],
            llm=self.groq_client
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            llm=self.groq_client
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            context=[self.research_task()],
            output_file=f'reports/lead_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Leadgen crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
