from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class ResearchAgent():
    """ResearchAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    def __init__(self):
        self.search_tool = SerperDevTool(
        search_url="https://serpapi.com/search.json?engine=google&q=Coffee"
        )
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.5,
        )
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            tools=[self.search_tool],
            llm=self.llm
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], 
            verbose=True,
            llm=self.llm
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
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchAgent crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True
        )
