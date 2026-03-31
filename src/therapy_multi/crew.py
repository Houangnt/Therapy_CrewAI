from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class TherapyMulti():

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def router_agent(self) -> Agent:
        return Agent(config=self.agents_config["router_agent"])

    @agent
    def normalizing_agent(self) -> Agent:
        return Agent(config=self.agents_config["normalizing_agent"])

    @agent
    def reflection_agent(self) -> Agent:
        return Agent(config=self.agents_config["reflection_agent"])

    @agent
    def questioning_agent(self) -> Agent:
        return Agent(config=self.agents_config["questioning_agent"])

    @agent
    def solutions_agent(self) -> Agent:
        return Agent(config=self.agents_config["solutions_agent"])

    @agent
    def psychoeducation_agent(self) -> Agent:
        return Agent(config=self.agents_config["psychoeducation_agent"])

    @agent
    def recognition_agent(self) -> Agent:
        return Agent(config=self.agents_config["recognition_agent"])

    @agent
    def holding_space_agent(self) -> Agent:
        return Agent(config=self.agents_config["holding_space_agent"])

    @task
    def router_task(self) -> Task:
        return Task(config=self.tasks_config["router_task"])

    @task
    def response_task(self) -> Task:
        return Task(config=self.tasks_config["response_task"])


    def _get_agent_by_name(self, name: str) -> Agent:
        agent_map = {
            "normalizing":     self.normalizing_agent(),
            "reflection":      self.reflection_agent(),
            "questioning":     self.questioning_agent(),
            "solutions":       self.solutions_agent(),
            "psychoeducation": self.psychoeducation_agent(),
            "holding_space":   self.holding_space_agent(),
            "recognition":     self.recognition_agent(),
        }
        return agent_map.get(name, self.holding_space_agent())