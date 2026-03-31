from crewai.flow.flow import Flow, start, listen, router
from crewai import Crew, Process, Task
from pydantic import BaseModel

from therapy_multi.crew import TherapyMulti


SAFETY_RESPONSE = (
    "I can see you're going through something very heavy. "
    "Are you thinking about self-harm? I'm here to listen."
)

AGENT_ROUTE_NAMES = [
    "normalizing", "reflection", "questioning",
    "solutions", "psychoeducation", "holding_space", "recognition",
]


class TherapyMultiState(BaseModel):
    user_message: str = ""
    conversation_history: str = ""
    chosen_agent: str = ""
    response: str = ""


class TherapyMultiFlow(Flow[TherapyMultiState]):

    def __init__(self):
        super().__init__()
        self._crew = TherapyMulti()  

    @start()
    def route(self):
        inputs = {
            "user_message": self.state.user_message,
            "conversation_history": self.state.conversation_history,
        }
        router_crew = Crew(
            agents=[self._crew.router_agent()],
            tasks=[self._crew.router_task()],
            process=Process.sequential,
            verbose=False,
        )
        result = router_crew.kickoff(inputs=inputs)
        self.state.chosen_agent = result.raw.strip().lower()

    @router(route)
    def dispatch(self):
        chosen = self.state.chosen_agent
        if chosen == "safety_screening":
            return "safety"
        if chosen not in AGENT_ROUTE_NAMES:
            self.state.chosen_agent = "holding_space"
            return "holding_space"
        return chosen
    
    def _respond(self, agent_key: str):
        agent = self._crew._get_agent_by_name(agent_key)
        task = Task(
            config=self._crew.tasks_config["response_task"],
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff(inputs={
            "user_message": self.state.user_message,
            "conversation_history": self.state.conversation_history,
        })
        self.state.response = result.raw
        self.state.chosen_agent = agent_key

    @listen("safety")
    def handle_safety(self):
        self.state.response = SAFETY_RESPONSE
        self.state.chosen_agent = "safety_screening"

    @listen("holding_space")
    def handle_holding_space(self):
        self._respond("holding_space")

    @listen("normalizing")
    def handle_normalizing(self):
        self._respond("normalizing")

    @listen("reflection")
    def handle_reflection(self):
        self._respond("reflection")

    @listen("questioning")
    def handle_questioning(self):
        self._respond("questioning")

    @listen("solutions")
    def handle_solutions(self):
        self._respond("solutions")

    @listen("psychoeducation")
    def handle_psychoeducation(self):
        self._respond("psychoeducation")

    @listen("recognition")
    def handle_recognition(self):
        self._respond("recognition")


def run_turn(user_message: str, history: list[dict]) -> dict:
    flow = TherapyMultiFlow()
    flow.kickoff(inputs={
        "user_message": user_message,
        "conversation_history": _format_history(history),
    })
    return {
        "agent_used": flow.state.chosen_agent,
        "response": flow.state.response,
    }


def _format_history(history: list[dict]) -> str:
    if not history:
        return "No prior conversation."
    lines = []
    for turn in history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)