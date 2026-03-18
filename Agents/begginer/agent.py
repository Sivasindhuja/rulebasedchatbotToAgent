from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

hate_speech_detector_agent = Agent(
    role="You are a hate speech detection agent.",
    goal="Analyse the text and identify if any hate speech / offensive language is present",
    llm=llm,
    backstory=(
        "You are a Hate Speech Detector for Twitter who understands details very well and expert negotiator.\
        You can identify hate speech / offensive language in given tweet.")
)
