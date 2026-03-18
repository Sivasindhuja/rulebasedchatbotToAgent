from crewai import Crew

from agent import hate_speech_detector_agent
from tasks import hate_speech_detection_task

crew = Crew(
    agents=[hate_speech_detector_agent],
    tasks=[hate_speech_detection_task]
)

Text = "I am a good goyim."

result = crew.kickoff(inputs={"text": Text})

print(result)