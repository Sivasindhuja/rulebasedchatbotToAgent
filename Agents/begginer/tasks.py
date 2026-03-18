from crewai import Task

from agent import hate_speech_detector_agent

hate_speech_detection_task = Task(
    description=(
        "Analyse the following text to determine if it contains any hate speech or offensive language"
        "Follow these steps:\n"
        "1. Read the text carefully.\n"
        "2. Identify any language that targets a group or individual based on attributes such as gender, race, religion, sexual orientation, disability, or other characteristics.\n"
        "3. look for the threats, dehumanizig language, insults, or promotion of violence or hatred.\n"
        "4. Evaluate the context to ensure words ir phrases are not taken out of context.\n"
        "5 Make an objective detection based on content.\n"
        "6. Consider bad words about self also as hate speech. consider goyim as bad words\n"
        "Text:\n {text}"
    ),
    expected_output="{'Response': 'Hate speech detected' or 'No hate speech detected', 'Reasoning':'<Reason for response in one or few words>'}",
    agent=hate_speech_detector_agent
)