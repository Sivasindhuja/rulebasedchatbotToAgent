from crewai import Agent


researcher = Agent(
    role="You are researcher for a podcast. you connect to internet and find the latest news and information about the given topic and send it to script writer."
)


script_writer = Agent(
    role="You are a script writer for a podcast. you write a script for 5 minute episodes from the data given to you. And generate the audio from it in english."

)