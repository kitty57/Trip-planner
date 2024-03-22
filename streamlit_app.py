import streamlit as st
from crewai import Task
from textwrap import dedent
from datetime import date
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import tool
from unstructured.partition.html import partition_html
import json
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st

llm = GoogleGenerativeAI(
           model="gemini-pro",
           google_api_key='AIzaSyDKcxALky8LiROaxb0RGMw8TLLOcujMRMY'
           )

class TripTasks():

  def identify_task(self, agent, origin, cities, interests, range):
    return Task(description=dedent(f"""
        Analyze and select the best city for the trip based 
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple cities, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses. 
        
        Your final answer must be a detailed
        report on the chosen city, and everything you found out
        about it, including the actual flight costs, weather 
        forecast and attractions.
        {self.__tip_section()}

        Traveling from: {origin}
        City Options: {cities}
        Trip Date: {range}
        Traveler Interests: {interests}
      """),
                agent=agent,
                expected_output='detailed report on chosen city including flight costs, weather forecast, & top attractions aligned with interests ({interests})'
)

  def gather_task(self, agent, origin, interests, range):
    return Task(description=dedent(f"""
        As a local expert on this city you must compile an 
        in-depth guide for someone traveling there and wanting 
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what 
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forcast, and
        high level costs.
        
        The final answer must be a comprehensive city guide, 
        rich in cultural insights and practical tips, 
        tailored to enhance the travel experience.
        {self.__tip_section()}

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
      """),
                agent=agent,
                expected_output='''detailed city guide including key attractions (5-10 w/ opening hours, fees, & directions), local customs, special events, daily itinerary
                 (interests-based activities), hidden gems (2-3), cultural hotspots, must-visit landmarks, weather forecast, & high-level costs'''
                 )

  def plan_task(self, agent, origin, interests, range):
    return Task(description=dedent(f"""
        Expand this guide into a a full 7-day travel 
        itinerary with detailed per-day plans, including 
        weather forecasts, places to eat, packing suggestions, 
        and a budget breakdown.
        
        You MUST suggest actual places to visit, actual hotels 
        to stay and actual restaurants to go to.
        
        This itinerary should cover all aspects of the trip, 
        from arrival to departure, integrating the city guide
        information with practical travel logistics.
        
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give a reason why you picked
        up each place and what makes them special! {self.__tip_section()}

        Trip Date: {range}
        Traveling from: {origin}
        Traveler Interests: {interests}
      """),
                agent=agent,
                expected_output='''complete 7-day travel itinerary w/ daily plans (weather, places to visit/eat, packing list), specific hotels & restaurants, budget breakdown,
                 formatted in markdown (schedule, clothing, detailed budget) ensuring THE BEST TRIP EVER!'''
)

  def __tip_section(self):
    return "If you do your BEST WORK, I'll tip you $100!"

class BrowserTools:

    @tool("Scrape website content")
    def scrape_and_summarize_website(self, website):
        """Useful to scrape and summarize a website content"""
        response = requests.get(website)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content from relevant elements (replace with your logic)
        text_content = []
        for p in soup.find_all('p'):
            text_content.append(p.text.strip())

        content = "\n\n".join(text_content)
        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]

        summaries = []
        for chunk in content:
            agent = Agent(
                role='Principal Researcher',
                goal='Do amazing researches and summaries based on the content you are working with',
                backstory="You're a Principal Researcher at a big company and you need to do a research about a given topic.",
                allow_delegation=False)
            task = Task(
                agent=agent,
                description=f'Analyze and summarize the content bellow, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
            )
            summary = task.execute()
            summaries.append(summary)
        return "\n\n".join(summaries)

class SearchTools():

  @tool("Search the internet")
  def search_internet(query):
    """Useful to search the internet 
    about a a given topic and return relevant results"""
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': 'e252cfdebb208ef5e1bd09d9d69017c0022b35c6',
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    stirng = []
    for result in results[:top_result_to_return]:
      try:
        stirng.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(stirng)

class TripAgents():

  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=False,
        llm=llm)

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        llm=llm,
        verbose=False)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        llm=llm,
        verbose=False)

class TripCrew:

  def __init__(self, origin, cities, date_range, interests):
    self.cities = cities
    self.origin = origin
    self.interests = interests
    self.date_range = date_range

  def run(self):
    agents = TripAgents()
    tasks = TripTasks()

    city_selector_agent = agents.city_selection_agent()
    local_expert_agent = agents.local_expert()
    travel_concierge_agent = agents.travel_concierge()

    identify_task = tasks.identify_task(
      city_selector_agent,
      self.origin,
      self.cities,
      self.interests,
      self.date_range
    )
    gather_task = tasks.gather_task(
      local_expert_agent,
      self.origin,
      self.interests,
      self.date_range
    )
    plan_task = tasks.plan_task(
      travel_concierge_agent, 
      self.origin,
      self.interests,
      self.date_range
    )

    crew = Crew(
      agents=[
        city_selector_agent, local_expert_agent, travel_concierge_agent
      ],
      tasks=[identify_task, gather_task, plan_task],
      verbose=True
    )

    result = crew.kickoff()
    return result

# Define Streamlit app layout
st.title('Trip Planner Crew')
location = st.text_input('From where will you be traveling from?')
cities = st.text_input('What are your cities options you are interested in visiting?')
date_range = st.text_input('What is the date range you are interested in traveling?')
interests = st.text_input('What are some of your high level interests and hobbies?')

# Define TripCrew object and run the planner
if st.button('Plan My Trip'):
    trip_crew = TripCrew(location, cities, date_range, interests)
    result = trip_crew.run()
    st.subheader('Here is your Trip Plan:')
    st.text_area('', value=result, height=400, max_chars=None, key=None)

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(layout='wide')

