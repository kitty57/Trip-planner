import streamlit as st
from crewai import Task, Agent, Crew
from textwrap import dedent
from langchain_google_genai import GoogleGenerativeAI
import json
import os
import requests
from bs4 import BeautifulSoup

# Initialize LLM with environment variable for security
llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class TripTasks:
    def identify_task(self, agent, origin, cities, interests, date_range):
        return Task(
            description=dedent(f"""
                Analyze and select the best city for the trip based 
                on weather patterns, seasonal events, and travel costs.
                Provide a detailed report on the chosen city including 
                actual flight costs, weather forecast, and attractions.

                Traveling from: {origin}
                City Options: {cities}
                Trip Date: {date_range}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Detailed report on the best city based on travel costs, weather forecast & attractions."
        )

    def gather_task(self, agent, origin, interests, date_range):
        return Task(
            description=dedent(f"""
                Compile an in-depth guide for the city, including:
                - Key attractions (with hours, fees, and directions)
                - Local customs, special events, cultural hotspots
                - Hidden gems, must-visit landmarks, and cost estimates
                
                Traveling from: {origin}
                Trip Date: {date_range}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Comprehensive city guide including key attractions, customs, events, and cost details."
        )

    def plan_task(self, agent, origin, interests, date_range):
        return Task(
            description=dedent(f"""
                Create a full 7-day itinerary with:
                - Weather forecasts, recommended places to visit/eat
                - Packing suggestions, budget breakdown
                - Actual hotel and restaurant recommendations

                Traveling from: {origin}
                Trip Date: {date_range}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="A complete 7-day travel itinerary with schedule, budget breakdown, and local recommendations."
        )

class TripAgents:
    def city_selection_agent(self):
        return Agent(
            role="City Selection Expert",
            goal="Select the best city based on weather, season, and prices",
            tools=[],
            llm=llm,
            verbose=False
        )

    def local_expert(self):
        return Agent(
            role="Local Expert",
            goal="Provide the best insights about the selected city",
            tools=[],
            llm=llm,
            verbose=False
        )

    def travel_concierge(self):
        return Agent(
            role="Travel Concierge",
            goal="Create detailed travel itineraries including budget and packing suggestions",
            tools=[],
            llm=llm,
            verbose=False
        )

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(city_selector_agent, self.origin, self.cities, self.interests, self.date_range)
        gather_task = tasks.gather_task(local_expert_agent, self.origin, self.interests, self.date_range)
        plan_task = tasks.plan_task(travel_concierge_agent, self.origin, self.interests, self.date_range)

        crew = Crew(
            agents=[city_selector_agent, local_expert_agent, travel_concierge_agent],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        return result

# Streamlit App
st.title("Trip Planner Crew")
location = st.text_input("Where will you be traveling from?")
cities = st.text_input("Which cities are you interested in visiting?")
date_range = st.text_input("What are your travel dates?")
interests = st.text_input("What are your travel interests?")

if st.button("Plan My Trip"):
    trip_crew = TripCrew(location, cities, date_range, interests)
    result = trip_crew.run()
    st.subheader("Here is your Trip Plan:")
    st.text_area("", value=result, height=400)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
