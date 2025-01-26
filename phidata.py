import os 
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo

# load enviroment variables
load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

##web search agent
web_search_agent= Agent(
    name="AI News Linkedin Curator",
    role="Create a professional Linkedin post about the latest AI developments",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Format news as a compelling Linkedin post"
                  "Include 3-5 key AI news development"
                  "Write in a professional engaging tone"
                  "use bullet points for readability"
                  "Include revelent hastags"
                  "provide source links for creability"
                  "highlight the border impact of ai development"
                  "end with an engagement prompt"
                  "ensure content is suitable for professional networking audience"
                  "keep the total post length under 3000 chracters"],
    show_tools_calls=True,
    markdown=True,
)
# news revelance agent
news_relevance_agent=Agent(
    name="AI News Linkedin Curator",
    role="Create a professional Linkedin post about the latest AI developments",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Carefully assess the generated AI news content",
        "Determine if the content is suitable for LinkedIn posting",
        """Check for:
            - Professionalism
            - Current relevance
            - Potential impact
            - Absence of controversial content""",
        """Provide a structured evaluation with:
            - Suitability score (0-10)
            - Posting recommendation (Yes/No)
            - Specific reasons for evaluation""",
        "If not suitable, explain specific reasons",
        "Suggest potential modifications if needed",
        "Respond with 'No' in the posting recommendation if content is not suitable"
    ],
    show_tools_calls=True,
    markdown=True
)

def main():
    # Generate AI news content
    news_response = web_search_agent.run("5 latest significant AI news developments with sources", stream=False)
    
    # Validate the generated news content
    validation_response = news_relevance_agent.run(
        f"Evaluate the following AI content for LinkedIn posting suitability:\n\n{news_response.content}", 
        stream=False
    )
    
    # Check if validation recommends not posting
    news_content=news_response.content
    if "<function=duckduckgo_news" in validation_response.content:
        news_content =""
    else:
        news_content=news_response.content
    #news_content= "" if "NO" in validation_response.content else news_response.content 
    
    return {
        "news_content": news_content,
        "validation": validation_response.content
    }

if __name__ == '__main__':
    result = main()
    print("Generated News:")
    print(result['news_content'])
    print("\nValidation Result:")
    print(result['validation'])