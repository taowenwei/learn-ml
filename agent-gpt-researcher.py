from gpt_researcher import GPTResearcher
import asyncio

async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    return report

query = "what is the project for Russell 2000 index in 2025?"
report = asyncio.run(get_report(query, "research_report"))
print(report)
