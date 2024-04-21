import os
from typing import List

import openai
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from models.cluster_label import ClusterLabel

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
pydantic_parser = PydanticOutputParser(pydantic_object=ClusterLabel)
output_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=model)


prompt = """\
You are a helpful AI bot. You will be given a list of related sentences, separated by a new line. 
Your job is to generate a short label representative of all the sentences.
The label must be less than 10 words.

{format_instructions}
"""


def extract(texts: List[str]) -> str:
    try:
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt,
                ),
                (
                    "human",
                    f"Here are the sentences:\n\n{os.linesep.join(texts)}",
                ),
            ]
        )
        msgs = template.format_messages(
            format_instructions=output_parser.get_format_instructions()
        )

        res = model.invoke(msgs)
        parsed = output_parser.parse(res.content)

        return parsed.label
    except openai.BadRequestError as e:
        print("OpenAI error:", e)
        return "Default"
    except Exception as e:
        print("Error:", e)
        return "Default"


def main(clusters: List[List[str]]) -> List[str]:
    res = []
    for cluster in clusters:
        res.append(extract(cluster))

    return res
