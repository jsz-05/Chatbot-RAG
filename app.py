 # In local run uvicorn app:app --reload

from fastapi import FastAPI, APIRouter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import json
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from scrape_website import crawl_website
from parse_website import parse

class IsAnswerable(Enum):
    YES = "YES - the given 'question' can be confidently answered using the given 'context'"
    NO = "NO - the given 'question' cannot be answered with the given 'context'"  

class AnswerStatus(BaseModel):
    status: IsAnswerable = Field(description="")
    answer: str = Field(description="answer the student's 'question' based solely on the given 'context'. Answer only in HTML format, include images if relevant, and use math style for equations. ")

class FAQBot():
    def __init__(self):

        self.model = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_ORGANIZATION"),
        )

        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory="./chroma/db", embedding_function=self.embedding_function, collection_name="course")
        self.db_faq = Chroma(persist_directory="./chroma/db_faq", embedding_function=self.embedding_function, collection_name="faq")
        self.qna_dict = json.load(open('./chroma/qna_dict'))
        self.course_db = json.load(open('./chroma/course_db'))

        self.parser = PydanticOutputParser(pydantic_object=AnswerStatus)
        self.fix_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.model, max_retries=3)
        self.prompt = PromptTemplate(
            template =  '''
                You're a helpful teaching assistant for a technical course on {course}. You will only answer student's 'question' based on the given 'context' of the course.\n
                The 'context' is a combination of two things - 1) Previous question and answers on the {course} that are similar to the student's question, and 2) some snippets of text from the course contents that are relevant to the student's 'question'.
                
                {format_instructions}\n
                
                ***
                'query' : {question}
                ***
                
                $$$
                'context' : {context}
                $$$
                I am reminding you again, you are a teaching assistant, do not add any facts into the answer that is not given in the 'context'. 
                Answer only in HTML format, include images if relevant, and use math style for equations.
            ''',
                input_variables=["question", "context", "course"],
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions(),
                },
            )    
        self.search_conf_thresh = 1
        self.excuse_me_msg = '''<p>I dont think I know the answer for this, let me check with the professor.</p>'''

        self.router = APIRouter()
        self.router.add_api_route("/ask_question", self.ask_question, methods=["GET"])
        self.router.add_api_route("/refresh_bot", self.update_db, methods=["GET"])
    
    def update_db(self):
        # take the latest dump of the website
        self.db.delete_collection()
        self.db_faq.delete_collection()
        crawl_website()
        parse(embedding_function=self.embedding_function)
        self.db = Chroma(persist_directory="./chroma/db", embedding_function=self.embedding_function, collection_name="course")
        self.db_faq = Chroma(persist_directory="./chroma/db_faq", embedding_function=self.embedding_function, collection_name="faq")
        self.qna_dict = json.load(open('./chroma/qna_dict'))
        self.course_db = json.load(open('./chroma/course_db'))
        
        return
        
    def ask_question(self, question, verbose=False):
        
        retrieved_answers = ''
        ## search in faq
        if verbose:
            print('Search in FAQ')
        results_faq = self.db_faq.similarity_search_with_score(question, k=3)

        ### save only the high confidence search results
        if verbose:
            print('\tanswers retrieved')

        is_faq_title_printed = False
        for i, val in enumerate(results_faq):
            if verbose:
                print('\t\ttext: {}\n\t\tChapter: {}\n\t\tconf:{}\n'.format(val[0].page_content, val[0].metadata, val[1]))

            if val[1] < self.search_conf_thresh: 
                if not is_faq_title_printed:
                    retrieved_answers += '''Question and Answers from the past that are similar to the student's question\n-----------------\n'''
                    is_faq_title_printed = True
                # collect the corresponding answers of the qna pair for gpt 
                retrieved_answers += ' Question:{}\n Answer:{}\n'.format(val[0].page_content, self.qna_dict[val[0].page_content])    



        ## search in coursework
        if verbose:
            print('Search in coursework')            
        results = self.db.similarity_search_with_score(question, k=5)

        ### save only the high confidence search results
        if verbose:
            print('\tanswers retrieved')

        is_snippet_title_printed = False
        max_chapters = 3
        neighboring_sections = 2 # + or -
        chapter_cnt = 0
        seen_chapters = []
        for i, val in enumerate(results):
            if verbose:
                print('\t\ttext: {}\n\t\tChapter: {}\n\t\tSection: {}\n\t\tconf:{}\n'.format(val[0].page_content, val[0].metadata['source'], val[0].metadata['split'], val[1]))
                print(self.course_db[val[0].metadata['source']].keys())
            if val[1] < self.search_conf_thresh: 
                if not is_snippet_title_printed:
                    retrieved_answers += '''\n$$$$$$$$$$\nSnippets of text from the course that are relevant to the student's question\n-----------------\n'''
                    is_snippet_title_printed = True

                if val[0].metadata['source'] not in seen_chapters and chapter_cnt<max_chapters:
                    
                    html_str = self.course_db[val[0].metadata['source']][str((val[0].metadata['split']))]
                    extended_context = ''
                    for ind in range(val[0].metadata['split']-neighboring_sections, val[0].metadata['split']+neighboring_sections):
                        if str(ind) in self.course_db[val[0].metadata['source']]:
                            extended_context += '\n{}'.format(self.course_db[val[0].metadata['source']][str(ind)])
                    
                    retrieved_answers += '\n Relevant text snippet {}: {}\n\n '.format(chapter_cnt, extended_context) 
                    if verbose:
                        print('\n\t\tlength:({}, {})'.format(len(html_str), len(extended_context)))

                    seen_chapters.append(val[0].metadata['source'])
                    chapter_cnt += 1
                    if len(retrieved_answers)>2000:
                        if verbose:
                            print('retrieved_answers length greater than 2000 : {}'.format(len(retrieved_answers)))
                        break
                    



        
        #### if there is atleast one search result ask GPT to answer
        if len(retrieved_answers):
            # ask GPT to answer
            prompt_string = self.prompt.format_prompt(question=question, context=retrieved_answers, course = 'Distributed Algorithms').to_string()

            if verbose:
                print(prompt_string)
            response = self.model([
                HumanMessage(
                    prompt_string
                ) 
            ])

            if verbose:
                print('\t\t\tRaw GPT response: {}\n'.format(response))
                
            faq_response = None
            try:
                faq_response = self.parser.parse(response.content)
            except Exception as e:
                faq_response = self.fix_parser.parse(response.content)                

            if verbose:
                print('\t\t\tfinal response: {}\n'.format(faq_response))

            if faq_response != None and faq_response.status == IsAnswerable.YES:
                return faq_response.answer
            else:
                return self.excuse_me_msg
        else:
            return self.excuse_me_msg


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://0.0.0.0",  # Add 0.0.0.0 as an allowed origin
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
fb = FAQBot()
app.include_router(fb.router)
