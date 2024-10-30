# FAQ parser class
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from typing import List
from pydantic import BaseModel, Field
from enum import Enum
from bs4 import BeautifulSoup, Comment  
import re
import os
import json
import csv
import shutil
import logging 

from dotenv import load_dotenv
load_dotenv()

class QnA(BaseModel):
    question: str = Field(description="one specific question from the given 'faq document'")
    answer: str = Field(description="answer to the above question from the given 'faq document'. Do not generate an answer, simply copy-paste the entire the text of the answer as-is.")

class QnAList(BaseModel):
    faq: List[QnA] = Field(description="list all the question and answer pairs from the given 'document'")

class FAQProcessor():
    def __init__(self):
        
        self.model = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_ORGANIZATION"),
        )

        self.parser = PydanticOutputParser(pydantic_object=QnAList)
        self.fix_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.model)
        self.prompt = PromptTemplate(
            template =  '''
                You are a bot helping with text parsing. 
                Given an 'FAQ document', parse the list of question and answer pairs.\n
                The 'FAQ document' can be noisy, with some unrelated text, make sure to ignore this unrealted text.
                
                {format_instructions}\n
                
                ***
                'FAQ document' : {faq_doc}
                ***


                I am reminding you again, do not add any new questions or facts in asnwers that are not given in the 'FAQ document'.
            ''',
                input_variables=["faq_doc"],
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions(),
                },
            )    
        self.search_conf_thresh = 1
        
    def parse(self, faq_doc):
        response = self.model([HumanMessage(self.prompt.format_prompt(faq_doc=faq_doc).to_string()) ])

        response_output = None
        try:
            response_output = self.parser.parse(response.content)
        except Exception as e:
            response_output = self.fix_parser.parse(response.content)                
        return response_output
        
fp = FAQProcessor()        



#load all the contents of the course
def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text  
    
def remove_html_comments(html_content):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all comment nodes and remove them
    comments = soup.find_all(text=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # Get the HTML content without comments
    html_content_without_comments = str(soup)
    
    return html_content_without_comments

def split_to_sections(html_content):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all headers and their corresponding sections
    header_list = []
    index_list = []
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):    
        header_list.append(header)
        index_list.append(html_content.find(str(header)))
    
    sections = []
    for ind, index in enumerate(index_list[:-1]):
        sections.append(remove_html_comments(html_content[index:index_list[ind+1]]))
    return sections

def extract_image_info(html):
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find all image tags
    img_tags = soup.find_all(['img', 'figure'])

    # Extract image URLs and titles
    image_data = []
    used_img_src = set()
    for tag in img_tags:
        if tag.name == 'img' and tag.get('src') not in used_img_src:
            url = tag.get('src')
            title = tag.get('alt', '')
            image_data.append({'url': url, 'title': title})
            used_img_src.add(url)
            # Remove image tags from HTML
            tag.extract()
        elif tag.name == 'figure':
            figcaption = tag.find('figcaption')
            if figcaption:
                title = figcaption.get_text()
            else:
                title = ''
            img_tag = tag.find('img')
            if img_tag and img_tag.get('src') not in used_img_src:
                url = img_tag.get('src')
                image_data.append({'url': url, 'title': title})
                used_img_src.add(url)
                # Remove image tags from HTML
                tag.extract()

    # Get modified HTML text
    modified_html = str(soup)

    return image_data, modified_html

# Read the HTML file
def read_html(html_filename):
    html_content = None
    with open(html_filename, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


def prefix_image_urls(html_content, prefix_folder):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Define a function to prefix URL
    def prefix_url(url, prefix):
        # If URL is relative, prefix it
        if not url.startswith(('http://', 'https://')):
            return os.path.join(prefix, url)
        return url
    
    # Find all figure tags and images within them
    for figure in soup.find_all('figure'):
        if figure.img and 'src' in figure.img.attrs:
            original_url = figure.img['src']
            figure.img['src'] = prefix_url(original_url, prefix_folder)

    # Find all image tags that are not inside figure tags
    for img in soup.find_all('img'):
        if 'src' in img.attrs and not img.find_parent('figure'):
            original_url = img['src']
            img['src'] = prefix_url(original_url, prefix_folder)

    # Return the modified HTML as a string
    return str(soup)


def parse_html(html_content, html_path):
    html_content = prefix_image_urls(html_content, html_path)
    sections_with_images = split_to_sections(html_content)

    sections_without_images = []
    # Example usage
    for section in sections_with_images:
        image_info, html_without_images = extract_image_info(section)
        sections_without_images.append('\n{}'.format(html_without_images))
        ## [TODO] Solve for images later
        # for info in image_info:
        #     print("\t\tImage Info$$$$$$$$$$$$$:")
        #     print("\t\tURL:", info['url'])
        #     print("\t\tTitle:", info['title'])
    return sections_with_images, sections_without_images


def parse(embedding_function, course_dir='''raw_webcrawl_data'''):
    os.makedirs('./chroma/', exist_ok=True)
    loader = DirectoryLoader(course_dir, glob="**/*.html", loader_cls=TextLoader)
    documents = loader.load()
    print('Loaded {} documents'.format(len(documents)))

    process_faqs = True
    ## output
    faq_processed_dir = '{}_faq_processed'.format(course_dir)
    os.makedirs(faq_processed_dir, exist_ok=True)
    
    
    #save the full text in a different DB for QnA on it
    all_sections = []
    all_sections_html = {}
    course_db = {}
    qna_dict = {}
    docs_procesessed_cnt = 0
    section_procesessed_cnt = 0
    qna_cnt = 0
    faq_cnt = 0
    for doc_cnt, doc in enumerate(documents):
        html_path, html_file = os.path.split(documents[doc_cnt].metadata['source'])
        # print('\n@@@@@@@@@@@@@@@@')
        # print(html_path)
        if 'faq' not in documents[doc_cnt].metadata['source'] and 'FAQ' not in documents[doc_cnt].metadata['source']:
            sections_with_images, sections_without_images = parse_html(documents[doc_cnt].page_content, html_path)
            docs_procesessed_cnt +=1
            for section_cnt, section in enumerate(sections_without_images):
                # save text without html tags for retrieval purposes. embedding model doesnt do good job with html tags
                all_sections.append(Document(page_content=remove_html_tags(section), metadata={"source": documents[doc_cnt].metadata['source'], "split":section_cnt}))
    
                # save text in html format for equations, since equations are written with html+latex tags. ChatGPT does a good job reading and comprehending this.
                if documents[doc_cnt].metadata['source'] not in course_db:
                    course_db[documents[doc_cnt].metadata['source']] = {}
                if section_cnt not in course_db[documents[doc_cnt].metadata['source']]:
                    course_db[documents[doc_cnt].metadata['source']][section_cnt] = sections_with_images[section_cnt]
                    section_procesessed_cnt+=1
        elif process_faqs:
            # load document, and compute QnA pairs using gpt
            faq_list = fp.parse(documents[doc_cnt].page_content)
            faq_cnt += 1
            with open(os.path.join(faq_processed_dir, html_file.replace('.html','.csv')), 'w') as csvfile:
                faqwriter = csv.writer(csvfile)
                for faq in faq_list.faq:
                    faqwriter.writerow([faq.question, faq.answer])
                    qna_dict[faq.question] = [faq.answer]
                    qna_cnt+=1
    
    # save the documents in a dict
    json.dump(course_db, open('./chroma/course_db', 'w'))
    print('Processed {} documents, with {} sections'.format(docs_procesessed_cnt, section_procesessed_cnt))
    
    # split the rest of the documents into chunks that can be handled by embedding model 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256, length_function=len, is_separator_regex=False,)
    
    split_docs = text_splitter.split_documents(all_sections)
    print('Split the {} sections into {} splits'.format(len(all_sections), len(split_docs)))
    
    # load split coursework and fat into separate Chroma search db
    db_dir = "./chroma/db"
    db = Chroma.from_documents(split_docs, embedding_function, collection_name="course", persist_directory=db_dir)
    print('Loaded the splits to search engine')
    
    
    # save the QnA dict
    if process_faqs:
        json.dump(qna_dict, open('./chroma/qna_dict', 'w'))
        db_faq_dir = "./chroma/db_faq"
        db_faq = Chroma.from_texts(list(qna_dict.keys()), embedding_function, collection_name="faq", persist_directory=db_faq_dir)
        print('Processed {} FAQ documents, with {} QnA pairs, and loaded it to search engine.'.format(faq_cnt, qna_cnt))
    return


if __name__ == "__main__":
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    parse(embedding_function)