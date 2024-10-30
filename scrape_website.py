import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
import os
import urllib.parse
from scrapy.crawler import CrawlerProcess
# import logging
# logging.getLogger('scrapy').setLevel(logging.WARNING)
# logging.getLogger('scrapy').propagate = False


class MySpider(CrawlSpider):

    name = "raw_webcrawl_data"
    allowed_domains = ["kmchandy.github.io"]
    start_urls = ["https://kmchandy.github.io/"]

    rules = (
        Rule(LinkExtractor(allow_domains=['kmchandy.github.io']), process_links='process_links', callback='parse_item', follow=True),
        # for images: When `deny_extensions` parameter is not set, it defaults to scrapy.linkextractors.IGNORED_EXTENSIONS, which contains jpeg, png, and other extensions. This means the link extractor avoid links found containing said extensions.
        Rule(LinkExtractor(deny_extensions=set(), tags = ('img',),attrs=('src',),canonicalize = True, unique = True), follow = False, callback='parse_image_link')

    )
    def process_links(self, links):
        for link in links:
            link.url = urllib.parse.urljoin("https://kmchandy.github.io/", link.url)
            yield link
        
    def parse_item(self, response):
        # filename to save
        page = response.url.replace('https://kmchandy.github.io/', '')
        self.log("Saving page: {}".format(page)) 

        filename = os.path.abspath(os.path.join(self.name, page))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # save html file
        with open(filename, 'w') as f:
            f.write(response.text)
        self.log(f"Saved file {filename}")

    def parse_image_link(self, response):
        # filename to save
        page = response.url.replace('https://kmchandy.github.io/', '')
        self.log("Saving page: {}".format(page))
        filename = os.path.abspath(os.path.join(self.name, page))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # save image file
        with open(filename, 'wb') as f:
            f.write(response.body )
        self.log(f"Saved file {filename}") 

def crawl_website():    
    process = CrawlerProcess()
    process.crawl(MySpider)
    process.start()  # the script will block here until the crawling is finished
    return
    
if __name__ == "__main__":
    crawl_website()
