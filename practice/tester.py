
#from langchain.document_loaders import URLDataLoader
# from langchain.docstore import URLDataLoader

# url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
# url_dataloader = URLDataLoader(url)
# document = url_dataloader.load()
# print(document)

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

url = "https://docs.python.org/3.9/"
loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
docs = loader.load()

print(docs)
print(len(docs))