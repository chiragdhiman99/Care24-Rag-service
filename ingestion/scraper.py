import requests
from bs4 import BeautifulSoup

def scrape_medline(disease_name):
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": disease_name}
    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.text, "xml")
    results = []
    for doc in soup.find_all("document"):
        title   = doc.find("content", attrs={"name": "title"})
        summary = doc.find("content", attrs={"name": "FullSummary"})
        results.append({
            "title"  : BeautifulSoup(title.get_text(), "html.parser").get_text().strip() if title else "",
            "summary": BeautifulSoup(summary.get_text(), "html.parser").get_text().strip() if summary else "",
        })
    return results

def all_topics_names():
    all_topics = []
    
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {"db": "healthTopics", "term": letter, "retmax": "100"}
        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, "xml")
        
        for doc in soup.find_all("document"):
            title = doc.find("content", attrs={"name": "title"})
            if title:
                all_topics.append(title.get_text().strip())
        
    
    return all_topics

topics = all_topics_names()

all_data =[]

for topic in topics:
    data = scrape_medline(topic)
    all_data.extend(data)


