import requests
import os
import re
import string
from bs4 import BeautifulSoup


"""
This function extracts the data from the HTML files and returns it in a dictionary format.
"""
def dataExtractor(path):
    names = os.listdir(path)
    allData = {}
    for nam in names:
        with open(os.path.join(path, nam), encoding='utf-8') as f:
            rawData = f.read()
        
        soup = BeautifulSoup(rawData, 'html.parser')
        l = [loc.text.strip() for loc in soup.find_all('h3', class_ = 'font104')]
        data = {}

        for i,j in zip(range(len(l)), l):
            names = []
            links = []
            for k in (soup.find_all('ul', class_ = 'leftul')[i].find_all('li')):
                names.append(k.text.strip())
            # print(names)

            for k in soup.find_all('ul', class_ = 'leftul')[i].find_all('li'):
                links.append('https://pib.gov.in/' + k.find('a').attrs['href'])
            # print(links)    
            data[j] = (names, links)
        allData[nam] = data
    return allData



"""
This is the format of the dictionary that is returned by the dataExtractor function: 

{"month":{"ministry1":([heading 1, heading2, ...], [link1, link2, ...]),
            "ministry2":([heading 1, heading2, ...], [link1, link2, ...]),
            ......},

"month":{"ministry1":([heading 1, heading2, ...], [link1, link2, ...]),
            "ministry2":([heading 1, heading2, ...], [link1, link2, ...]),
            },
......}            

"""




"""
This function cleans the text by removing HTML tags, punctuation marks, and extra spaces.
"""

def clean_text(text):
    # Remove HTML tags using regex
    html_tag_pattern = re.compile(r'<[^>]+>')
    text = re.sub(html_tag_pattern, '', text)

    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace multiple spaces with a single underscore
    text = re.sub(r'\s+', '_', text.strip())
    
    # Truncate text to a maximum length of 40 characters
    if len(text) > 40:
        text = text[:40]
    
    return text

"""
This function creates text files for each heading in the data dictionary.
"""

def textFilesCreator(data):
    months = data.keys()
    for i in months: #If you want to customize the months, you can use list slicing here
        types = data[i].keys()
        for j in types:
            path = os.path.join(clean_text(i), clean_text(j)) # Here Clean Text function is used to remove special characters from the folder name
            os.makedirs(path, exist_ok=True)  # Avoid error if directory already exists
            
            for k in range(len(data[i][j][0])):
                heading = data[i][j][0][k]
                link = data[i][j][1][k]
                linkRes = requests.get(link)
                
                if linkRes.status_code == 200:
                    linkSoup = BeautifulSoup(linkRes.content, 'html.parser')
                    s = ''
                    
                    for l in linkSoup.find_all('h2'):
                        s += l.text.strip() + '\n'
                        
                    for m in linkSoup.find_all('h3'):
                        s += m.text.strip() + '\n'
                        
                    n = len(linkSoup.find_all('p')) // 2

                    for o in linkSoup.find_all('p')[:n]:
                        s += o.text.strip() + '\n'

                    # Create the file and write the content
                    file_path = os.path.join(path, f"{clean_text(heading)}.txt")
                    with open(file_path, 'w+', encoding='utf-8') as f:
                        f.write(s)

if __name__ == "__main__":
    path = 'specify the path of html files here'
    data = dataExtractor(path)
    textFilesCreator(data)
    print("Text files created successfully!")