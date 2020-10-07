import concurrent.futures
import requests                # This is not standard library
 
URLS = [
    'https://docs.python.org/3/library/ast.html',
    'https://docs.python.org/3/library/abc.html',
    'https://docs.python.org/3/library/time.html',
    'https://docs.python.org/3/library/os.html',
    'https://docs.python.org/3/library/sys.html',
    'https://docs.python.org/3/library/io.html',
    'https://docs.python.org/3/library/pdb.html',
    'https://docs.python.org/3/library/weakref.html'
]
 
 
def get_content(url):
    return requests.get(url).text
 
 
def scrap():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(get_content, url): url for url in URLS}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page length is %d' % (url, len(data)))
 
 
def main():
    for url in URLS:
        try:
            data = get_content(url)
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page length is %d' % (url, len(data)))
 
 
if __name__ == '__main__':
    scrap()