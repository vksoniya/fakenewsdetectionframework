#crawl web to get a similar claim and show this is as crawled justification later
#myKey = "751c2357c01545f4937b1accdf9cc2f0" #Ayush 12.03.2021
#myKey = "d61c317f229a42ebb5cec2106165dcf2" #xotic 12.03.2021
#myKey = "23a5a703954d41db82e6e93bb557ab22" #8vijayak used
myKey = "f2ab6f15e39f485f9408718c83a3d360" #baw1725 13.03.2021
#myKey = "a0a187f18d7f4c919a56e5adc0cf786c" #sps - 11.03.2021
#myKey = "7d47a5fe394a40c39de4a0785deaef8f" #vk.soniya - 12.03.2021
import newspaper
from newspaper import Article
from newsapi import NewsApiClient
import pandas as pd
import numpy as np

newsapi = NewsApiClient(api_key=myKey)
news_sources = newsapi.get_sources()
article_limit_per_keyword = 10

#def create_crawled_articles_df(len_input_keywords):
def create_crawled_articles_df():
    
    column_names = ["Keyword", "Crawled Article Title", "Crawled Article Text", "Crawled Article Link", "Crawled Article Summary", "Crawled Article Keywords"]
    #df_crawled_articles = pd.DataFrame(np.nan, index=range(0,len_input_keywords*article_limit_per_keyword), columns=column_names)
    df_crawled_articles = pd.DataFrame(columns=column_names)

    return df_crawled_articles

def crawl_web(input_keywords):  
    c = 0
    #l = len(input_keywords)
    df_crawled_articles = create_crawled_articles_df()
   
    try:    
        for keyword in input_keywords:
            print(keyword)
            skeyword = str(keyword[0])
                
            all_articles = newsapi.get_everything(q=skeyword, language='en', sort_by='relevancy')
            num_articles = all_articles['totalResults']
            print("Number of articles for keywords %s is %s" %(str(keyword), str(num_articles)))
            if(num_articles > 1):
                if (num_articles < article_limit_per_keyword):
                    index_range = num_articles
                else:
                    index_range = article_limit_per_keyword
                for n in range(int(index_range)):
                    article_to_append = all_articles['articles'][n]
                    stitle = article_to_append['title']
                    
                    url = article_to_append['url']
                    
                    scraped_article = Article(url)
                    
                    try:
                        scraped_article.download()
                        scraped_article.parse()
                        scraped_article.nlp()
                    except newspaper.article.ArticleException:
                        continue

                    #check for superset keyword match. if all keywords present only then store
                    articleSummary = str(scraped_article.summary)
                    #isMatch = all(string in articleSummary.lower() for string in skeyword)
                    isMatch = all(string in articleSummary.lower() for string in skeyword)
                    if (isMatch):
                        print("Match available")
                        df_crawled_articles = df_crawled_articles.append({'Keyword': keyword, 'Crawled Article Title': stitle, 'Crawled Article Link': article_to_append['url'], 'Crawled Article Text': scraped_article.text, 'Crawled Article Summary':scraped_article.summary, 'Crawled Article Keywords':scraped_article.keywords }, ignore_index=True)
                        
                    c =  c + 1
                    print("at index: " + str(c))
    except Exception as ex:
        print("There is an exception in web crawler: " + str(ex))
        pass

    return df_crawled_articles