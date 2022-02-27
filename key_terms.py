from lxml import etree
from nltk import tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_news(xml_path):
    news_dict = {}
    for news in etree.parse(xml_path).getroot()[0]:
        news_dict[news[0].text] = news[1].text
    return news_dict


def lemmatization(news_dict):
    lemmat_dict = {}
    lemmatizer = WordNetLemmatizer()
    for headline, news_text in news_dict.items():
        tokens = sorted(tokenize.word_tokenize(news_text.lower()), reverse=False)
        lemmats = []
        for token in tokens:
            lemmat = lemmatizer.lemmatize(token)
            if lemmat not in stopwords.words('english') + list(punctuation) and pos_tag([lemmat])[0][-1] == 'NN':
                lemmats.append(lemmat)
        lemmat_dict[headline] = " ".join(lemmats)
    return lemmat_dict


def main():
    news_lammat = lemmatization(extract_news('news.xml'))
    vectorizer = TfidfVectorizer(input='content')
    tfidf_matrix = vectorizer.fit_transform([values for values in news_lammat.values()])
    terms = vectorizer.get_feature_names_out()
    n, m = tfidf_matrix.shape
    for k in range(n):
        list_ = sorted(sorted([(i, tfidf_matrix[k][(0, i)]) for i in range(m)], reverse=True), key=lambda x: x[-1], reverse=True)
        print(list(news_lammat)[k] + ':')
        print(*[terms[i] for i, _ in list_[:5]])


if __name__ == '__main__':
    main()
