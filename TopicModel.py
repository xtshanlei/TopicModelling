import streamlit as st
import pandas as pd
import sys
basedir='../'
sys.path.append(basedir)
import pylab as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from hlda.sampler import HierarchicalLDA
from ipywidgets import widgets
from IPython.core.display import HTML, display
import pandas as pd
import string
import re
import glob
import nltk
import base64
nltk.download('stopwords')
nltk.download('punkt')
from tqdm import tqdm_notebook as tqdm
from send_email_attach import send_mail
@st.cache
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

st.title("Automatic Topic Modelling")
st.write("If ValueError appears, just refresh the page")
uploaded_file = st.file_uploader("Choose a file", type=['.csv'])
email_address = st.text_input('Please type your email address here. We will send the result via email')
if st.button('Download Dataframe as CSV'):
    tmp_download_link = download_link(uploaded_file, 'file.csv', 'Click here to download your data!')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
if email_address:
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1', engine = 'python')
        if len(df)>50000:
            df = df.sample(n = 10000)
        st.write(df['texts'])

        stopset = stopwords.words('english')+['covid','vaccin','http','https','say','thi','coronaviru']
        corpus=[]
        vocab=set()
        stemmer = PorterStemmer()
        comments = df['texts']

        def preprocess_text(sen):
            # Remove punctuations and numbers
            sentence = re.sub('[^a-zA-Z]', ' ', sen)

            # Single character removal
            sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence)
            data =[]
            data_words =[]
            text,sep,tail = str(sentence).partition('http')

            return text

        for comment in tqdm(comments):
          text = preprocess_text(comment)
          tokens = word_tokenize(text)
          filtered = []
          for w in tokens:
              w = stemmer.stem(w.lower()) # use Porter's stemmer
              if len(w) < 3:              # remove short tokens
                  continue
              if w in stopset:            # remove stop words
                  continue
              filtered.append(w)
          vocab.update(filtered)
          if filtered in corpus:
            continue
          corpus.append(filtered)

        vocab = sorted(list(vocab))
        vocab_index = {}
        for i, w in enumerate(vocab):
            vocab_index[w] = i

        wordcloud = WordCloud(background_color='white',stopwords = stopset).generate(' '.join(filtered))
        plt.figure(figsize=(12, 12))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        new_corpus = []
        for doc in corpus:
            new_doc = []
            for word in doc:
                word_idx = vocab_index[word]
                new_doc.append(word_idx)
            new_corpus.append(new_doc)


        n_samples = 100      # no of iterations for the sampler
        alpha = 10.0          # smoothing over level distributions
        gamma = 1.0           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
        eta = 0.1             # smoothing over topic-word distributions
        num_levels = 3        # the number of levels in the tree
        display_topics = 5   # the number of iterations between printing a brief summary of the topics so far
        n_words = 5           # the number of most probable words to print for each topic after model estimation
        with_weights = False  # whether to print the words with the weights
        st.info('The process may take quite a long time, please be patient...')
        hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
        hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
        st.success('Well done! You did it!')
        st.balloons()
        d = 0
        n =0
        node = hlda.document_leaves[d]
        st.write('topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers) +node.get_top_words(n_words, with_weights))
        st.write(node.parent.node_id, node.parent, node.total_words)

        #hlda.print_nodes(n_words, with_weights)

        def topic_df(model,node, indent, n_words, with_weights):
                out = '   ' * indent
                out += 'topic=%d level=%d (documents=%d): ' % (node.node_id, node.level, node.customers)
                out += node.get_top_words(n_words, with_weights)
                print(out, node.total_words)
                for child in node.children:
                    topic_df(model,child, indent+1, n_words, with_weights)
        def topic_level3(model,node,n_words,with_weights):
          topic_words = []
          parent_words = []
          total_words = []
          for child in node.children:
            for i in child.children:
              topic_words.append(i.get_top_words(n_words, with_weights))
              parent_words.append(i.parent.get_top_words(n_words, with_weights))
              total_words.append(i.total_words)
          return topic_words, parent_words, total_words
              #i.total_words, i.get_top_words(n_words, with_weights), i.parent.get_top_words(n_words, with_weights)

        topic_words, parent_words, total_words = topic_level3(hlda,hlda.root_node,n_words,with_weights)

        results_df = pd.DataFrame({'topic_words':topic_words,
                           'parent_words':parent_words,
                           'total_words':total_words,
                           'grandparents':hlda.root_node.get_top_words(n_words,with_weights)})
        results_df.to_csv('results.csv')



        send_mail('yulei.li@durham.ac.uk', email_address, 'Hierarchical topics results','Please reference us:', files=['results.csv'],
                      server="smtp-relay.sendinblue.com", port=587, username=config.USERNAME, password=config.PASSWORD,
                      use_tls=True)
        # Examples

        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(results_df, 'topics.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
