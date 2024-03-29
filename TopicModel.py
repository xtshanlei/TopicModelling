import streamlit as st
import pathlib
import pandas as pd
import numpy as np
import sys
basedir='../'
sys.path.append(basedir)
import pylab as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from hlda.sampler import HierarchicalLDA

from IPython.core.display import HTML, display
import pandas as pd
import string
import re
import glob
import nltk
import base64
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import streamlit.components.v1 as components
#import wget
nltk.download('stopwords')
nltk.download('punkt')
from tqdm import tqdm_notebook as tqdm
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
    href = href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
################UPDATE NOTES#############################################
st.sidebar.title("AutoTopic v2.0")
st.sidebar.write('by Yulei')
st.sidebar.subheader('v2.0 Update Notes')
st.sidebar.write('- Add the support for LDA model')

#####################################################################
uploaded_file = st.file_uploader("Choose a file", type=['.csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1', engine = 'python')
    df = df.dropna(how='all').replace(np.nan, '',regex=True).reset_index()
    st.write(df)
    merge_required = st.checkbox('Need to merge columns?')

    if merge_required:
        columns_selected = st.multiselect('Which columns do you want to merge?',df.columns)
        df['texts'] = df[columns_selected].agg(' '.join, axis=1)
        st.write(df['texts'])
        comments = df['texts'].astype(str)
    else:
        text_column = st.selectbox('Please choose the column name of the texts:',df.columns)
        comments = df[text_column].astype(str)
    ExStopWords = st.text_input("Any extra words to be removed? Split using space. e.g. good nice")
    ExStopWords_l = ExStopWords.split()
    stopset = stopwords.words('english') + ExStopWords_l
    corpus=[]
    vocab=set()
    stemmer = PorterStemmer()
    all_filtered_words = []

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
          all_filtered_words.append(w)
      vocab.update(filtered)
      if filtered in corpus:
        continue
      corpus.append(filtered)
    vocab = sorted(list(vocab))
    vocab_index = {}
    for i, w in enumerate(vocab):
        vocab_index[w] = i
    st.subheader("Wordcloud")
    no_of_words = st.slider('How many words do you want?', 1, 50, 20)
    wordcloud = WordCloud(background_color='white',stopwords = stopset, max_words =no_of_words).generate(' '.join(all_filtered_words))
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
    st.subheader('Topic Modelling')
    model_type = st.radio("Please choose the topic model", ('LDA - Latent Dirichlet Allocation', 'hLDA - hierarchical LDA'))
    if model_type == 'hLDA - hierarchical LDA': #HLDA模型
        st.subheader("Parameters for hLDA:")
        n_samples = st.slider('No of iterations for the sampler(Default:100)', 10,200,100)    # no of iterations for the sampler
        alpha = st.slider('Smoothing over level distributions(Default:10.0)',1.0, 50.0,10.0)         # smoothing over level distributions
        gamma = st.slider('CRP smoothing parameter; number of imaginary customers at next, as yet unused table(Default:1.0)', 1.0, 10.0, 1.0)           # CRP smoothing parameter; number of imaginary customers at next, as yet unused table
        eta = st.slider('Smoothing over topic-word distributions(Default:0.1)', 0.1, 5.0, 0.1)             # smoothing over topic-word distributions
        num_levels = 3        # the number of levels in the tree
        display_topics = 5   # the number of iterations between printing a brief summary of the topics so far
        n_words = 5           # the number of most probable words to print for each topic after model estimation
        with_weights = False  # whether to print the words with the weights
        topic_model_start = st.button('Press to generate topics...')
        results_df= pd.DataFrame()
        if topic_model_start:
            st.info("If it's a large data,The process may take quite a long time, please be patient...")
            hlda = HierarchicalLDA(new_corpus, vocab, alpha=alpha, gamma=gamma, eta=eta, num_levels=num_levels)
            hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words, with_weights=with_weights)
            st.success('Well done! You did it!')
            st.balloons()
            d = 0
            n =0
            node = hlda.document_leaves[d]

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
            st.subheader("Results:")
            st.write('You can download the results by clicking the button below!')
            st.write(results_df)
            # Examples

        if not results_df.empty:
            tmp_download_link = download_link(results_df, 'h_topics.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
    elif model_type == 'LDA - Latent Dirichlet Allocation': #LDA模型
        def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
            """
            Compute c_v coherence for various number of topics

            Parameters:
            ----------
            dictionary : Gensim dictionary
            corpus : Gensim corpus
            texts : List of input texts
            limit : Max num of topics

            Returns:
            -------
            model_list : List of LDA topic models
            coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            """
            coherence_values = []
            model_list = []
            for num_topics in range(start, limit, step):
                model = gensim.models.LdaModel(corpus=corpus,
                                                   num_topics=num_topics,
                                                   id2word=id2word)
                model_list.append(model)
                coherencemodel = CoherenceModel(model=model,
                                                texts=texts,
                                                dictionary=dictionary,
                                                coherence='c_v')
                coherence_values.append(coherencemodel.get_coherence())

            return model_list, coherence_values
        id2word = corpora.Dictionary(corpus)
        data_lemmatized = corpus
        corpus = [id2word.doc2bow(text) for text in corpus]
        st.subheader('Please choose the parameters for LDA')
        topic_range = st.slider('Select a range of the topic number:', 2,30, (2,20),step=5)
        limit=topic_range[1]
        step=st.slider('Steps', 1, 5, value =5, step=1)
        start=topic_range[0]

        start_coherence = st.button('Press to start LDA automatically')
        if start_coherence:
            st.info('It may take a long time, please wait...')

            model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                                    corpus=corpus,
                                                                    texts=data_lemmatized,
                                                                    start = start,
                                                                    limit= limit,
                                                                    step=step)

            x = range(start, limit, step)
            chart_data = pd.DataFrame(coherence_values, columns=[x])
            st.line_chart(chart_data)
            for m, cv in zip(x, coherence_values):
                st.write("Num Topics =", m, " has Coherence Value of", round(cv, 4))
            model_index = coherence_values.index(max(coherence_values))
            best_model = model_list[model_index]
            st.write('The {} is selected'.format(x[model_index]))
            vis = pyLDAvis.gensim.prepare(best_model, corpus, id2word)
            pyLDAvis.save_html(vis, 'optimal_model.html')
            lda_visualisation = open('optimal_model.html', 'r', encoding ='utf-8')
            source_code = lda_visualisation.read()
            st.subheader('Visualisation of Topics:')
            components.html(source_code, height = 1080, width=1920)
            lda_visualisation.close()
            st.write(best_model.print_topics(num_words=5))


    else:
        st.write('Please choose the topic model above!')
