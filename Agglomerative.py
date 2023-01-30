import spacy
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import HdpModel
import pandas as pd
import numpy as np
import re
import gensim
from sklearn.metrics.pairwise import cosine_similarity


#load english dictionary for spacy
en=r'en_core_web_sm\en_core_web_sm-2.3.1'
nlp=spacy.load(en)



def clean(text):
    #remove https
    text=re.sub('http.*', '', text)
    #remove emojis and irregullar puntuations
    text=''.join(re.findall(r'[\w\s,.\-]+', text))
    text= re.sub('\w+\.\w+.*', '', text)
    return text.lower()
                 
def clean_text_via_tokenizer(text):
    if pd.isna(text):
        return text
    text=nlp(clean(text))
    tokinized=[element.lemma_ for element in text if not (element.is_punct or element.is_stop or 
                                                          element.is_space or element.is_digit)]
    joined =' '.join(tokinized)
    return joined 


def fit_predict_topics(df, cleaned_col='topic'):
    
    #convert to 
    cv = CountVectorizer(stop_words=ENGLISH_STOP_WORDS.union({'1','2','3','i','ii', 'iii'}), 
                         min_df=2, ngram_range=(1,2), max_df=0.5)
    data_cv = cv.fit_transform(df[cleaned_col])
    
    corpus = gensim.matutils.Sparse2Corpus(data_cv, documents_columns=False)
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())
    
    hdpmodel=HdpModel(corpus=corpus, id2word=id2word)
    prediction=df[cleaned_col].apply(lambda x: topic_prediction(x,hdpmodel, cv))
    return prediction

def topic_prediction(cleaned_text, model, cv):
    if cleaned_text == 'notavaliable':
        return np.nan
    string_input = [cleaned_text]
    X = cv.transform(string_input)
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    try:
        output = list(model[corpus])[0]
        topics = sorted(output,key=lambda x:x[1],reverse=True)

        return topics[0][0]
    except IndexError:
        return np.nan
    
    
    
def convert_string_to_unique_list(string):
    new_list = string.lower().split(' ')
    new_list=list(set(new_list))
    return new_list





def check_similary(series):
    
    if series.shape[0]<=3:
        threshold=0.70
    else:
        threshold=0.80
    
    count_vectorizer = CountVectorizer(ngram_range=(1,2), 
                                       stop_words=ENGLISH_STOP_WORDS.union({'1','2','3','i','ii', 'iii'}))
    
    category_cv = count_vectorizer.fit_transform(series)

   


    similarity= cosine_similarity(category_cv).mean()
    return similarity >= threshold

def major_similar_videos(df, tokenized_string_col='topic', return_mask=False):
    return check_similary(df[tokenized_string_col])






def remove_ones(df):
    values=df['prediction'].value_counts()
    values=values[values==1].index
    
    df.loc[df['prediction'].isin(values),'prediction']=np.nan
    df.loc[df['prediction'].isin(values),'similar']=np.nan
    
    return df



    



def check_or_predict_again(df, x_indices):
    
    ''' 
    This does the following
    
    1. Takes in a dataframe and checks if predicted values are similar, if they are not it fits and predict a new model
    2. Changes the indices of predictions to avoid overlaps. 
    
    '''
    df.index.name=''
    name=df['prediction'].iloc[0]
    x=  x_indices[name]

    if major_similar_videos(df):
        df['similar'] =True
        
    else:
        try:
            pred= fit_predict_topics(df) +x
            
            df.loc[pred.index,'prediction'] = pred
            df.loc[pred.index,'similar'] = np.nan
            
            

        except ValueError:
            
            df['prediction'] = np.nan
            df['similar'] = np.nan
    return df



# d.groupby('prediction').apply(_try_return_dataframe).reset_index(drop=True).set_index('index')

def multiple_fit_predict_iter(dataframe, iteration_checks=20):
    
    """
    This is the main Agglemartive functions, the only custom hyperparameter is
    iteration_checks, which is used to determine how many iterations to allow 
    until there are no futher improvements in the model
    
    default is 20, this means that after every 20 iterations, it compairs similarity
    ratio with the value in the previous 20th iteration, if there are no different, then it breaks,
    It is kind of similar to Deep learnings early stoping.
    """
    
    
    df=dataframe.copy()
    
    ## add new columns
    df['prediction']= fit_predict_topics(df)
    df['similar'] = np.nan
    df['prediction']=df['prediction'].replace(np.nan,-1) #replace np.nan with -1
    
    
    #dataframes to work with
    d_similar=df[df['similar'].notnull()]
    tempo_df=df[df['similar'].isna()]
    
    

    old_similar_ratio=0
    
   
    iteration_count=1
    while tempo_df.shape[0] != 0:
        
        
        
        new_similar_ratio = d_similar.shape[0]/df.shape[0] #to know the proportion for the entire dataset that is similar
        
        
        
        
        #CHECKS
        #dico stores assigns values to every prediction values to ensure that they do not overlap when new prediction is made
        dico= ((pd.Series(dict([i[::-1] 
                              for i in list( 
                                  enumerate(tempo_df['prediction'].unique(), 1))]
                              )
                          
                        )*200)+x).to_dict()

        
        tempo_df=tempo_df.groupby('prediction').apply(check_or_predict_again,
                                                      dico).reset_index(drop=True)
        
        tempo_df=remove_ones(tempo_df)
        
        
        #replace old dataframes 
        
        d_similar=pd.concat([d_similar, 
                             tempo_df[(tempo_df['similar'].notnull()
                                      & tempo_df['prediction'].notnull())]])
        
        tempo_df=tempo_df[(tempo_df['similar'].isna()|
                           tempo_df['prediction'].isna())]

        df=pd.concat([d_similar,tempo_df])
        tempo_df['prediction']=tempo_df['prediction'].replace(np.nan,-1)
        
        
        print('{} : {:.2%}'.format(iteration_count,new_similar_ratio))
        
        
        
        #Using iteration checks
        if ((iteration_count%iteration_checks)==0) and (old_similar_ratio==new_similar_ratio):# 
            
            tempo_df['prediction']=-1 #make all leftovers outliers
            
            df=pd.concat([d_similar,tempo_df])
            
            
            break
        
        elif ((iteration_count%10)==0):
            old_similar_ratio=new_similar_ratio
            
            
        iteration_count+=1
            
    return df