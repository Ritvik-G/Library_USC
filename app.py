# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request 
from flask_cors import CORS
from pymed import PubMed
import json
from transformers import DPRContextEncoder
from transformers import DPRContextEncoderTokenizerFast
from transformers import DPRContextEncoderTokenizer
import time
import math
from transformers import DPRQuestionEncoderTokenizerFast
import re
from pymed import PubMed
import json
from transformers import DPRContextEncoder
from transformers import DPRContextEncoderTokenizerFast
from transformers import DPRContextEncoderTokenizer
import time
import math
from transformers import DPRQuestionEncoderTokenizerFast
import textwrap
import faiss
import numpy as np
import torch
import time
import datetime
import pandas as pd
from transformers import DPRQuestionEncoder
from tqdm import tqdm
import ast
import json
import textwrap
import faiss
import numpy as np
import torch
import time
import datetime
import pandas as pd
import uvicorn

#Kaushik: Imports for integrating masked LM
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import ast


from fastapi import FastAPI, Request, HTTPException

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

else:

    device = torch.device("cpu")

nlp = spacy.load("en_core_web_sm")


from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
model_biobert = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# Move the encoder model to the GPU.
q_encoder = q_encoder.to(device=device)

q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")


# Create a PubMed object that GraphQL can use to query
# Note that the parameters are not required but kindly requested by PubMed Central
# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
pubmed = PubMed(tool="MyTool", email="my@email.address")
pubmed._rateLimit = 100

# Create a GraphQL query in plain text
#query = "Intraabdominal Infections"

# sample_queries = ['Management', 'outcomes', 'intra-abdominal', 'infections', 'treatment']

output_dict = {}
out_titles = []
out_ctx = []

all_titles = []
all_ctx = []

score_dict_all = {}
outputs_all = []

# masks_df = pd.read_csv('sample-outputs-library.csv')
# masks_all = list(masks_df['masks'])
# #print(len(masks_all))
# masks_list = [ eval(i) for i in masks_all ]
# #print(len(masks_list))
# count = 0


def get_masks(query):
    doc = nlp(query)
    np = [np.text for np in doc.noun_chunks]
    #print(f'noun phrases are: {np}')
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    #print(f'nouns are: {nouns}')
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    #print(f'verbs are: {verbs}')

    np_single = [] #break multi word noun phrase entry to single word entries
    for phrase in np:
        words = phrase.split()
        np_single.extend(words)
    #print(np_single)

    x = set(np_single).union(set(nouns))
    mask_list_all = list(x.union(set(verbs)))
    mask_list = [word for word in x if word.lower() not in nlp.Defaults.stop_words]
    for word in mask_list:
        #print(word)
        if len(word)<=2:
            #print(word)
            mask_list.remove(word)
    return mask_list

def get_data(query): #returns two lists of titles and abstracts corresponding to a biopython API search respectively
    titles = []
    articles = []
    pass_dict = {}
    query = ''.join([i for i in query if i.isalpha()])
    results = pubmed.query(query, max_results=20)

    for article in results:
        article_dict = article.toDict()
        title = article_dict['title']
        abstract = article_dict['abstract']
        # title = article.title
        # abstract = article.abstract
        #print(title)
        #print(abstract)
        titles.append(title)
        articles.append(abstract)
        pass_dict[title] = abstract
    #print(f'\n LENGTH OF TITLES: {len(titles)}')
    #print(f'\n LENGTH OF ARTICLES: {len(articles)}')
    #print(f'\n LENGTH OF PASS DICT: {len(pass_dict)}')

    return titles, articles, pass_dict


def get_ctx(query, user_inp):

    # Execute the query against the API
    #print(query)
    query = ''.join([i for i in query if i.isalpha()])
    titles, articles, pass_dict = get_data(query)
    #print('Before splitting, {:,} articles.\n'.format(len(titles)))

    # We have two lists, 'titles' and 'articles'.

    passage_titles = []
    passages = []

    #print('Splitting...')

    # For each article and its title...
    for i in range(len(titles)):

        title = str(titles[i])  #convert biopython string element to python string
        article = str(articles[i])
        # print(title)   
        # print(type(title))
        

        # Skip over any without contents.
        if (article is None or len(article) == 0):
            print('Skipping empty article:', title)
            continue

        # Split the text on whitespace.
        # By default, this removes all whitespace, including newline and tab
        # characters.
        words = article.split()

        # Loop over the words, incrementing by 100.
        for i in range(0, len(words), 100):

            # Select the next 100 words.
            # Python slices automatically stop at the end of the array.
            chunk_words = words[i : i + 100]

            # Recombine the words into a passage by joining with whitespace.
            chunk = " ".join(chunk_words)

            # Remove any trailing whitespace.
            chunk = chunk.strip()

            # To avoid a possible edge case, skip any empty chunks.
            if len(chunk) == 0:
                continue

            # Store the chunk. Every chunk in the article uses the article title.
            passage_titles.append(title)
            passages.append(chunk)

    #print('  Done.\n')

    chunked_corpus = {'title': passage_titles, 'text': passages}

    num_passages = len(chunked_corpus['title'])

    #print('Tokenizing {:,} passages for DPR...'.format(num_passages))

    # Tokenize the whole dataset! This will take ~15 to 20 seconds.
    outputs = ctx_tokenizer(
        chunked_corpus["title"],
        chunked_corpus["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt"
        #add_special_tokens=True
    )

    #print('  DONE.')

    # `input_ids` holds the encoded tokens for the entire corpus.
    input_ids = outputs["input_ids"]

    
    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))




    # We're running the model forward only, so no need for gradients.
    torch.set_grad_enabled(False)

    # Track elapsed time for progress updates.
    t0 = time.time()

    # Track the current batch number, also for progress updates.
    step = 0

    # How many passages to process per batch.
    batch_size = 16

    # Get the number of passages in the dataset, we'll use this in a few places.
    num_passages = input_ids.size()[0]

    # Calculate the number of batches in the dataset.
    num_batches = math.ceil(num_passages / batch_size)

    # As we embed the passages in batches, accumulate them in this list.
    embeds_batches = []

    print('Generating embeddings for {:,} passages...'.format(num_passages))

    # For each batch of passages...
    for i in range(0, num_passages, batch_size):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            #print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, num_batches, elapsed))

        # Select the next batch.
        batch_ids = input_ids[i:i + 16, :]

        # Move them to the GPU.
        batch_ids = batch_ids.to(device)

        # Run the encoder!
        outputs = ctx_encoder(
            batch_ids,
            return_dict=True
        )

        # The embeddings are in the field "pooler_output"
        embeddings = outputs["pooler_output"]

        # Bring the embeddings back over from the GPU and convert to numpy (out of
        # pytorch)
        embeddings = embeddings.detach().cpu().numpy()

        embeds_batches.append(embeddings)

        step += 1




    # Combine the results across all batches.
    embeddings = np.concatenate(embeds_batches, axis=0)


    dim = 768

    m = 128

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)


    t0 = time.time()

    index.train(embeddings)

    index.add(embeddings)

    #print('  DONE.')

    #print('  Adding embeddings to index took', format_time(time.time() - t0))



    


        # Tokenize the question.
    input_ids = q_tokenizer.encode(user_inp, return_tensors="pt")

    # Move the question over to the GPU.
    input_ids = input_ids.to(device)

    # Run the question through BERT and generate the question embedding.
    outputs = q_encoder(input_ids)

    # The embedding is stored in the `pooler_output` property.
    q_embed = outputs['pooler_output']

    # Our FAISS index is on the CPU, not the GPU, so we need to transfer this
    # question embedding over to the CPU to do our search.
    q_embed = q_embed.cpu().numpy()

    # Check out the embedding's size, out of curiosity :)
    #print("Query embedding:", q_embed.shape)


    # Find the k=3 most similar passages to the question embedding `q_embed`.
    D, I = index.search(q_embed, k=10)

    # Print out the indeces and simlarity scores--we'll print out the passage text
    # next.
    #print(type(D))
    #print(type(I))
    
    #print('Closest matching indeces:', I)
    #print('Inner Products:', D)
    D = D.tolist()
    I = I.tolist()
    
    #print(D)
    #print(I)
    #print(chunked_corpus['title'][6])
    
    title_list = [chunked_corpus['title'][i] for i in I[0]]
    #print(f'title list is: {title_list}')
    #score_list = list(map(lambda x, y:(x,y), I[0], D[0]))
    score_list = list(map(lambda x, y:(x,y), title_list, D[0]))
    #print(f'\n score list is: {score_list}')
    score_dict = {}
    output_dict = {}
    out_titles = []
    out_ctx = []

    score_dict[query] = score_list
    #print(f'\n score dict is: {score_dict}')


    # Wrap text to 80 characters.
    wrapper = textwrap.TextWrapper(width=80)

    # For each of the top 'k' results..
    j = 0
    for i in I[0]:

        #print('Index:', i)

        # Retrieve passage and its title.
        title = chunked_corpus['title'][i]
        passage = chunked_corpus['text'][i]
        #print(query)
        #print('Article Title:   ', title, '\n')
        #print('Passage:')
        #print(wrapper.fill(passage))

        #print('')
        
        if title not in output_dict:    
            output_dict[title] = passage
        
        else:
            mod_title = f'{title}_{str(j)}'
            output_dict[mod_title] = passage
        out_titles.append(query + ': ' + title)
        out_ctx.append(passage)
    return output_dict, out_titles, out_ctx, score_dict, pass_dict


# creating a Flask app 
app = Flask(__name__) 
CORS(app)

#s = "intrabdominal infections may required surgical interventions"



def try_main(s):
    user_inp = s
    #user_inp = "intrabdominal infections may required surgical interventions"
    user_inp = json.dumps(user_inp)
    masks_list = get_masks(user_inp)
    #ans, titles, ctxs = get_ctx(masks_list[0])
    outputs_all = []
    for mask in masks_list:
        # if not mask:
        #     raise HTTPException(status_code=404, detail="Item not found")
        ans, titles, ctxs, d, e = get_ctx(mask, user_inp)
        n = len(ctxs)
        flat_data = [(key, title, score) for key, values in d.items() for title, score in values]
        flat_data = [tuple(list(flat_data[i])+[ctxs[i]]) for i in range(n)]
        print(f'flat_data: {flat_data}')
        sorted_data = sorted(flat_data, key=lambda x: x[2], reverse=True)
        print(f'sorted data: {sorted_data}')
        # Get the top 3 scores
        top_scores = sorted_data[:3]
        print(f'top score: {top_scores}')
            
        # all_titles.append(titles)
        # all_ctx.append(ctxs)
        #d.update({ans[mask]})
        score_dict_all.update(d)
        print(score_dict_all)

    for key, title, score, passage in top_scores:
        print(f"Key: {key}, Title: {title}, Score: {score}")
        #print(f'context: {ans[title]}')
        tokenizer = AutoTokenizer.from_pretrained("RohanVB/umlsbert_ner")
        model = AutoModelForTokenClassification.from_pretrained("RohanVB/umlsbert_ner")
        pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
        highlights = [item['word'] for item in pipe(passage)]
        output = {
            "Key": key,
            "Title": title,
            "Score": score,
            "Context": passage,
            "Highlights": highlights
        }
        outputs_all.append(output)
    print ("="*40)
    print (outputs_all)
    print ("="*40)
    #input()
    df = pd.DataFrame(outputs_all)
    df.to_csv('api_sample_output.csv', )
    return outputs_all

# on the terminal type: curl http://127.0.0.1:5000/ 
# returns hello world when we use GET. 
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST']) 
def home1(): 
    return "Server running properly"
    
@app.route('/get_answers', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'): 
        s = request.args.get('query')
        print(s)
        return jsonify(try_main(s)) 



# driver function 
if __name__ == '__main__': 

	app.run(host='127.0.0.1', port=5000,debug = False) 
