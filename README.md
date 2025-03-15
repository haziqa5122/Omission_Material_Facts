### README  

#### Approach  
1. **Identifying Omissions**: We prompted an LLM using structured categories (e.g., side effects, interactions) to identify possible omissions in the post, minimizing hallucinations.  
2. **Cross-Referencing**: Observastions were validated by performing a similarity search against relevant sections in a PDF about the drug discussed. If an observation is matched with a document a review will be formed as shown:

#### Tools  
- **LLM**: (e.g., OpenAI GPT)  
- **Similarity Search**: (e.g., ApertureDB)  
- **PDF Parsing**: (e.g., Unstructured)  
- **Embeddings**: (e.g., SentenceTransformers)

#### How to

1. Install the packages by:

``` pip install -r requirements.txt ```

2. Add your OpenAI key in the .env file. 

3. Ingest the documents in the vector database if it's the first time:

``` python3 storage/ingest.py ```

4. After ingestion is done, run the main file:

``` python3 main.py ```

