## Week 1 Notes

### Video 2 - Understanding the Architecture & Project Structure


:::spoiler Detailed

### Project Overview and Problem Statement

- **Problem Solving Focus**: The project aims to create a meaningful, production-like system rather than a toy project. It mimics real-world systems in companies, focusing on staying updated with the latest AI research papers.
- **Use Case**: Build a local RAG (Retrieval-Augmented Generation) system to query academic papers from arXiv. For example, querying "what is attention mechanism?" retrieves relevant chunks from papers like "Attention Is All You Need" (2017) and generates answers based on them.
- **Data Source**: arXiv archive, which publishes 80-100+ AI papers daily. Used for managers/companies to ensure developments align with the latest global knowledge, avoiding outdated or SEO-biased Google search results.
- **Advantages Over LLMs**: Latest LLMs may lack access to very recent papers; this system ensures up-to-date, source-specific information.

### Data Ingestion Pipeline

- **arXiv API**: Public API for fetching papers, but rate-limited (cool down of 3 seconds between concurrent downloads to avoid rejection).
- **Orchestration Tool**: Apache Airflow used for scheduling and logging. Triggers daily (e.g., 10 AM) to fetch papers, handle failures via UI logs. Alternatives: Metaflow, Dagster, AWS Lambda.
- **ETL Pipeline**: Represents real-world data engineering pipelines for keeping RAG databases updated (daily or every 5 minutes for critical cases). Ignored in many courses but essential for production.
- **PDF Processing**:
  - Download PDFs via API.
  - Extract text using **OCR (Optical Character Recognition)** mechanisms.
  - Primary: **Docling** (chosen for ease and effectiveness after testing).
  - Alternative: **Grobid** (specific to academic papers).
  - Fallback Mechanism: Planned but simplified to Docling for output consistency. In production, multiple fallbacks ensure pipeline reliability.
- **Chunking Strategies**: Experiment with strategies (e.g., paragraph-based vs. section-based) to break text into manageable pieces for storage and retrieval.
- **Observability**: Error handling, logging, and monitoring to fix issues quickly.

### Storage and Database Architecture

- **Intermediate Storage**: PostgreSQL used as a middle layer to store metadata and extracted text from papers (e.g., last 3-4 months). Avoids re-downloading/re-processing if chunking strategy changes.
- **Vector Database**: OpenSearch for storing chunks, metadata, and embeddings.
  - Why OpenSearch: Supports **hybrid search** (keyword + vector retrieval), built on Elasticsearch (long-standing search engine for e-commerce like Amazon).
  - **Keyword Retrieval**: Ensures exact terms (e.g., in medical fields) are matched, avoiding misses from pure embedding-based similarity.
  - **Vector Retrieval**: Uses embeddings for semantic similarity (e.g., cosine similarity, HNSW - Hierarchical Navigable Small World graphs).
  - Customization: Supports sorting (e.g., by publication date), pre/post-filtering, low latency, distributed setup.
  - Alternatives: Qdrant, Pinecone, Weaviate (selection based on cost, maintenance, latency, cloud support; researched 30+ databases).
  - Considerations: Must support multiple retrieval methods (e.g., FAISS - Facebook AI Similarity Search) and scale for 5-10 years.

### Retrieval Pipeline

- **Hybrid Search**: Combines **BM25** (keyword search benchmark, fast for exact matches) with vector embeddings from user queries.
- **Embeddings**: Query converted to embeddings for similarity search (e.g., cosine similarity or HNSW).
- **Top-K Retrieval**: Selects top matching documents/chunks.
- **Advanced Techniques** (not implemented here but mentioned):
  - **Re-ranking**: Compute-intensive; fine-tunes models (e.g., Cohere's re-rankers) on own data to ensure top results are most relevant. Covered in future series.
  - Alternatives to Vector: Classification-based retrieval or simplest methods if use case allows.
- **Objective**: Provide the best context for user queries efficiently, not assuming vector as default.

### API and Frontend

- **API Layer**: FastAPI (Python-based, supports async for I/O-bound tasks, faster performance). Reflects production over Jupyter notebooks (good for experiments but not deployment).
- **Frontend Interface**: Gradio (popular for quick UX; alternatives: Streamlit). User enters query, gets response.
- **Query Flow**:
  - User query → FastAPI → Retrieval (embeddings + hybrid search) → LLM with prompt template (includes retrieved chunks) → Response back to frontend.
- **Async Mechanism**: Improves speed for input/output operations.

### LLM Integration and Observability

- **LLM Role**: Uses retrieved chunks in a prompt template to generate answers. Local setup (e.g., Ollama with Llama 3.2, context window ~100K tokens).
- **Tokenization**: Only relevant for context window limits (truncate if exceeding). No daily token caps like cloud services (e.g., AWS Bedrock's 1M-100M token limits).
- **Observability Tool**: Langfuse for monitoring queries, retrievals, and LLM performance (e.g., what query was asked, how retrieval performed).
- **Logging**: Info, debug, error levels. In production: Push to cloud log management, use tools like DataDog for analysis, PagerDuty for alerts to responsible personnel.
- **Guardrails**: Not implemented here but planned for future (examine query before processing; reject/absurd queries get generic responses).

### Additional Concepts and Best Practices

- **Production Reflections**: System designed to mirror real-world (e.g., ETL for updates, fallbacks, hybrid retrieval, observability). Emphasizes research in tool selection (e.g., databases).
- **Extensibility**: Future additions like summarizing latest papers (via sorting), fine-tuning for ranking (not LLM generation), guardrails.
- **Fine-Tuning Clarification**: For ranking/re-ranking objectives, not general LLM tasks (different from next-word prediction or classification).
- **Alternatives Emphasized**: Not locked to specific tools (e.g., Airflow → others; Docling → Grobid; OpenSearch → Qdrant). Focus on understanding why choices are made.

These notes cover the transcript's key points, with technical terms bolded for emphasis. Review sections sequentially to understand the end-to-end architecture.

:::

::: spoiler Detailed Self
- Importnat thing is to undersant the data source
- Always look for new data
- Atention mechanicim
- Rate Limi
- setup cron job -- metaflow, dagster but logs are important.
- ETL pipeline, step function. rag database up to date, airlfow orchestration tool fetch, parse pdf, ocr, grobid,
- alwasy setup fallback mechnicsum- 2 OCR method, consistency, mulitpel fall back mechanisum based ont he criticality of step.
- chuncks, metatdata -- Update the chuncking stratergy. Use the data from db copy
- opensearch why ..? vector and keyword reterival. Hybrid approch. Sorting very easy on query.
- qudrant, pinecone, weweate ??
- db useful for the next atleast next 5. may be10 years, latency, distrubutes, pre filter, post filtering. HNSW, various reterival method.
- client interface, where user can enter the query like gradio or streamlit.
- fast api at production not juypter. input output format
- bm25 keyword search mechanisum.
- Rag can have mulitple mechnicsum/steps to find the best maching. kohere reranking model on your data
- always this to find the simplest way to find the document
- langfuse llm layer observable 
- pager duty tools
- guardrails- not pass abusurd query to llm
:::


### Video 3 Infrastructure Walkthrough

:::spoiler Self
- Good Folder/Project structure
- compose.yml what cloud would have we have in local
- variables to Specific to airlfow or to software variables.
- UV enviroment managment unsing rust very fast
- RUFF, myPy, pytest, unit test,, integeration test
- pydantic, sql alembic migration script

:::

## Week 2 Notes

:::spoiler Introduction to Data pipeline & ingestion in RAG + QnA 1
- Almost every LLM has RAG.
- Understand use case and extract the data from the proper source identification.May be data from 2,3 sources or website or api
- Dummy question in jupyter and test code
- evalutaion data set, possible user question and there answer
- 4/5 chunking stratergy, compare response.
- Handle if data is not there to answer
- Enrich user query- fix spell- add more keywords enhance, trigger multiple query, again and again with differnt query to make sure if we can find the answer or not. if it can not find, in agetnic system we have writer agent or system push the
- pipeline data engineer role -- we have to coordinate with them
- Ai enginner talk to data engineer. AI enginer starts with basic data scrape and move to data ennginner team
- DBT pipeline/ Dagster
- Latency incerease if the context window is too long/big, more infor than 
- Latency and Halluciantion 
- pipeline standard -- data validation, num-- range, automate using gpt to check the qualitty of data
- Case by case.. if every data data is changing-- no standard way--
- Small POC with chunking and data scraping
- 

:::

:::spoiler Data Ingestion walk through
- Airflow better picture - many companies are using it
- Define schedule, see log of each step
- Fetch only meta data
- Llama Index end to end, llama parser -- replace for dockline
- test 10-15 solution, longivity, reliablity,
- some add to much abstraction- balance with
- Automate as much as we can-- not things happening 1 to 4
- convert table to markdown and then chunk, llm able to porcess better
- create table summary and enrich the embeding
- give llm table and summary
- we messed up something and go back to the backfilling or middle layer
- maintain consistenccy if we have different sources
- concurrency level depends on resource- on cloud pay when machine is on
- Not every
- 


:::

:::spoiler Part 2 QnA
- How do we deal with new data source ?
    - Strong product team, make sure all the data source is listed
    - Maintain a new piple for new datasource. Data source may be available at different time
    - PreProcessing will be differnt/ all data source has differnt metadata
- Tabular data as text or image, which embeding to choose ?
    - KolePali

- Why would we use database migration. Zero downtime
    - Create copy of index and migrate the index. And use new index to retrive the data
    - Very Manual for vector
    - new metadata gets added we need for filtering
    - Embeding vector dimsion get changed 700 to 500. for which we need to add a new field

- Storing data in vector db. Heavy over the network. Pointer in vector or actual data in nosql.
    - it'll add addional call may add network latency. if we dont have any latncey we can just store all the data in vector db
    - Both ways are fine.
- Entire abstact in 1 column. how do we chunck ?
    - sections
    - chuking more like feature engineering. there are various startergy. but choose what make sense to  you. use varios like by paragraph  or by heading or by sction or by new chapter. start with simplest and check the quality of answere. balance between the latency, simpliciy and quality o the answere given by llm. we cn be creative depending on the use case.
    - if we go by the subsection, we make sure we inclde some context from the section as well
-  Agent is reponsible for the creating sql query based on User query.
    -  Handle most frequent query first



:::

# Week 3

:::spoiler
- If we can solve by keyword is good and then move to vector
- Big Miss, Never Jump to vector skipping the keword search
- Old School algo
    - TF
    - ID-TF
    - BM25

- To fetch old data
    - docker exec -it rag-airflow airflow dags backfill arxiv_paper_ingestion --start-date 2024-01-10 --end-date 2024-01-122

- Open search, understand schema, type of field, and there advantages, keyword search, text search.
- we can preporcess, make it lowercase, store data as token, pizza, eat, shantnu etc,-- eating to eat.
- he is eating pizza-- one word/token, for each field we can define certain prepossing, lke store synonym.
- We can setup analyzers, preprossing of data or pre processing of data. We can do preporcess by search engine, we dont need to do much preprocessing by our self in most case before storing to seachengine db.
- we can control scoring of keywords in search query
- we can also search by fuzziness - In case of type 2 chars here and there will also give result
- Possibilty is infinity the way we want to search. We can add filters in data

- Search api using fast api will build search query in opensearch query format = same as openserach query console - Opensearch dependecny injection

- we can't manintain synonyms, fine control keyword search
- Sometimes if we have billions of document Vrctor search might be better, we dont do brute search, HNSW like other machine learning algorithm

#### QNA
- Multilang, Vector search shine, if embeding handle multilingual
- Chat gpt follow up question, just enrich the prompt to ask follow up question
- 2 set of filtering, we add before query without user knowing. ask user to which filter need to aplly, we recommend filters to user.
- IN real world mostly strt with vector search Rag, easy to use no brainer. Hybrid Rag an then we have graph RAG.
- Complex reterivla needs complex RAG.
- Agentic RAG is the Solution to all. 
- Moer Hype Things to sell various RAG or Vector Db- Companies look for simpler solution, not fancy solution
:::
