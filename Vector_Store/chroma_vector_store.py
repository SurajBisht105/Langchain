# pip install chromadb tiktoken openai langchain_openai

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Manually structured Document objects for 10 IPL players
doc1 = Document(
    page_content="Virat Kohli: The highest run-getter in IPL history and former captain of RCB.",
    metadata={"team": "RCB", "role": "Batsman", "runs": 8004}
)

doc2 = Document(
    page_content="MS Dhoni: Legendary captain of CSK known for finishing games and 5 IPL titles.",
    metadata={"team": "CSK", "role": "Wicket-keeper Batsman", "titles": 5}
)

doc3 = Document(
    page_content="Rohit Sharma: Led Mumbai Indians to 5 titles and is a prolific top-order batsman.",
    metadata={"team": "MI", "role": "Batsman", "titles": 5}
)

doc4 = Document(
    page_content="David Warner: The most successful overseas batsman with the most fifties in IPL.",
    metadata={"team": "DC", "role": "Batsman", "runs": 6565}
)

doc5 = Document(
    page_content="Suresh Raina: Known as 'Mr. IPL' for his consistent performance over a decade for CSK.",
    metadata={"team": "CSK", "role": "Batsman", "status": "Retired"}
)

doc6 = Document(
    page_content="Lasith Malinga: All-time great pace bowler for MI, famous for his slinging action and yorkers.",
    metadata={"team": "MI", "role": "Bowler", "wickets": 170}
)

doc7 = Document(
    page_content="AB de Villiers: Known as 'Mr. 360' for his ability to hit the ball anywhere for RCB.",
    metadata={"team": "RCB", "role": "Batsman", "strike_rate": 151.68}
)

doc8 = Document(
    page_content="Chris Gayle: The 'Universe Boss' who holds the record for the highest individual score (175*).",
    metadata={"team": "PBKS", "role": "Batsman", "sixes": 357}
)

doc9 = Document(
    page_content="Rashid Khan: A world-class leg-spinner known for his economical bowling and match-winning spells.",
    metadata={"team": "GT", "role": "Bowler", "economy": 6.73}
)

doc10 = Document(
    page_content="Jasprit Bumrah: The spearhead of the MI bowling attack, elite in both powerplay and death overs.",
    metadata={"team": "MI", "role": "Bowler", "best_bowling": "5/10"}
)

# Putting them into a list for use in Chroma
docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory='chroma_db',
    collection_name='sample'
)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=['embeddings','documents','metadatas'])

# search documents
vector_store.similarity_search(
    query='who among these are a bowler?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='who among these are a bowler?',
    k=2
)

# meta-data filtering
vector_store.similarity_search_with_score(
    query='',
    filter={'team': "CSK"}
)