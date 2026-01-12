# ! pip Install -q youtube-transcript-api language-community langchain-openai faiss-cpu tiktoken python-dotenv

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAXZ-ab-YpzEH60feQZtQpV58f9WeeVeJo"

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

video_id = "Gfr50f6ZBvo"   #   only the ID, not full URL
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    transcript = " ".join(chunk ["text"] for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video.")