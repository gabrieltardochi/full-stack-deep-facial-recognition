import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
MF_RUN_ID = os.environ["MF_RUN_ID"]
ES_INDEX = os.environ["ES_INDEX"]
ES_URL = os.environ["ES_URL"]
