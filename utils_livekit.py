import pickle
import requests
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from pathlib import Path
import os
from redis_utils_Wg import load_conversation_from_redis, redis_client, save_conversation_to_redis, \
    Conversation

BASE_DIR = "/home/chatsystem/botserver/botgen/"
DATA_FILES_DIR = os.path.join(BASE_DIR, 'ConversationalAIWG/DataFilesWG')


def contains_files(folder_path):
    try:
        # Get a list of all items in the directory
        print(f"folder_path: {folder_path}")
        items = os.listdir(folder_path)
        print(f"items: {len(items)}")

        # Check if any of the items is a file (not a directory)
        for item in items:
            if os.path.isfile(os.path.join(folder_path, item)):
                return True  # Folder contains at least one file
        return False  # Folder contains no files
    except FileNotFoundError as e:
        print(f"Error: The folder '{folder_path}' was not found. {e}")
        return False
    except PermissionError as e:
        print(f"Error: Permission denied to access '{folder_path}'. {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking folder '{folder_path}': {e}")
        return False


def build_storage(directory):
    scrapped_dir = os.path.join(directory, 'scrapped')
    uploads_dir = os.path.join(directory, 'uploads')
    text_dir = os.path.join(directory, 'text-file')
    persist_dir = os.path.join(directory, 'storage')
    print(f"persist_dir: {persist_dir}")

    print(f"Status os.path.exists(directory): {os.path.exists(directory)}, {directory}")

    documents_scrapped = []
    documents_uploads = []
    documents_texts = []

    try:
        print(f"Status os.path.exists(scrapped_dir): {os.path.exists(scrapped_dir)}")
        if os.path.exists(scrapped_dir) and contains_files(scrapped_dir):
            documents_scrapped = SimpleDirectoryReader(scrapped_dir).load_data()
            print(f"Scrapped document length: {len(documents_scrapped)}, type: {type(documents_scrapped)}")

        if os.path.exists(uploads_dir) and contains_files(uploads_dir):
            documents_uploads = SimpleDirectoryReader(uploads_dir).load_data()
            print(f"uploads_dir document length: {len(documents_uploads)}, type: {type(documents_uploads)}")

        if os.path.exists(text_dir) and contains_files(text_dir):
            documents_texts = SimpleDirectoryReader(text_dir).load_data()
            print(f"text document length: {len(documents_texts)}, type: {type(documents_texts)}")

        documents = documents_scrapped + documents_uploads + documents_texts
        print(f"document length: {len(documents)}")

        if documents:
            # Create and persist index
            index = GPTVectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir)
            print("Vector Storage Built")
            return True
        else:

            return False

    except FileNotFoundError as e:
        print(f"Error: One of the directories does not exist. {e}")
        return False
    except PermissionError as e:
        print(f"Error: Permission denied to access the directory. {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during the storage build: {e}")
        return False


def read_from_storage(persist_dir):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Error reading from storage: {e}")
        return None


def fetch_context(q, con, botid, Domain):
    print("QUERY::::::::::::::::::::>>>>", q)
    # Define persistent storage and data directories
    global DATA_FILES_DIR
    persist_dir = os.path.join(DATA_FILES_DIR, botid, Domain, 'storage')
    data_dir = os.path.join(DATA_FILES_DIR, botid, Domain)

    # If no valid data directory, return empty context
    if not data_dir:
        return ""

    print(f"Using Data Directory: {data_dir}")

    # Fetch bot_id from conversation object
    Agent = str(botid) + "_" + str(Domain)
    bot_id = con.Agent
    bot_index_key = f"{Agent}_index"

    index = None  # Initialize index

    # 1️⃣ **Check if Index is Already Cached in Memory (Best Performance)**
    if hasattr(con, "myIndex") and con.myIndex is not None:
        index = con.myIndex
        print(f"✅ Using Cached Index for {bot_id}")

    # 1️⃣ **First, try Redis (Primary Source)**
    elif redis_client.exists(bot_index_key):
        index = pickle.loads(redis_client.get(bot_index_key))
        if hasattr(index, "as_retriever"):
            con.myIndex = index  # Cache in memory
            print(f"✅ Loaded Vector Store from Redis for {bot_id}")
        else:
            print(f"⚠️ Warning: Invalid vector store in Redis. Will attempt to load from storage.")
            index = None  # Mark Redis data as invalid so we load from storage

    # 2️⃣ **If Redis fails, try Persistent Storage**
    elif index is None and os.path.exists(persist_dir):
        index = read_from_storage(persist_dir)
        if hasattr(index, "as_retriever"):
            con.myIndex = index  # Cache in memory
            redis_client.set(bot_index_key, pickle.dumps(index))  # ✅ Save to Redis for future use
            print(f"✅ Loaded Vector Store from Disk and Cached in Redis for {bot_id}")
        else:
            print(f"⚠️ Warning: Invalid vector store in Storage. Will attempt to rebuild.")
            index = None  # Mark storage data as invalid

    # 3️⃣ **As a last resort, rebuild it only if necessary**
    elif index is None:
        print(f"⏳ No valid index found. Building new vector store for {bot_id}...")
        index = build_storage(data_dir)
        con.myIndex = index
        redis_client.set(bot_index_key, pickle.dumps(index))  # ✅ Save to Redis for future use
        print(f"✅ Built and Cached New Vector Store for {bot_id}")

    print(f"Vector Store Ready for {bot_id}")

    # Ensure index is a valid retriever object before calling `as_retriever`
    if not hasattr(index, "as_retriever"):
        print("❌ Error: Retrieved object is not a valid vector store.")
        return ""

    # Retrieve context from the vector store
    retriever = index.as_retriever(choice_batch_size=5, context_length=500)
    nodes = retriever.retrieve(q)
    context = nodes[0].text if nodes else ""  # Return empty context if no nodes found

    return context


def get_conversation(chatid: str):
    url = f"https://blue.thelivechatsoftware.com/ChatAppApi/api/botfront/GetEmailConversationForBot?chatid={chatid}"
    # url = f"https://blue.thelivechatsoftware.com/ChatAppApi/api/botfront/GetEmailConversationForBot?chatid=920880"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        conversation = response.json()["data"]
        prior = []
        for conv in conversation:
            if conv["userId"] != 0:
                prior.append({
                    "role": "assistant",
                    "content": conv["message"]
                })
            else:
                prior.append({
                    "role": "user",
                    "content": conv["message"]
                })
        print(prior)
        return prior  # Return parsed JSON response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []


def getBotFlow(botId):
    url = "https://dev1.thelivechatsoftware.com/ChatAppApi/api/botai/getbotflow"
    params = {'botId': botId}
    headers = {'accept': 'text/plain'}
    try:
        # Send the GET request
        response = requests.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Assuming the response has a 'data' field
                bot_flow_data = response.json().get('data', None)
                if bot_flow_data is not None:
                    return response.json().get('data', None)
                else:
                    print("No 'data' found in the response.")
                    return None
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
            except KeyError as e:
                print(f"Missing expected key in JSON response: {e}")
            return None
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch all exceptions related to the request (e.g., network issues, timeout)
        print(f"Request failed: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        return None


def getBotPrompt(botId):
    url = "https://dev1.thelivechatsoftware.com/ChatAppApi/api/botai/botpromptlistbybotid"
    params = {'botId': botId}
    headers = {'accept': 'text/plain'}

    try:
        # Send the GET request
        response = requests.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            try:
                # Assuming the response has a 'data' field
                bot_prompts = response.json().get('data', None)
                if bot_prompts:
                    for botpromot in bot_prompts:
                        if botpromot['botPromptTypeName'] == 'Bot Prompt':
                            return botpromot['prompt']
                    return ''
            except ValueError as e:
                print(f"Error parsing JSON: {e}")
            except KeyError as e:
                print(f"Missing expected key in JSON response: {e}")
            return ''
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return ''
    except requests.exceptions.RequestException as e:
        # Catch all exceptions related to the request (e.g., network issues, timeout)
        print(f"Request failed: {e}")
        return ''
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        return ''



def fetch_tree(con, botId):
    if con.tree:
        return con.tree
    else:
        data = getBotFlow(botId)
        if data:
            generatedTree = data.get('generatedFlow', '')
            con.tree = generatedTree
            return generatedTree
        else:
            return ''


def fetch_Instructions(con, botId):
    if con.Instructions:
        return con.Instructions
    else:
        data = getBotPrompt(botId)
        if data:
            con.Instructions = data
            return data
        else:
            return ''
