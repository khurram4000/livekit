import json
import os
import sys
import redis
import time
import pickle
import logging
import redis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
redis_client = redis.StrictRedis(host="localhost",
                                 port=6377,
                                 #port=6379,
                                 db=0,
                                 decode_responses=False)


print("Radis Connection built...")

class Conversation:
    """Base class for conversations with history management."""

    def __init__(self, bot_id, Agent):
        self.bot_id = bot_id  # Associate conversation with a bot
        self.Agent = Agent
        self.conversation_history = []
        self.myIndex = None  # Now inherited by child classes
        self.tree = ''
        self.Instructions = ''
        self.name = ""
        self.phone_number = ""
        self.email = ""

    def add_system_message(self, system_message):
        self.conversation_history.append({'role': 'system', 'content': system_message})

    def add_user_message(self, user_message):
        self.conversation_history.append({'role': 'user', 'content': user_message})

    def add_assistant_message(self, assistant_message):
        self.conversation_history.append({'role': 'assistant', 'content': assistant_message})

    @classmethod
    def from_dict(cls, data):
        """Create a Conversation object from a dictionary."""
        con = cls(bot_id=data.get('bot_id', None), Agent=data.get('Agent', None))
        con.Instructions = data.get('Instructions', '')
        con.tree = data.get('tree', '')
        con.conversation_history = data.get('conversation_history', [])
        con.name = data.get('name', '')
        con.email = data.get('email', '')
        con.phone_number = data.get('phone_number', '')
        return con

def save_conversation_to_redis(cid, conversation, is_chat_ended=False):
    conversation_data = {
        "type": "Conversation",
        "bot_id": conversation.bot_id,
        "Agent": conversation.Agent,
        "conversation_history": conversation.conversation_history,
        "name": conversation.name,
        "phone_number": conversation.phone_number,
        "email": conversation.email,
        "last_active_at": int(time.time()),
        "tree": conversation.tree,
        "Instructions": conversation.Instructions,
        "is_chat_ended": is_chat_ended,
    }

    redis_client.set(cid, json.dumps(conversation_data))
    redis_client.expire(cid, 3600)

    # IMPORTANT: Project 2 should NOT save bot index
    return "ok"


def load_conversation_from_redis(cid):
    """
    Safe loader (NO deletion / NO touching index key):
    - Loads conversation JSON.
    - Never deletes/updates the bot index key.
    - If index unpickle fails, it simply skips loading myIndex (sets it to None).
    - Cleans up stale/ended chat by deleting ONLY the chat cid key (optional behavior kept).
    """
    try:
        conversation_data = redis_client.get(cid)
        if not conversation_data:
            return None

        # Redis returns bytes when decode_responses=False
        if isinstance(conversation_data, (bytes, bytearray)):
            conversation_data = conversation_data.decode("utf-8", errors="ignore")

        data = json.loads(conversation_data)

        # Build Conversation object
        con = Conversation.from_dict(data)

        # Stale / ended chat cleanup (only affects cid key, not bot index)
        last_active_at = data.get("last_active_at", 0)
        is_chat_ended = data.get("is_chat_ended", False)
        if is_chat_ended or (int(time.time()) - int(last_active_at) > 3600):
            redis_client.delete(cid)
            return None

        # Best-effort load of shared bot index (DO NOT delete/update key on failure)
        con.myIndex = None
        bot_index_key = f"{con.Agent}_index"
        raw_index = redis_client.get(bot_index_key)

        if raw_index:
            try:
                con.myIndex = pickle.loads(raw_index)
            except Exception as e:
                # Do not touch Redis key; just skip loading index
                print(f"[WARN] Failed to unpickle bot index '{bot_index_key}', skipping. Error: {e}")
                con.myIndex = None

        return con

    except redis.exceptions.ConnectionError as e:
        print(f"Redis Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data from Redis: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in load_conversation_from_redis: {e}")
        return None


# def save_conversation_to_redis(cid, conversation, is_chat_ended=False):
#     conversation_type = "Conversation"

#     conversation_data = {
#         'type': conversation_type,
#         'bot_id': conversation.bot_id,
#         'Agent': conversation.Agent,
#         'conversation_history': conversation.conversation_history,
#         'name': conversation.name,
#         'phone_number': conversation.phone_number,
#         'email': conversation.email,
#         'last_active_at': int(time.time()),
#         'tree': conversation.tree,
#         'Instructions': conversation.Instructions,
#         'is_chat_ended': is_chat_ended,
#     }
#     print("*-*-*-*-*-*-*-*-* Conversation Data to be saved to Redis *-*-*-*-*-*-*-*-*-*", conversation_data)
#     try:
#         redis_client.set(cid, json.dumps(conversation_data))
#         redis_client.expire(cid, 3600)  # Set expiry (1 hour)

#         # Store myIndex for bot (one per bot, shared among all chats)
#         bot_index_key = f"{conversation.Agent}_index"
#         if conversation.myIndex and not redis_client.exists(bot_index_key):
#             redis_client.set(bot_index_key, pickle.dumps(conversation.myIndex))
#             redis_client.expire(bot_index_key, 3600)  # Expire bot index after 1 hour
#         elif conversation.myIndex:
#             # Update the bot index if it changes and reset expiration
#             redis_client.set(bot_index_key, pickle.dumps(conversation.myIndex))
#             redis_client.expire(bot_index_key, 3600)  # Reset expiration
#         return f"following conversation data has been set to Redis :\n {conversation_data}"
#     except redis.exceptions.ConnectionError as e:
#         print(f"Redis Error: {e}")
#         return f"Faild to set conversation data to Redis, Error: {e}"
