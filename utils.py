import requests

def get_conversation(chatid: str):
    url = f"https://blue.thelivechatsoftware.com/ChatAppApi/api/botfront/GetEmailConversationForBot?chatid={chatid}"
    
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
        return prior       # Return parsed JSON response
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []

# chat_id = "920880"
# conversation = get_conversation(chat_id)
# if conversation:
#     print("Response received:")
#     print(conversation)
# else:
#     print("No response or an error occurred.")
