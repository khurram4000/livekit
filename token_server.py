from flask import Flask, request, jsonify
# from livekit import AccessToken, VideoGrant
from livekit.api import AccessToken, VideoGrants
import os
from flask_cors import CORS
import jwt
import time
import json

app = Flask(__name__)
CORS(app)

# These should be securely stored in environment variables (Cloud vestion livekit)
API_KEY = os.getenv("LIVEKIT_API_KEY", "APIADjFdwpcbRVi")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "5eWuD9DlpHJfAg3nnAryRTEM43tYp0EL5QRTftFRekUC")

# API Key:  APIenwwxTUexakE
# API Secret:  9ipzq8bumiGR0RLTQO4Hbi6mfNYoUe1AdT0YCIf0LYtA

# API_KEY = os.getenv("LIVEKIT_API_KEY", "APIQhTSMgBtsdCm")
# API_SECRET = os.getenv("LIVEKIT_API_SECRET", "MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA")

@app.route("/get-token", methods=["POST"])
def get_token():
    data = request.get_json()
    print("*****************", data)
    identity = data.get("identity", "user")
    room = data.get("room", "voice-assistant-room")
    chat_id = data.get("chatId", "temId2323")
    bot_id = data.get("bot_id", "492")
    domain = data.get("domainName", "liveadmins.com")

    endTime = data.get("endTime", "false")
    websiteId = data.get("websiteId", "")
    websiteURL = data.get("websiteURL", "")
    visitorId = data.get("visitorId", "")
    visitorName = data.get("visitorName", "")
    softwareUserId = data.get("softwareUserId", "")
    userId = data.get("userId", "")
    managerId = data.get("managerId", "")
    timeStamp = data.get("timeStamp", "")
    nickName = data.get("nickName", "")
    miscellaneous = data.get("miscellaneous", "")
    isCustomMessage = data.get("isCustomMessage", "")
    greetId = data.get("greetId", "")
    agent = data.get("agent", "")
    
    payload = {
        "identity": str(identity),
        "room": str(room),
        "chat_id": str(chat_id),
        "bot_id": str(bot_id),
        "domain": str(domain),
        "endTime": str(endTime),
        "websiteId": str(websiteId),
        "websiteURL": str(websiteURL),
        "visitorId": str(visitorId),
        "visitorName": str(visitorName),
        "softwareUserId": str(softwareUserId),
        "userId": str(userId),
        "managerId": str(managerId),
        "timeStamp": str(timeStamp),
        "nickName": str(nickName),
        "miscellaneous": str(miscellaneous),
        "isCustomMessage": str(isCustomMessage),
        "greetId": str(greetId),
        "agent": str(agent),
    }
    print("============PAYLOAD=============>", payload)
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not api_key or not api_secret:
        return {"error": "Server configuration error"}, 500

    # Generate JWT token
    # token = jwt.encode(
    #     {
    #         "iss": api_key,
    #         "name": identity,
    #         "sub": identity,
    #         "exp": int(time.time()) + 3600,
    #         "video": {
    #             "room": room,
    #             "roomJoin": True,
    #         }
    #     },
    #     api_secret,
    #     algorithm="HS256"
    # )
    # print("==========", token)
    token = AccessToken(API_KEY, API_SECRET) \
    .with_identity(identity) \
    .with_metadata(json.dumps({
        "identity": str(identity),
        "room": str(room),
        "chat_id": str(chat_id),
        "bot_id": str(bot_id),
        "domain": str(domain),
        "endTime": str(endTime),
        "websiteId": str(websiteId),
        "websiteURL": str(websiteURL),
        "visitorId": str(visitorId),
        "visitorName": str(visitorName),
        "softwareUserId": str(softwareUserId),
        "userId": str(userId),
        "managerId": str(managerId),
        "timeStamp": str(timeStamp),
        "nickName": str(nickName),
        "miscellaneous": str(miscellaneous),
        "isCustomMessage": str(isCustomMessage),
        "greetId": str(greetId),
        "agent": str(agent),
    })) \
    .with_name(identity) \
    .with_grants(VideoGrants(
        room_join=True,
        room=room,
        # agent="voicebot-ui-aai-eleven",
    )).to_jwt()
    print("=====......=====", token)
    return {"token": token}
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
    # app.run(host="0.0.0.0", port=5002, ssl_context=('cert.pem', 'key.pem'))
