import aiohttp
import asyncio

async def post_message_to_conversation(
    chat_id: int,
    message: str,
    user_id: int,
    manager_id: int,
    visitor_id: int,
    website_id: str,
    nick_name: str,
    operator_name: str
):
    """
    Asynchronously sends a message to a conversation using aiohttp.
    """
    url = f"https://dev1.thelivechatsoftware.com/ChatAppApi/api/botai/postmessagetoconversation"

    params = {
        "access-token": "ICNCbRfQ2U2aNqe7J32fpvT9EPuZSa4GGcwRLUnmEBWPeSNV7TT"
    }

    headers = {
        "api-version": "1.0",
        "cache-control": "no-cache",
        "Content-Type": "application/json"
    }

    # payload = {
    #     "chatId": chat_id,
    #     "message": message,
    #     "userId": user_id,
    #     "managerId": manager_id,
    #     "visitorId": visitor_id,
    #     "operatorName": operator_name
    # }
    payload = {
        "chatId": chat_id,
        "messageBody": message,
        "userId": user_id,
        "managerId": manager_id,
        "visitorId": visitor_id,
        "websiteId": website_id,
        "nickName": nick_name,
        "isCustomMessage": "Data",
        "chatType": "voicebot",
        "operatorName": operator_name
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, params=params, headers=headers, json=payload) as response:
                if response.status == 200:
                    print("✅ Message sent successfully (background task)")
                    data = await response.json()
                    print("RESPONSE :", data)
                else:
                    text = await response.text()
                    print(f"❌ Failed [{response.status}]: {text}")
        except Exception as e:
            print(f"⚠️ Background task error: {e}")
