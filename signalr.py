# from signalrcore.hub_connection_builder import HubConnectionBuilder
# import json
# import time
# url = "wss://blue.thelivechatsoftware.com/signalrserver/signalr"

# # Create SignalR connection
# hub_connection = HubConnectionBuilder()\
#     .with_url(url)\
#     .build()
# print("=========Connection build========")
# # hub_connection.start()

# print("=========Connection started========")
# # # Replace these with your variables
# # variablesForChat = {
# #     "identity": "web-user-8096", 
# #     "room": "voice-roomr-718", 
# #     "chat_id": "263423", 
# #     "bot_id": "",
# #     "domain": "testing.webgreeter.com/zem/hulk", ###actual is ‘DomainName’
# #     "endTime": "false", 
# #     "websiteId": 14, 
# #     "websiteURL":"https://clientdemo.webgreeter.com/AudioBot/",
# #     "visitorId": "21941282", 
# #     "visitorName": "Visitor21941282", 
# #     "softwareUserId": "5253", 
# #     "userId": "5253", 
# #     "managerId": "5253", 
# #     "timeStamp": "", 
# #     "nickName": "Pebble",
# #     "miscellaneous": "Pebble", 
# #     "isCustomMessage": "GreetMessage", 
# #     "greetId": "48316380",
# #     "agent": "autos_open_dev"   ###actual is ‘Agent’ but coming in metadata ‘agent’
# # }


# # language = "en"
# # msg = "This is a test message"
# # parentTitle = "Parent Page Title"

# # Function to build a message packet
# def build_message_packet(variablesForChat, end_time="false", message_body=""):
#     return {
#         "ChatId": variablesForChat["chat_id"],
#         "EndTime": end_time,
#         "WebsiteId": variablesForChat["websiteId"],
#         "WebsiteURL": variablesForChat["WebsiteURL"], 
#         "VisitorId": variablesForChat["visitorId"],
#         "VisitorName": variablesForChat["visitorName"],
#         "SoftwareUserId": variablesForChat["softwareUserId"],
#         "UserId": variablesForChat["UserId"],
#         "ManagerId": variablesForChat["ManagerId"],
#         "TimeStamp": "",
#         "NickName": variablesForChat["NickName"],
#         "Miscellaneous": variablesForChat["miscellaneous"],
#         "IsCustomMessage": variablesForChat["IsCustomMessage"],
#         "Agent": variablesForChat["agent"],
#         "Lang": "en",
#         "DomainName": variablesForChat["domain"],
#         "ServerURL": variablesForChat["ServerURL"],
#         "MessageBody": message_body
#     }

# # # Send normal message
# # sendMessagePacket = build_message_packet(
# #     variablesForChat, 
# #     end_time="false", 
# #     message_body="looking  for car"
# #     )
# # hub_connection.send("sendMessageForBotChatGeneric", [sendMessagePacket])

# # # Send invitation message
# # invMessagePacket = build_message_packet(
# #     variablesForChat,
# #     end_time="false", 
# #     message_body="", 
# #     is_custom="GreetMessage"
# #     )
# # invMessagePacket.update({
# #     "UserId": variablesForChat["actualUserId"],
# #     "ManagerId": variablesForChat["actualUserId"],
# #     "NickName": variablesForChat["userName"],
# #     "GreetId": variablesForChat["greetId"]
# # })
# # hub_connection.send("sendInvitationForBotChatGeneric", [invMessagePacket])

# # # Send first message
# # firstMessagePacket = build_message_packet(end_time="false", message_body=variablesForChat["first_message_text"], is_custom="FirstMessage")
# # firstMessagePacket.update({
# #     "GreetId": variablesForChat["greetId"]
# # })
# # hub_connection.send("sendInvitationForBotChatGeneric", [firstMessagePacket])

# # hub_connection.stop()
# # print("=========sent and closed========")

import asyncio
from livekit import api
# Import the protobuf types for egress
from livekit.protocol.egress import ListEgressRequest

async def main():
    lkapi = api.LiveKitAPI(
        "https://botdev1.thelivechatsoftware.com",
        api_key="APIQhTSMgBtsdCm",
        api_secret="MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA",
    )

    egress = lkapi.egress
    # Create request object with filters (or empty for all)
    req = ListEgressRequest(
        room_name="your_room_name",   # or omit if you want all
        active=True                   # optional
    )
    # Call the method
    resp = await egress.list_egress(req)
    print("Egress list:", resp)

    # Close session
    await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(main())



# from livekit import api
# lkapi = api.LiveKitAPI("https://botdev1.thelivechatsoftware.com", api_key="APIQhTSMgBtsdCm", api_secret="MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA")
# egress = lkapi.egress

# print(egress.list_egress())