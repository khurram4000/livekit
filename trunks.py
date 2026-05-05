import asyncio
import os
from livekit import api
from livekit.protocol.sip import CreateSIPOutboundTrunkRequest, SIPOutboundTrunkInfo, ListSIPOutboundTrunkRequest


async def create_trunk(name: str, address: str, numbers: list[str],):
    # Use environment variables if not provided
    api_key = "APIQhTSMgBtsdCm"
    api_secret = "MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA"
    # api_url = "wss://botdev1.thelivechatsoftware.com"
    api_url = "http://localhost:7880"

    # Initialize LiveKit API client
    lkapi = api.LiveKitAPI(
        url=api_url,
        api_key=api_key,
        api_secret=api_secret,
    )

    # Build trunk info
    trunk_info = SIPOutboundTrunkInfo(
        name=name,
        address=address,
        numbers=numbers,
    )

    # Create trunk request
    req = CreateSIPOutboundTrunkRequest(trunk=trunk_info)
    resp = await lkapi.sip.create_sip_outbound_trunk(req)
    print("==========Response: ",resp)
    print(f"✅ Created trunk '{name}' successfully!")
    # print(f"Trunk ID: {resp.trunk.id}")
    # print(f"Trunk Info: {resp.trunk}")

    await lkapi.aclose()
    # return resp.trunk
    return resp

async def list_trunks():
    api_key = "APIQhTSMgBtsdCm"
    api_secret = "MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA"
    # api_url = "wss://botdev1.thelivechatsoftware.com"
    api_url = "http://localhost:7880"


    # Initialize LiveKit API client
    lkapi = api.LiveKitAPI(
        url=api_url,
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        # Create request to list trunks
        req = ListSIPOutboundTrunkRequest()
        # resp = await lkapi.sip.list_sip_outbound_trunks(req)
        resp = await lkapi.sip.list_sip_outbound_trunk(req)
        print("***********", resp)
        # Convert to Python list of dictionaries
        trunks = []
        for item in resp.items:
            trunks.append({
                "sip_trunk_id": item.sip_trunk_id,
                "name": item.name,
                "address": item.address,
                "transport": str(item.transport),
                "numbers": list(item.numbers),  # repeated field
            })
        print(trunks)
        return trunks  # ✅ return instead of printing

    except Exception as e:
        print("Error listing trunks:", e)
        return []
    finally:    
        await lkapi.aclose()

async def delete_trunk(trunk_id: str):
    """
    Delete a SIP outbound trunk using its trunk ID.

    Args:
        trunk_id (str): The ID of the trunk to delete.
    """
    api_key = "APIQhTSMgBtsdCm"
    api_secret = "MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA"
    # api_url = "wss://botdev1.thelivechatsoftware.com"
    api_url = "http://localhost:7880"

    # Initialize LiveKit API client
    lkapi = api.LiveKitAPI(
        url=api_url,
        api_key=api_key,
        api_secret=api_secret,
    )

    try:
        req = api.DeleteSIPTrunkRequest(sip_trunk_id=trunk_id)
        resp = await lkapi.sip.delete_sip_trunk(req)
        print(f"✅ Successfully deleted trunk {trunk_id}")
        return resp
    except Exception as e:
        print(f"Error deleting trunk {trunk_id}:", e)
    finally:
        await lkapi.aclose()


# Example usage



if __name__ == "__main__":
    # Create OutBound Trunk
    # asyncio.run(
    #     create_trunk(
    #         name="Outbound Call Trunk - Campaign A",
    #         address="208.109.214.11",
    #         numbers=["303100", "303101", "303102"],
    #         # auth_username="sip_user",
    #         # auth_password="sip_pass",
    #         # api_key="APIQhTSMgBtsdCm",
    #         # api_secret="MsAjH3SgH1eVkPc1i8DfwSwlPUTBhNyMsJcT5LXeLmLA",
    #         # api_url="wss://botdev1.thelivechatsoftware.com",
    #     )
    # )

    # List Outbound Trunks
    asyncio.run(list_trunks())

    # Delete Outbound Trunk
    # for i in ["ST_Ha3YV7PMd2eB", "ST_UbDwgHU35cTm", "ST_XZ7a9GpQDFsg", "ST_gGNWXiAagyzi", "ST_hgPGpJJDRawE", "ST_kAXVWAfrYvG8", "ST_nQAwwtHpjYUr"]:

    #     asyncio.run(delete_trunk(i))
    
