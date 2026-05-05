"""
Raw WebSocket SignalR Manager - Bypasses signalrcore library issues
Implements SignalR protocol directly for better control
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, Callable
import requests
from urllib.parse import urlencode
import websockets
from websockets.client import WebSocketClientProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawSignalRManager:
    """
    Raw WebSocket implementation of SignalR client
    Bypasses signalrcore library to handle protocol directly
    """
    
    def __init__(self, hub_url: str, auth_token: str = None, hub_name: str = "chathub"):
        """
        Initialize Raw SignalR Manager
        
        Args:
            hub_url: The SignalR hub endpoint URL (wss:// or https://)
            auth_token: Optional authentication token
            hub_name: SignalR hub name (default: chathub)
        """
        self.hub_url = self._normalize_url(hub_url)
        self.auth_token = auth_token
        self.hub_name = hub_name
        self.websocket: Optional[WebSocketClientProtocol] = None
        self._is_connected = False
        self._connection_lock = asyncio.Lock()
        self._connection_id = None
        self._connection_token = None
        self._message_id = 0
        self._event_handlers = {}
        self._receive_task = None
        
        # Headers for authentication
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL format"""
        url = url.rstrip('/')
        
        # Convert to HTTPS for negotiation
        if url.startswith('wss://'):
            url = 'https://' + url[6:]
        elif url.startswith('ws://'):
            url = 'http://' + url[5:]
        elif not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url
    
    def _get_websocket_url(self, http_url: str) -> str:
        """Convert HTTP URL to WebSocket URL"""
        if http_url.startswith('https://'):
            return 'wss://' + http_url[8:]
        elif http_url.startswith('http://'):
            return 'ws://' + http_url[7:]
        return http_url
    
    async def negotiate(self) -> Dict[str, Any]:
        """Perform SignalR negotiation"""
        negotiate_url = f"{self.hub_url}/negotiate"
        
        params = {
            'clientProtocol': '1.5',
            'connectionData': json.dumps([{'name': self.hub_name}])
        }
        
        full_url = f"{negotiate_url}?{urlencode(params)}"
        logger.info(f"🔄 Negotiating: {full_url}")
        
        try:
            response = await asyncio.to_thread(
                requests.post,
                full_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self._connection_id = data.get('ConnectionId')
                self._connection_token = data.get('ConnectionToken')
                
                logger.info(f"✅ Negotiation successful")
                logger.info(f"   Connection ID: {self._connection_id}")
                logger.info(f"   Protocol Version: {data.get('ProtocolVersion', 'N/A')}")
                
                return data
            else:
                logger.error(f"❌ Negotiation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Negotiation error: {e}")
            return None
    
    def _build_connect_url(self) -> str:
        """Build WebSocket connection URL with all required parameters"""
        ws_base = self._get_websocket_url(self.hub_url)
        
        params = {
            'transport': 'webSockets',
            'clientProtocol': '1.5',
            'connectionToken': self._connection_token,
            'connectionData': json.dumps([{'name': self.hub_name}]),
            'tid': '10'  # Transport ID
        }
        
        return f"{ws_base}/connect?{urlencode(params)}"
    
    async def _start_connection(self) -> bool:
        """Send the start command after WebSocket connection"""
        start_url = f"{self.hub_url}/start"
        
        params = {
            'transport': 'webSockets',
            'clientProtocol': '1.5',
            'connectionToken': self._connection_token,
            'connectionData': json.dumps([{'name': self.hub_name}])
        }
        
        full_url = f"{start_url}?{urlencode(params)}"
        
        try:
            response = await asyncio.to_thread(
                requests.get,
                full_url,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Start command successful: {data}")
                return True
            else:
                logger.error(f"❌ Start command failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Start command error: {e}")
            return False
    
    async def _receive_loop(self):
        """Continuously receive and process messages"""
        try:
            async for message in self.websocket:
                await self._process_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
            self._is_connected = False
        except Exception as e:
            logger.error(f"❌ Receive loop error: {e}")
            self._is_connected = False
    
    async def _process_message(self, message: str):
        """Process incoming SignalR message"""
        try:
            logger.debug(f"📨 Received: {message}")
            
            # SignalR messages can be empty (keep-alive)
            if not message or message == "{}":
                return
            
            data = json.loads(message)
            
            # Handle initialization message
            if 'S' in data:  # S = Success (connection initialized)
                logger.info("✅ SignalR connection initialized")
                self._is_connected = True
                return
            
            # Handle hub invocation messages
            if 'M' in data:  # M = Messages
                for msg in data['M']:
                    method = msg.get('M')  # Method name
                    args = msg.get('A', [])  # Arguments
                    
                    logger.info(f"📩 Hub method: {method}")
                    
                    # Call registered handler
                    if method in self._event_handlers:
                        await self._event_handlers[method](*args)
            
            # Handle errors
            if 'E' in data:
                logger.error(f"❌ SignalR error: {data['E']}")
            
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Non-JSON message: {message}")
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
    
    async def connect(self, max_retries: int = 3) -> bool:
        """Establish SignalR connection"""
        async with self._connection_lock:
            if self._is_connected:
                logger.info("Already connected")
                return True
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Connection attempt {attempt + 1}/{max_retries}")
                    
                    # Step 1: Negotiate
                    if not await self.negotiate():
                        continue
                    
                    # Step 2: Connect WebSocket
                    ws_url = self._build_connect_url()
                    logger.info(f"🔌 Connecting to: {ws_url[:100]}...")
                    
                    # Add auth header to WebSocket connection
                    extra_headers = self.headers if self.headers else None
                    
                    # self.websocket = await websockets.connect(
                    #     ws_url,
                    #     extra_headers=extra_headers,
                    #     ping_interval=20,
                    #     ping_timeout=10
                    # )
                    connect_kwargs = dict(
                        ping_interval=20,
                        ping_timeout=10,
                    )
                    if extra_headers:
                        connect_kwargs["additional_headers"] = extra_headers  # websockets>=13.0 syntax

                    self.websocket = await websockets.connect(ws_url, **connect_kwargs)
                    
                    logger.info("🔗 WebSocket connected")
                    
                    # Step 3: Start SignalR connection
                    if not await self._start_connection():
                        await self.websocket.close()
                        continue
                    
                    # Step 4: Start receive loop
                    self._receive_task = asyncio.create_task(self._receive_loop())
                    
                    # Wait for initialization
                    for _ in range(10):
                        if self._is_connected:
                            logger.info("✅ SignalR fully connected!")
                            return True
                        await asyncio.sleep(0.5)
                    
                    logger.warning("⚠️ Connected but not initialized")
                    
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                    
                    if self.websocket:
                        await self.websocket.close()
                        self.websocket = None
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            
            return False
    
    async def disconnect(self):
        """Close SignalR connection"""
        async with self._connection_lock:
            self._is_connected = False
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
                self._receive_task = None
            
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            logger.info("🔌 Disconnected from SignalR")
    
    async def send_message(self, method: str, args: list) -> bool:
        """
        Send message to SignalR hub
        
        Args:
            method: Hub method name
            args: List of arguments for the method
        """
        if not self._is_connected or not self.websocket:
            logger.error("❌ Not connected")
            return False
        
        try:
            self._message_id += 1
            
            # Build SignalR message format
            message = {
                'H': self.hub_name,  # Hub name
                'M': method,         # Method name
                'A': args,           # Arguments
                'I': str(self._message_id)  # Message ID
            }
            
            message_json = json.dumps(message)
            logger.info(f"📤 Sending: {method}")
            logger.debug(f"   Data: {message_json}")
            
            await self.websocket.send(message_json)
            logger.info("✅ Message sent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Send error: {e}")
            return False
    
    def on(self, event: str, callback: Callable):
        """Register event handler"""
        self._event_handlers[event] = callback
        logger.info(f"📝 Registered handler for: {event}")


async def diagnose_raw_signalr(hub_url: str, auth_token: str = None):
    """Diagnostic tool for raw SignalR connection"""
    print("\n" + "="*60)
    print("🔍 Raw SignalR Connection Diagnostics")
    print("="*60)
    
    manager = RawSignalRManager(hub_url, auth_token)
    
    print(f"\n1️⃣ URL Configuration:")
    print(f"   Hub URL: {manager.hub_url}")
    print(f"   WebSocket: {manager._get_websocket_url(manager.hub_url)}")
    print(f"   Hub Name: {manager.hub_name}")
    
    print(f"\n2️⃣ Testing negotiation...")
    result = await manager.negotiate()
    print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
    
    print(f"\n3️⃣ Testing full connection...")
    connected = await manager.connect(max_retries=1)
    print(f"   Result: {'✅ Connected' if connected else '❌ Failed'}")
    
    if connected:
        print(f"\n4️⃣ Testing message send...")
        
        # Register a test handler
        def test_handler(*args):
            print(f"   📨 Received callback: {args}")
        
        manager.on('testResponse', test_handler)
        
        # Try sending a test message
        sent = await manager.send_message('testMethod', ['Hello', 'World'])
        print(f"   Send result: {'✅ Sent' if sent else '❌ Failed'}")
        
        # Wait a bit for any responses
        await asyncio.sleep(3)
    
    await manager.disconnect()
    
    print("\n" + "="*60)
    print("Diagnostics complete")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    async def test():
        hub_url = "wss://blue.thelivechatsoftware.com/signalrserver/signalr"
        
        # Run diagnostics
        await diagnose_raw_signalr(hub_url)
        
        # Example of actual usage
        print("\n" + "="*60)
        print("Example Usage")
        print("="*60 + "\n")
        
        manager = RawSignalRManager(hub_url)
        
        # Register message handlers
        def on_message_received(message):
            print(f"📨 Message: {message}")
        
        manager.on('messageReceived', on_message_received)
        
        # Connect
        if await manager.connect():
            # Send a message
            await manager.send_message('sendMessage', [{
                'ChatId': 'test-123',
                'MessageBody': 'Hello from Python!',
                'EndTime': 'false'
            }])
            
            # Keep connection alive
            await asyncio.sleep(10)
            
            # Disconnect
            await manager.disconnect()
    
    asyncio.run(test())