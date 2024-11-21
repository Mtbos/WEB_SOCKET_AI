'''
import asyncio
import websockets

async def test_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Example data: 10 comma-separated numeric values
        data = "0.12,0.23,0.43,0.64,0.59,0.65,0.1347,0.56748,0.99,1.0"
        print(f"Sending: {data}")
        await websocket.send(data)

        # Receive and print the response
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(test_client())
'''
# Can Use this file if needed
import asyncio
import websockets
import numpy as np
import json

async def send_message(uri, data):
    """
    Sends a message to the WebSocket server, waits for a response.
    """
    try:
        # Connect to the WebSocket server
        async with websockets.connect(uri) as websocket:
            # Send the message (comma-separated values)
            message = ','.join(map(str, data))
            await websocket.send(message)

            # Wait for the response
            response = await websocket.recv()
            print(f"Server response: {response}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    # Prepare some dummy data for the prediction (values between 0 and 1)
    data = np.random.rand(10).tolist()  # 10 random features in the range [0, 1]
    
    # Define the server address and path (choose between /regression or /classification)
    server_address = "ws://localhost:8765"
    
    # Choose the path (either /regression or /classification)
    path = "/regression"  # Change to "/classification" for classification model
    
    uri = f"{server_address}{path}"
    
    print(f"Sending data to {uri}")
    await send_message(uri, data)

if __name__ == "__main__":
    asyncio.run(main())
