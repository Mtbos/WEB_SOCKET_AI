'''
import asyncio
import websockets
import numpy as np
import tensorflow as tf

# Load the pre-trained AI model
model = tf.keras.models.load_model("models/model.h5")

def validate_message(message):
    """
    Validates incoming messages. Ensure it is numeric data (as an example).
    """
    try:
        data = np.array(message.split(','), dtype=float)
        return data
    except ValueError:
        return None

def preprocess(data):
    """
    Preprocesses the incoming data for model prediction.
    Assumes data needs to be reshaped to match the model input.
    """
    return np.expand_dims(data, axis=0)  # Example: Add batch dimension

async def handle_connection(websocket):
    """
    Handles WebSocket connections, receives data, validates it,
    processes it using an AI model, and sends the results back.
    """
    async for message in websocket:
        print(f"Received message: {message}")

        # Validate the incoming message
        data = validate_message(message)
        if data is None:
            await websocket.send("Invalid data format. Please send numeric values.")
            continue

        # Preprocess the data
        preprocessed_data = preprocess(data)

        # Predict using the AI model
        prediction = model.predict(preprocessed_data)

        # Send the prediction result back to the client
        result = prediction.tolist()  # Convert the prediction to a list
        await websocket.send(f"Prediction result: {result}")

async def main():
    """
    Starts the WebSocket server.
    """
    async with websockets.serve(handle_connection, "localhost", 8765):
        print("WebSocket server is running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
    
    



import asyncio
import websockets
import numpy as np
import tensorflow as tf
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the pre-trained AI models
models = {
    "regression": tf.keras.models.load_model("models/regression_model.h5"),
    "classification": tf.keras.models.load_model("models/classification_model.h5")
}

# Constants
MAX_MESSAGE_LENGTH = 1024  # Limit the input size
VALID_PATHS = ["/regression", "/classification"]  # Allowed WebSocket paths

def validate_message(message):
    """
    Validates the incoming message. Ensure it contains numeric data between 0 and 1.
    """
    try:
        # Parse the message as a comma-separated list of floats
        data = np.array(message.split(','), dtype=float)
        
        # Check if all values are in the range [0, 1]
        if np.all((data >= 0) & (data <= 1)):
            return data
        else:
            return None
    except ValueError:
        return None

def preprocess(data):
    """
    Preprocesses the incoming data for model prediction.
    Adds a batch dimension to the input array.
    """
    return np.expand_dims(data, axis=0)

async def handle_connection(websocket, path):
    """
    Handles WebSocket connections.
    Validates input, processes it using the AI model, and sends results back.
    """
    if path not in VALID_PATHS:
        logging.warning(f"Invalid path accessed: {path}")
        await websocket.send("Error: Invalid path. Use /regression or /classification.")
        return

    model_type = "regression" if path == "/regression" else "classification"
    model = models[model_type]

    logging.info(f"New connection on path: {path}")

    async for message in websocket:
        logging.info(f"Received message: {message}")

        # Validate input size
        if len(message) > MAX_MESSAGE_LENGTH:
            await websocket.send("Error: Message too large. Limit is 1 KB.")
            continue

        # Validate the incoming message
        data = validate_message(message)
        if data is None:
            await websocket.send("Error: Invalid data. Ensure values are numeric and between 0 and 1.")
            logging.warning("Invalid data received.")
            continue

        # Preprocess the data
        preprocessed_data = preprocess(data)

        # Predict using the AI model
        start_time = time()
        try:
            prediction = model.predict(preprocessed_data)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            await websocket.send("Error: Prediction failed.")
            continue
        elapsed_time = time() - start_time

        # Prepare and send the response
        response = {
            "model": model_type,
            "prediction": prediction.tolist(),
            "processing_time": f"{elapsed_time:.2f} seconds"
        }
        logging.info(f"Prediction sent: {response}")
        await websocket.send(str(response))

async def main():
    """
    Starts the WebSocket server.
    """
    async with websockets.serve(handle_connection, "localhost", 8765):
        logging.info("WebSocket server is running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
'''
import asyncio
import websockets
import numpy as np
import joblib
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the pre-trained scikit-learn models
models = {
    "regression": joblib.load("models/regression_model.pkl"),
    "classification": joblib.load("models/classification_model.pkl")  # Assuming a classification model is available
}

# Constants
MAX_MESSAGE_LENGTH = 1024  # Limit the input size

def validate_message(message):
    """
    Validates the incoming message. Ensure it contains numeric data between 0 and 1.
    """
    try:
        # Parse the message as a comma-separated list of floats
        data = np.array(message.split(','), dtype=float)
        
        # Check if all values are in the range [0, 1]
        if np.all((data >= 0) & (data <= 1)):
            return data
        else:
            return None
    except ValueError:
        return None

def preprocess(data):
    """
    Preprocesses the incoming data for model prediction.
    Reshapes the data to match the model input.
    """
    return data.reshape(1, -1)  # Reshape to match model input (1 sample, n features)

async def handle_connection(websocket):
    """
    Handles WebSocket connections.
    Validates input, processes it using the AI model, and sends results back.
    """

    async for message in websocket:
        logging.info(f"Received message: {message}")

        # Validate input size
        if len(message) > MAX_MESSAGE_LENGTH:
            await websocket.send("Error: Message too large. Limit is 1 KB.")
            continue

        # Validate the incoming message
        data = validate_message(message)
        if data is None:
            await websocket.send("Error: Invalid data. Ensure values are numeric and between 0 and 1.")
            logging.warning("Invalid data received.")
            continue

        # Preprocess the data
        preprocessed_data = preprocess(data)

        # Predict using the appropriate model (you can select either 'regression' or 'classification' model here)
        # Change this based on your requirement
        model_type = "regression"  # or "classification"
        model = models[model_type]

        # Predict using the scikit-learn model
        start_time = time()
        try:
            prediction = model.predict(preprocessed_data)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            await websocket.send("Error: Prediction failed.")
            continue
        elapsed_time = time() - start_time

        # Prepare and send the response
        response = {
            "prediction": prediction.tolist(),
            "processing_time": f"{elapsed_time:.2f} seconds"
        }
        logging.info(f"Prediction sent: {response}")
        await websocket.send(str(response))

async def main():
    """
    Starts the WebSocket server.
    """
    async with websockets.serve(handle_connection, "localhost", 8765):
        logging.info("WebSocket server is running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
