#!/usr/bin/env python
"""
DriveStudio nerfview client.

Connects to the DriveStudio WebSocket server and visualizes the rendered
scenes using nerfview.
"""

import asyncio
import base64
import json
import logging
import threading
from typing import Optional, Tuple

import nerfview
import numpy as np
import viser
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WebSocketClient:
    """Client for connecting to DriveStudio WebSocket server."""

    def __init__(self, server_url: str = "ws://localhost:8766"):
        self.server_url = server_url
        self.websocket = None
        self.response_queue: Optional[asyncio.Queue] = None
        self.connection_event = threading.Event()
        self.running = False
        self.listen_task = None
        self.reconnect_task = None
        self.should_reconnect = True
        self.reconnect_delay = 2  # seconds

    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            # Increase max message size to 100MB to handle large rendered images
            self.websocket = await websockets.connect(
                self.server_url,
                max_size=100 * 1024 * 1024,  # 100MB limit
                max_queue=32,
            )
            self.response_queue = asyncio.Queue()
            self.running = True
            self.connection_event.set()
            logger.info(f"Connected to server at {self.server_url}")
            # Start listening task in the background, don't await it here
            self.listen_task = asyncio.create_task(self._listen_for_messages())
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.running = False
            self.connection_event.clear()
            raise

    async def _listen_for_messages(self):
        """Listen for messages from the server."""
        try:
            async for message in self.websocket:
                try:
                    response = json.loads(message)
                    await self.response_queue.put(response)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed by server, attempting to reconnect...")
            self.running = False
            self.connection_event.clear()
            if self.should_reconnect:
                # Trigger automatic reconnection
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
        except Exception as e:
            logger.error(f"Error listening for messages: {e}", exc_info=True)
            self.running = False
            self.connection_event.clear()
            if self.should_reconnect:
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Automatically attempt to reconnect to the server."""
        max_retries = 5
        retry_count = 0
        while self.should_reconnect and retry_count < max_retries:
            try:
                await asyncio.sleep(self.reconnect_delay)
                logger.info(f"Reconnecting to server... (attempt {retry_count + 1}/{max_retries})")
                await self.connect()
                retry_count = 0  # Reset on successful connection
            except Exception as e:
                retry_count += 1
                logger.warning(f"Reconnection failed: {e}")
                if retry_count >= max_retries:
                    logger.error("Max reconnection attempts reached. Giving up.")
                    self.should_reconnect = False

    async def send_request(self, request: dict) -> Optional[dict]:
        """
        Send a render request to the server and wait for response.

        Args:
            request: Render request dictionary

        Returns:
            Response from server or None if failed
        """
        if not self.running or self.websocket is None:
            logger.error("Not connected to server")
            return None

        try:
            await self.websocket.send(json.dumps(request))
            # Use async queue get with timeout
            response = await asyncio.wait_for(self.response_queue.get(), timeout=300)
            return response
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response from server")
            return None
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return None

    async def close(self):
        """Close the connection."""
        self.should_reconnect = False
        if self.websocket:
            await self.websocket.close()
        self.running = False


class NerfViewClient:
    """DriveStudio client using nerfview for visualization."""

    def __init__(self, server_url: str = "ws://localhost:8766"):
        self.server_url = server_url
        self.ws_client = None
        self.event_loop = None
        self.thread = None
        self.loop_ready = threading.Event()
        self._start_event_loop()

    def _start_event_loop(self):
        """Start a separate thread for asyncio event loop."""

        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.loop_ready.set()  # Signal that the loop is ready
            self.event_loop.run_forever()

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        # Wait for event loop to be ready
        self.loop_ready.wait(timeout=5.0)
        if self.event_loop is None:
            raise RuntimeError("Failed to initialize event loop")

    def connect(self, timeout: float = 10.0):
        """Connect to the WebSocket server."""
        self.ws_client = WebSocketClient(self.server_url)

        # Schedule connect in the event loop (don't wait for it to complete)
        asyncio.run_coroutine_threadsafe(self.ws_client.connect(), self.event_loop)

        # Wait for connection to be established
        if not self.ws_client.connection_event.wait(timeout=timeout):
            raise RuntimeError("Connection timeout")
        if not self.ws_client.running:
            raise RuntimeError("Failed to establish connection")
        logger.info("Client connected successfully")

    def render(
        self,
        camera_state: nerfview.CameraState,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        """
        Render a frame by sending request to server.

        Args:
            camera_state: Current camera state from nerfview
            resolution: Tuple of (width, height) for rendering resolution

        Returns:
            Rendered image as uint8 numpy array [H, W, 3]
        """
        # Determine resolution
        width, height = resolution

        # Get camera parameters
        c2w = camera_state.c2w
        K = camera_state.get_K([width, height])

        # Prepare request
        request = {
            "H": height,
            "W": width,
            "time": 0.0,  # Default time, can be parameterized
            "K": K.tolist(),
            "c2w": c2w.tolist(),
        }

        logger.info(f"Sending render request: H={height}, W={width}")

        # Send request and wait for response
        try:
            # Check if connection is still active, reconnect if needed
            if not self.ws_client.running:
                logger.warning("Connection lost, reconnecting...")
                self.connect(timeout=10.0)

            future = asyncio.run_coroutine_threadsafe(self.ws_client.send_request(request), self.event_loop)
            response = future.result(timeout=10)

            if response is None:
                logger.error("No response from server")
                return np.zeros((height, width, 3), dtype=np.uint8)

            if response.get("status") == "success":
                # Decode base64 image
                image_b64 = response.get("image")
                image_bytes = base64.b64decode(image_b64)

                # Reshape to proper dimensions
                shape = response.get("shape")
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)

                logger.info(f"Render completed: {shape}")
                return image
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Render failed: {error_msg}")
                return np.zeros((height, width, 3), dtype=np.uint8)

        except Exception as e:
            logger.error(f"Error during render: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)

    def close(self):
        """Close the connection."""
        if self.ws_client and self.ws_client.running:
            asyncio.run_coroutine_threadsafe(self.ws_client.close(), self.event_loop)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DriveStudio nerfview client")
    parser.add_argument(
        "--server",
        default="ws://localhost:8766",
        help="WebSocket server URL (default: ws://localhost:8766)",
    )
    args = parser.parse_args()

    # Initialize client
    logger.info("Initializing DriveStudio nerfview client...")
    client = NerfViewClient(server_url=args.server)

    try:
        # Connect to server
        client.connect(timeout=10.0)

        # Create render function
        def render_fn(
            camera_state: nerfview.CameraState,
            resolution: Tuple[int, int],
        ) -> np.ndarray:
            return client.render(camera_state, resolution)

        # Initialize viser server and nerfview viewer
        logger.info("Initializing viser server and nerfview viewer...")
        server = viser.ViserServer(verbose=False)
        viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode="rendering")

        logger.info("Client is running. Connect to the viewer in your browser.")
        # Keep the server running
        while True:
            import time

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client.close()
        logger.info("Client closed")


if __name__ == "__main__":
    main()
