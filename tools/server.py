#!/usr/bin/env python
"""
DriveStudio server launcher.

Starts the WebSocket server for the DriveStudio viewer.
"""

import argparse
import asyncio
import base64
import concurrent.futures
import json
import logging
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
import websockets
from omegaconf import OmegaConf

from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from models.video_utils import get_numpy
from utils.misc import import_str
from utils.visualization import to8b

# Configure logging before importing other modules
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DriveStudioServer:
    def __init__(
        self,
        checkpoint_path: Path,
        host: str = "0.0.0.0",
        port: int = 8766,
    ):
        self.host = host
        self.port = port
        self.checkpoint_path = checkpoint_path

        self.setup_model(checkpoint_path)

        # Setup GPU rendering task queue for async processing
        self.task_queue = Queue()
        self.active_connections = set()
        self.queue_lock = threading.Lock()

        # Start worker thread for GPU operations
        self.worker_thread = threading.Thread(target=self._gpu_worker, daemon=True)
        self.worker_thread.start()
        logger.info("GPU worker thread started")

    def _gpu_worker(self):
        """
        Worker thread that processes GPU rendering tasks sequentially.
        This ensures only one GPU operation runs at a time.
        """
        while True:
            try:
                task_id, params, future = self.task_queue.get(timeout=0.1)
                try:
                    H, W, time_val, K, c2w = params
                    image = self.get_image(H, W, time_val, K, c2w)
                    future.set_result(("success", image))
                except Exception as e:
                    logger.error(f"Error rendering task {task_id}: {e}")
                    future.set_result(("error", str(e)))
                finally:
                    self.task_queue.task_done()
            except Empty:
                # Timeout waiting for task, continue
                continue

    def setup_model(self, checkpoint_path: Path):
        cfg = OmegaConf.load(checkpoint_path.parent / "config.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build dataset
        self.dataset = DrivingDataset(data_cfg=cfg.data)

        # setup trainer
        self.trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=self.dataset.num_img_timesteps,
            model_config=cfg.model,
            num_train_images=len(self.dataset.train_image_set),
            num_full_images=len(self.dataset.full_image_set),
            test_set_indices=self.dataset.test_timesteps,
            scene_aabb=self.dataset.get_aabb().reshape(2, 3),
            device=self.device,
        )
        self.trainer.resume_from_checkpoint(ckpt_path=checkpoint_path, load_only_model=True)
        self.trainer.set_eval()

        logger.info(f"Loading model from {checkpoint_path}, step {self.trainer.step}")

    def get_image(
        self,
        H: int,
        W: int,
        time: float,
        K: np.ndarray,  # [3, 3]
        c2w: np.ndarray,  # [4, 4], OpenCV
    ):
        with torch.no_grad():
            K = torch.tensor(K, device=self.device)
            c2w = torch.tensor(c2w, device=self.device)

            x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            x, y = x.to(self.device), y.to(self.device)

            origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, K)
            origins = origins.reshape(H, W, 3)
            viewdirs = viewdirs.reshape(H, W, 3)
            direction_norm = direction_norm.reshape(H, W, 1)

            image_infos = {
                "origins": origins,
                "viewdirs": viewdirs,
                "direction_norm": direction_norm,
                "img_idx": torch.full((H, W), 0, device=self.device),  # Always use first image
                "normed_time": torch.full((H, W), time, device=self.device),
                "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1),  # [H, W, 2]
            }

            cam_infos = {
                "camera_to_world": c2w,
                "intrinsics": K,
                "height": torch.tensor(H, dtype=torch.long, device=self.device),
                "width": torch.tensor(W, dtype=torch.long, device=self.device),
            }
            results = self.trainer(image_infos, cam_infos, novel_view=True)

            rgb = get_numpy(results["rgb"])  # [H, W, 3], float [0, 1]
            # opacity = get_numpy(results["opacity"])  # [H, W], float [0, 1]
            # depth = get_numpy(results["depth"])  # [H, W]
            # rgba = np.concatenate([rgb, opacity[..., None]], axis=-1)
            return to8b(rgb)  # [H, W, 3], uint8

    async def handle_client(self, websocket):
        """
        Handle a WebSocket client connection.
        Receives rendering requests and sends back rendered images.
        """
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")

        with self.queue_lock:
            self.active_connections.add(client_id)

        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    await self._process_render_request(websocket, client_id, request)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error processing request from client {client_id}: {e}")
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            with self.queue_lock:
                self.active_connections.discard(client_id)

    async def _process_render_request(self, websocket, client_id, request):
        """
        Process a render request from a client.
        Queues the rendering task and sends the result back.
        """
        task_id = f"{client_id}_{time.time()}"

        try:
            # Extract rendering parameters
            H = request.get("H", 480)
            W = request.get("W", 640)
            time_val = float(request.get("time", 0.0))
            K = np.array(request.get("K", [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]), dtype=np.float32)
            c2w = np.array(request.get("c2w", np.eye(4).tolist()), dtype=np.float32)

            # Validate parameters
            if H <= 0 or W <= 0:
                await websocket.send(json.dumps({"status": "error", "message": "Invalid image dimensions"}))
                return

            # Check queue size to prevent memory overload
            queue_size = self.task_queue.qsize()
            if queue_size > 10:
                logger.warning(f"Task queue overloaded: {queue_size} tasks waiting. Request rejected.")
                await websocket.send(
                    json.dumps({"status": "error", "message": f"Server is overloaded. Queue size: {queue_size}"})
                )
                return

            logger.info(f"Task {task_id} from client {client_id}: H={H}, W={W}, time={time_val}")

            # Create a future for this task
            future = concurrent.futures.Future()

            # Queue the rendering task
            params = (H, W, time_val, K, c2w)
            self.task_queue.put((task_id, params, future))

            logger.info(f"Task {task_id} queued. Queue size: {self.task_queue.qsize()}")

            # Wait for result asynchronously
            loop = asyncio.get_event_loop()
            try:
                status, result = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=300,  # 5 minutes timeout
                )

                if status == "success":
                    # Convert image to base64 asynchronously to avoid blocking
                    image_bytes = result.tobytes()
                    start_time = time.time()
                    image_b64 = await loop.run_in_executor(None, lambda: base64.b64encode(image_bytes).decode("utf-8"))
                    encode_time = time.time() - start_time
                    logger.info(
                        f"Task {task_id} encoding took {encode_time:.2f}s, size: {len(image_b64) / (1024 * 1024):.2f}MB"
                    )

                    response = {
                        "status": "success",
                        "task_id": task_id,
                        "image": image_b64,
                        "shape": list(result.shape),
                        "dtype": str(result.dtype),
                    }
                    logger.info(f"Task {task_id} completed successfully")
                else:
                    response = {
                        "status": "error",
                        "task_id": task_id,
                        "message": result,
                    }
                    logger.error(f"Task {task_id} failed: {result}")

                await websocket.send(json.dumps(response))
            except asyncio.TimeoutError:
                logger.error(f"Task {task_id} timeout")
                await websocket.send(
                    json.dumps({"status": "error", "task_id": task_id, "message": "Rendering timeout"})
                )
        except Exception as e:
            logger.error(f"Error in _process_render_request: {e}")
            await websocket.send(json.dumps({"status": "error", "message": str(e)}))

    async def start_server(self):
        """
        Start the WebSocket server.
        """
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        # Increase max message size to 100MB for large rendered images
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=100 * 1024 * 1024,  # 100MB limit
            max_queue=32,  # Limit concurrent messages
        ):
            logger.info(f"WebSocket server is running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def run(self):
        """
        Run the WebSocket server in an event loop.
        """
        asyncio.run(self.start_server())


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the DriveStudio server",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="path to checkpoint to resume from",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Server port (default: 8766)",
    )
    return parser.parse_args()


def main():
    """Start the DriveStudio server."""
    args = parse_args()
    logger.info("Initializing DriveStudio server...")

    logger.info("Creating server...")
    server = DriveStudioServer(
        checkpoint_path=args.ckpt,
        host=args.host,
        port=args.port,
    )

    logger.info("Starting server...")
    server.run()


if __name__ == "__main__":
    main()
