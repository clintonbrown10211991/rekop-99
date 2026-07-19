# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import asyncio
import threading
import argparse
import traceback

import bittensor as bt
try:
    from bittensor.core.errors import NotVerifiedException
except Exception:  # pragma: no cover - SDK compatibility shim
    class NotVerifiedException(Exception):
        pass

from poker44.base.neuron import BaseNeuron
from poker44.utils.config import add_miner_args
from poker44.validator.synapse import DetectionSynapse

from typing import Union


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self, config=None):
        super(BaseMinerNeuron, self).__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )
        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.Axon(
            wallet=self.wallet,
            config=self.config() if callable(self.config) else self.config,
        )

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info("Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn = self.forward,
            blacklist_fn = self.blacklist,
            priority_fn = self.priority,
        )
        if self.validator_hotkey_whitelist:
            self.axon.verify_fns[DetectionSynapse.__name__] = self.verify_validator_request
        # # self.axon.attach(
        #     forward_fn=self.forward_feedback,
        #     blacklist_fn=self.blacklist_feedback,
        #     priority_fn=self.priority_feedback,
        # )
        # self.axon.attach(
        #     forward_fn=self.forward_set_organic_endpoint,
        #     blacklist_fn=self.blacklist_set_organic_endpoint,
        #     priority_fn=self.priority_set_organic_endpoint,
        # )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    @property
    def validator_hotkey_whitelist(self) -> set[str]:
        configured = getattr(self.config.blacklist, "allowed_validator_hotkeys", []) or []
        return {str(hotkey).strip() for hotkey in configured if str(hotkey).strip()}

    async def verify_validator_request(self, synapse: DetectionSynapse) -> None:
        """Require signed requests from explicitly allowed validator hotkeys."""
        if synapse.dendrite is None:
            raise NotVerifiedException("Missing dendrite terminal in request")

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.validator_hotkey_whitelist:
            raise NotVerifiedException(f"{hotkey} is not a whitelisted validator")

        signature = getattr(synapse.dendrite, "signature", None)
        if not signature:
            raise NotVerifiedException("Request carries no signature header")

        default_verify = getattr(self.axon, "default_verify", None)
        if default_verify is None:
            raise NotVerifiedException("Axon default verification is unavailable")

        await default_verify(synapse)

    def common_blacklist(self, synapse: DetectionSynapse):
        """Shared miner admission policy with optional validator allowlist."""
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            print("[REQUEST] rejected reason=missing_dendrite_or_hotkey", flush=True)
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        hotkey = synapse.dendrite.hotkey
        whitelist = self.validator_hotkey_whitelist

        if whitelist:
            if hotkey in whitelist:
                print(
                    f"[REQUEST] allowed hotkey={hotkey} reason=whitelisted_validator",
                    flush=True,
                )
                bt.logging.trace(f"Allowing whitelisted validator hotkey {hotkey}")
                return False, "Whitelisted validator hotkey"
            print(
                f"[REQUEST] rejected hotkey={hotkey} reason=not_in_validator_allowlist",
                flush=True,
            )
            bt.logging.warning(f"Blacklisting non-whitelisted hotkey {hotkey}")
            return True, "Hotkey not in validator allowlist"

        if hotkey not in self.metagraph.hotkeys:
            if not self.config.blacklist.allow_non_registered:
                print(
                    f"[REQUEST] rejected hotkey={hotkey} reason=unregistered",
                    flush=True,
                )
                bt.logging.trace(f"Blacklisting un-registered hotkey {hotkey}")
                return True, "Unrecognized hotkey"
            print(
                f"[REQUEST] allowed hotkey={hotkey} reason=non_registered_allowed",
                flush=True,
            )
            return False, "Non-registered hotkey allowed"

        uid = self.metagraph.hotkeys.index(hotkey)
        if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
            print(
                f"[REQUEST] rejected hotkey={hotkey} uid={uid} reason=non_validator_hotkey",
                flush=True,
            )
            bt.logging.warning(f"Blacklisting a request from non-validator hotkey {hotkey}")
            return True, "Non-validator hotkey"

        print(
            f"[REQUEST] allowed hotkey={hotkey} uid={uid} reason=recognized_validator",
            flush=True,
        )
        bt.logging.trace(f"Not blacklisting recognized hotkey {hotkey}")
        return False, "Hotkey recognized"

    def caller_priority(self, synapse: DetectionSynapse) -> float:
        try:
            if synapse.dendrite is None or synapse.dendrite.hotkey is None:
                print("[PRIORITY] fallback reason=missing_dendrite_or_hotkey", flush=True)
                bt.logging.warning("Received a request without a dendrite or hotkey.")
                return 0.0

            hotkey = synapse.dendrite.hotkey
            if hotkey not in self.metagraph.hotkeys:
                print(
                    f"[PRIORITY] fallback hotkey={hotkey} reason=unregistered",
                    flush=True,
                )
                return 0.0

            caller_uid = self.metagraph.hotkeys.index(hotkey)
            stake_values = getattr(self.metagraph, "S", None)
            raw_priority = float(stake_values[caller_uid]) if stake_values is not None else 0.0
            validator_permits = getattr(self.metagraph, "validator_permit", None)
            has_validator_permit = (
                bool(validator_permits[caller_uid])
                if validator_permits is not None and caller_uid < len(validator_permits)
                else False
            )
            priority = max(raw_priority, 1_000_000.0) if has_validator_permit else raw_priority
            request_uuid = getattr(getattr(synapse, "dendrite", None), "uuid", "")
            chunk_count = len(getattr(synapse, "chunks", []) or [])
            print(
                f"[PRIORITY] hotkey={hotkey} uid={caller_uid} "
                f"priority={priority} raw_priority={raw_priority} "
                f"validator_permit={has_validator_permit} chunks={chunk_count} "
                f"uuid={request_uuid}",
                flush=True,
            )
            bt.logging.trace(f"Prioritizing {hotkey} with value: {priority}")
            return priority
        except Exception as exc:
            print(f"[PRIORITY] fallback reason=exception error={exc}", flush=True)
            bt.logging.warning(f"Priority calculation failed; using 0.0: {exc}")
            return 0.0

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        print(
            "[STARTUP] serving axon "
            f"netuid={self.config.netuid} "
            f"network={getattr(self.config.subtensor, 'chain_endpoint', None)} "
            f"external={getattr(self.config.axon, 'external_ip', None)}:{getattr(self.config.axon, 'external_port', None)}",
            flush=True,
        )
        try:
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
            print(
                "[STARTUP] axon served on-chain "
                f"uid={self.uid} "
                f"external={getattr(self.config.axon, 'external_ip', None)}:{getattr(self.config.axon, 'external_port', None)}",
                flush=True,
            )
            bt.logging.success(
                f"Miner axon served on-chain | uid={self.uid} "
                f"ip={getattr(self.config.axon, 'external_ip', None) or getattr(self.config.axon, 'ip', None)} "
                f"port={getattr(self.config.axon, 'external_port', None) or getattr(self.config.axon, 'port', None)}"
            )
        except Exception:
            bt.logging.error(f"Failed to serve miner axon:\n{traceback.format_exc()}")
            raise

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()
        print("[STARTUP] axon HTTP server started", flush=True)
        bt.logging.success("Miner axon HTTP server started.")

        # Check/sync after serving so metagraph RPC issues do not block axon publication.
        self.sync()

        bt.logging.info(f"Miner starting at block: {self.block}")
        print(f"[STARTUP] miner loop active block={self.block}", flush=True)

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        if getattr(self.metagraph, "is_fallback", False):
            try:
                self.metagraph = self._load_metagraph(allow_fallback=False)
                self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
                bt.logging.info("Recovered full metagraph from RPC.")
                return
            except Exception as exc:
                bt.logging.warning(f"Full metagraph still unavailable: {exc}")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

    # Overriding the abstract method from BaseNeuron to avoid instantiation error
    def set_weights(self):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)
