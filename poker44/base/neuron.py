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

import copy
import os
import sys
from types import SimpleNamespace
import bittensor as bt
from abc import ABC, abstractmethod

# Sync calls set weights and also resyncs the metagraph.
from poker44.utils.config import check_config, add_args, config
from poker44.utils.misc import ttl_get_block
import time
import traceback
import requests
import re
from poker44 import version_url
from poker44 import __version__, __spec_version__


class _ScalarInt(int):
    def item(self):
        return int(self)


class _MinimalMetagraph:
    """Small miner-only metagraph used when public RPC metagraph APIs trap."""

    is_fallback = True

    def __init__(self, netuid: int, uid: int, hotkey: str, block: int):
        size = max(256, uid + 1)
        self.netuid = int(netuid)
        self.n = size
        self.uids = list(range(size))
        self.hotkeys = [""] * size
        self.hotkeys[uid] = hotkey
        self.validator_permit = [False] * size
        self.S = [0.0] * size
        self.I = [0.0] * size
        self.last_update = [int(block)] * size
        self.block = _ScalarInt(block)

    def sync(self, subtensor=None):
        try:
            block = int(subtensor.get_current_block()) if subtensor is not None else int(time.time())
        except Exception:
            block = int(time.time())
        self.last_update = [block] * len(self.last_update)
        self.block = _ScalarInt(block)

    def __repr__(self):
        return f"MinimalMetagraph(netuid={self.netuid}, n={self.n}, fallback=True)"


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.Subtensor"
    wallet: "bt.Wallet"
    metagraph: "bt.metagraph"
    spec_version: int = __spec_version__

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self._apply_direct_overrides()
        self.check_config(self.config) 
        print(
            "[STARTUP] config "
            f"netuid={getattr(self.config, 'netuid', None)} "
            f"wallet={getattr(self.config.wallet, 'name', None)} "
            f"hotkey={getattr(self.config.wallet, 'hotkey', None)} "
            f"axon={getattr(self.config.axon, 'ip', None)}:{getattr(self.config.axon, 'port', None)} "
            f"external={getattr(self.config.axon, 'external_ip', None)}:{getattr(self.config.axon, 'external_port', None)}",
            flush=True,
        )

        # Version check
        self.parse_versions()

        # Set up logging with the provided configuration.
        bt.logging.set_config(config=self.config.logging)

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.

        self.wallet = self._build_wallet()
        while True:
            try:
                bt.logging.info("Initializing subtensor and metagraph")
                self.subtensor = self._build_subtensor()
                self.metagraph = self._load_metagraph()
                print(
                    "[STARTUP] bittensor ready "
                    f"network={getattr(self.subtensor, 'chain_endpoint', None)} "
                    f"metagraph_n={getattr(self.metagraph, 'n', None)} "
                    f"fallback={getattr(self.metagraph, 'is_fallback', False)}",
                    flush=True,
                )
                break
            except Exception as e:
                bt.logging.error(
                    "Couldn't init subtensor and metagraph with error: {}".format(e)
                )
                bt.logging.error(
                    "If you use public RPC endpoint try to move to local node"
                )
                time.sleep(5)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        print(
            "[STARTUP] registered "
            f"netuid={self.config.netuid} uid={self.uid} "
            f"hotkey={self.wallet.hotkey.ss58_address}",
            flush=True,
        )
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0
        self.last_update = 0

    def _apply_direct_overrides(self):
        """Normalize critical runtime values when dotted CLI args are not merged."""
        self._ensure_namespace("wallet")
        self._ensure_namespace("subtensor")
        self._ensure_namespace("axon")
        self._ensure_namespace("neuron")
        self._ensure_namespace("miner")
        self._ensure_namespace("blacklist")
        if not hasattr(self.config.blacklist, "force_validator_permit"):
            self.config.blacklist.force_validator_permit = True
        if not hasattr(self.config.blacklist, "allow_non_registered"):
            self.config.blacklist.allow_non_registered = False
        if not hasattr(self.config.blacklist, "allowed_validator_hotkeys"):
            self.config.blacklist.allowed_validator_hotkeys = []

        netuid = self._first_value(self._cli_arg("--netuid"), os.getenv("POKER44_NETUID"))
        if netuid not in (None, ""):
            self.config.netuid = int(netuid)
        elif getattr(self.config, "netuid", None) is None:
            self.config.netuid = 126

        uid = self._first_value(self._cli_arg("--neuron.uid"), os.getenv("POKER44_UID"))
        if uid not in (None, ""):
            self.config.neuron.uid = int(uid)

        self._set_text("wallet", "name", self._cli_arg("--wallet.name", "--wallet-name"), os.getenv("BT_WALLET_NAME"))
        self._set_text("wallet", "hotkey", self._cli_arg("--wallet.hotkey", "--wallet-hotkey"), os.getenv("BT_WALLET_HOTKEY"))
        self._set_text("wallet", "path", self._cli_arg("--wallet.path", "--wallet-path"), os.getenv("BT_WALLET_PATH"))
        self._set_text("subtensor", "network", self._cli_arg("--subtensor.network"), os.getenv("BT_SUBTENSOR_NETWORK"))
        self._set_text(
            "subtensor",
            "chain_endpoint",
            self._cli_arg("--subtensor.chain_endpoint"),
            os.getenv("BT_SUBTENSOR_CHAIN_ENDPOINT"),
        )
        self._set_text("miner", "model_path", self._cli_arg("--miner.model_path"), os.getenv("POKER44_MODEL_PATH"))
        self._set_text("axon", "ip", self._cli_arg("--axon.ip", "--axon-ip"), os.getenv("BT_AXON_IP"))
        self._set_text("axon", "external_ip", self._cli_arg("--axon.external_ip", "--axon-external-ip"), os.getenv("BT_AXON_EXTERNAL_IP"))

        axon_port = self._first_value(self._cli_arg("--axon.port", "--axon-port"), os.getenv("BT_AXON_PORT"))
        if axon_port not in (None, ""):
            self.config.axon.port = int(axon_port)
        axon_external_port = self._first_value(
            self._cli_arg("--axon.external_port", "--axon-external-port"),
            os.getenv("BT_AXON_EXTERNAL_PORT"),
        )
        if axon_external_port not in (None, ""):
            self.config.axon.external_port = int(axon_external_port)

        if self._cli_flag("--blacklist.allow_non_registered"):
            self.config.blacklist.allow_non_registered = True
        if self._cli_flag("--blacklist.force_validator_permit"):
            self.config.blacklist.force_validator_permit = True
        if self._cli_flag("--no-blacklist.force_validator_permit"):
            self.config.blacklist.force_validator_permit = False

    def _ensure_namespace(self, name: str):
        if getattr(self.config, name, None) is None:
            setattr(self.config, name, SimpleNamespace())

    def _set_text(self, namespace: str, key: str, *values):
        value = self._first_value(*values)
        if value not in (None, ""):
            setattr(getattr(self.config, namespace), key, str(value).strip())

    @staticmethod
    def _first_value(*values):
        for value in values:
            if value not in (None, ""):
                return value
        return None

    def _build_wallet(self):
        """Create wallet explicitly so dotted CLI args survive config merge quirks."""
        wallet_config = getattr(self.config, "wallet", None)
        name = (
            self._cli_arg("--wallet.name", "--wallet-name")
            or str(getattr(wallet_config, "name", "") or "").strip()
            or os.getenv("BT_WALLET_NAME", "").strip()
        )
        hotkey = (
            self._cli_arg("--wallet.hotkey", "--wallet-hotkey")
            or str(getattr(wallet_config, "hotkey", "") or "").strip()
            or os.getenv("BT_WALLET_HOTKEY", "").strip()
        )
        path = (
            self._cli_arg("--wallet.path", "--wallet-path")
            or str(getattr(wallet_config, "path", "") or "").strip()
            or os.getenv("BT_WALLET_PATH", "").strip()
        )

        if name or hotkey or path:
            if wallet_config is None:
                self.config.wallet = SimpleNamespace()
                wallet_config = self.config.wallet
            if name:
                wallet_config.name = name
            if hotkey:
                wallet_config.hotkey = hotkey
            if path:
                wallet_config.path = path

            kwargs = {}
            if name:
                kwargs["name"] = name
            if hotkey:
                kwargs["hotkey"] = hotkey
            if path:
                kwargs["path"] = path

            try:
                return bt.Wallet(**kwargs)
            except TypeError:
                return bt.Wallet(config=self.config)

        return bt.Wallet(config=self.config)

    def _build_subtensor(self):
        """Create subtensor directly from network/endpoint when config merge is lossy."""
        subtensor_config = getattr(self.config, "subtensor", None)
        chain_endpoint = str(getattr(subtensor_config, "chain_endpoint", "") or "").strip()
        network = str(getattr(subtensor_config, "network", "") or "").strip()

        if network:
            return bt.Subtensor(network=network)
        if chain_endpoint:
            return bt.Subtensor(network=chain_endpoint)
        return bt.Subtensor(config=self.config)

    def _load_metagraph(self, allow_fallback: bool = True):
        """Load metagraph, retrying full mode when the lite runtime API traps."""
        try:
            return self.subtensor.metagraph(self.config.netuid)
        except Exception as exc:
            message = str(exc)
            if "get_neurons_lite" not in message and "wasm" not in message.lower():
                raise
            bt.logging.warning(
                "Lite metagraph load failed; retrying with lite=False. "
                f"Original error: {exc}"
            )
            try:
                return self.subtensor.metagraph(self.config.netuid, lite=False)
            except Exception as full_exc:
                if not allow_fallback:
                    raise
                return self._fallback_metagraph(full_exc)

    def _fallback_metagraph(self, exc: Exception):
        uid = self._configured_uid()
        if uid < 0:
            raise RuntimeError(
                "Metagraph RPC failed and no fallback UID was configured. "
                "Run with --neuron.uid <your_registered_uid> or set POKER44_UID."
            ) from exc

        try:
            block = int(self.subtensor.get_current_block())
        except Exception:
            block = int(time.time())

        bt.logging.warning(
            "Using minimal miner metagraph fallback because RPC metagraph load failed. "
            "Validator permit filtering is unavailable until metagraph sync recovers."
        )
        return _MinimalMetagraph(
            netuid=self.config.netuid,
            uid=uid,
            hotkey=self.wallet.hotkey.ss58_address,
            block=block,
        )

    def _configured_uid(self) -> int:
        neuron_config = getattr(self.config, "neuron", None)
        candidates = [
            getattr(neuron_config, "uid", None),
            getattr(self.config, "neuron.uid", None),
            os.getenv("POKER44_UID"),
            self._cli_arg("--neuron.uid"),
        ]
        for value in candidates:
            if value in (None, ""):
                continue
            try:
                uid = int(value)
            except (TypeError, ValueError):
                continue
            if uid >= 0:
                return uid
        return -1

    @staticmethod
    def _cli_arg(*names: str) -> str:
        argv = sys.argv[1:]
        for index, arg in enumerate(argv):
            for name in names:
                if arg == name and index + 1 < len(argv):
                    return argv[index + 1].strip()
                prefix = f"{name}="
                if arg.startswith(prefix):
                    return arg[len(prefix):].strip()
        return ""

    @staticmethod
    def _cli_flag(*names: str) -> bool:
        return any(arg in names for arg in sys.argv[1:])

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def resync_metagraph(self):
        """
        Abstract method that forces subclasses to implement resync_metagraph.
        This ensures that all subclasses define their own way of resynchronizing
        the metagraph.
        """
        pass

    @abstractmethod
    def set_weights(self):

        pass

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        self.check_registered()

        try:
            if self.should_sync_metagraph():
                self.last_update = self.block
                self.resync_metagraph()

            if self.should_set_weights():
                self.set_weights()

            # Always save state.
            self.save_state()
        except Exception as e:
            bt.logging.error(
                "Coundn't sync metagraph or set weights: {}".format(
                    traceback.format_exc()
                )
            )
            bt.logging.error("If you use public RPC endpoint try to move to local node")
            time.sleep(5)

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.

        """
        if self.neuron_type != "MinerNeuron":
            last_update = self.metagraph.last_update[self.uid]
        else:
            last_update = self.last_update

        return (self.block - last_update) > self.config.neuron.epoch_length

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if self.config.neuron.disable_set_weights:
            return False

        # Define appropriate logic for when set weights.
        return (
            self.block - self.metagraph.last_update[self.uid]
        ) > self.config.neuron.epoch_length and self.neuron_type != "MinerNeuron"  # don't set weights if you're a miner

    def save_state(self):
        bt.logging.trace(
            "save_state() not implemented for this neuron. You can implement this function to save model checkpoints or other useful data."
        )

    def load_state(self):
        bt.logging.trace(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )

    def parse_versions(self):
        self.version = __version__
        remote_check_enabled = (
            os.getenv("POKER44_ENABLE_REMOTE_VERSION_CHECK", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        if not remote_check_enabled:
            bt.logging.info("Remote version check disabled; using local package version.")
            return

        bt.logging.info("Parsing versions...")
        try:
            response = requests.get(version_url, timeout=2.0)
            bt.logging.info(f"Response: {response.status_code}")
            if response.status_code != 200:
                return

            content = response.text
            version_pattern = r"__version__\s*=\s*['\"]([^'\"]+)['\"]"

            try:
                version = re.search(version_pattern, content).group(1)
            except AttributeError as e:
                bt.logging.error(f"While parsing versions got error: {e}")
                return

            self.version = version
        except Exception as e:
            bt.logging.warning(f"Remote version check failed: {e}")
        return
