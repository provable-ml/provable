"""Folded acquisition engine (the AC-series build of record).

The population-and-plug-in machinery behind the purchase layer's columns
leg: SyntheticHost (synthetic worlds only), SieveState (the fitted state,
renamed from the fold's ledger.py), Localizer, Acquisition (the pricing
laws), Envelope (water-fill walls and the terminal remainder), LabelChannel,
Currency, AdmissibilityGate, and the AC8a.4 OrderContract renderer. Internal:
addressed through provable.purchase, not imported directly by callers.
"""
from .host import SyntheticHost
from .state import SieveState
from .localize import Localizer
from .acquire import Acquisition
from .label import LabelChannel
from .currency import Currency
from .envelope import Envelope
from .gate import AdmissibilityGate
from .audit import SelfAudit
from .contract import PurchaseOrder as OrderContract, SEAM_ENUM, BRIDGE_LINE

__all__ = ["SyntheticHost", "SieveState", "Localizer", "Acquisition", "LabelChannel", "Currency", "Envelope", "AdmissibilityGate", "SelfAudit", "OrderContract", "SEAM_ENUM", "BRIDGE_LINE"]
