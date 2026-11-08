"""Processing stages that consume canonical events. Inspection and (opt-in) schema discovery exist so far."""

from agent_traces.processors.discovery import DiscoveryProcessor, render_discovery
from agent_traces.processors.inspection import InspectionProcessor, InspectionReport, render_text

__all__ = ["DiscoveryProcessor", "InspectionProcessor", "InspectionReport", "render_discovery", "render_text"]
