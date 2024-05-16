import random
from typing import Tuple, List, Dict, Callable

# Constants for event types
SUPERNOVA = "supernova"
WORMHOLE = "wormhole"
ALIEN_ARTIFACT = "alien_artifact"

# Type aliases
EventType = str
EventHandler = Callable[[Dict], None]
EventImpact = Tuple[float, float]  # (resource_impact, tech_impact)

class GalacticEventManager:
    """
    Manages the generation and handling of random galactic events.
    """

    def __init__(self):
        self.event_handlers: Dict[EventType, EventHandler] = {}
        self.anomaly_investigators: Dict[EventType, Callable[[Dict], EventImpact]] = {}

    def register_event_handler(self, event_type: EventType, handler: EventHandler):
        """
        Register a handler for a specific event type.
        """
        self.event_handlers[event_type] = handler

    def register_anomaly_investigator(self, event_type: EventType, investigator: Callable[[Dict], EventImpact]):
        """
        Register an investigator for a specific anomaly type.
        """
        self.anomaly_investigators[event_type] = investigator

    def generate_event(self, resources: Dict, tech_level: float) -> Tuple[EventType, Dict]:
        """
        Generate a random galactic event based on the current resource availability and technological capabilities.
        """
        event_type = random.choice([SUPERNOVA, WORMHOLE, ALIEN_ARTIFACT])
        event_data = {
            "resources": resources,
            "tech_level": tech_level,
        }
        return event_type, event_data

    def handle_event(self, event_type: EventType, event_data: Dict):
        """
        Handle a galactic event by calling the registered handler.
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type](event_data)
        else:
            print(f"No handler registered for event type: {event_type}")

    def investigate_anomaly(self, event_type: EventType, event_data: Dict) -> EventImpact:
        """
        Investigate an anomaly and determine its potential impacts.
        """
        if event_type in self.anomaly_investigators:
            return self.anomaly_investigators[event_type](event_data)
        else:
            print(f"No investigator registered for event type: {event_type}")
            return 0.0, 0.0

# Example event handlers and anomaly investigators

def handle_supernova(event_data: Dict):
    """
    Handle a supernova event by adjusting resource levels.
    """
    resources = event_data["resources"]
    # Simulate resource depletion due to supernova
    resources["energy"] *= 0.8
    resources["minerals"] *= 0.9

def investigate_wormhole(event_data: Dict) -> EventImpact:
    """
    Investigate a wormhole anomaly and determine its potential impacts.
    """
    tech_level = event_data["tech_level"]
    # Simulate potential for faster travel and resource acquisition
    resource_impact = 1.2 if tech_level >= 0.7 else 1.0
    tech_impact = 1.1 if tech_level >= 0.5 else 1.0
    return resource_impact, tech_impact

def investigate_alien_artifact(event_data: Dict) -> EventImpact:
    """
    Investigate an alien artifact anomaly and determine its potential impacts.
    """
    resources = event_data["resources"]
    tech_level = event_data["tech_level"]
    # Simulate potential for technological breakthrough
    resource_impact = 1.0
    tech_impact = 1.5 if resources["energy"] >= 1000 and tech_level >= 0.8 else 1.0
    return resource_impact, tech_impact

# Documentation and usage examples

# Initialize the GalacticEventManager
event_manager = GalacticEventManager()

# Register event handlers
event_manager.register_event_handler(SUPERNOVA, handle_supernova)

# Register anomaly investigators
event_manager.register_anomaly_investigator(WORMHOLE, investigate_wormhole)
event_manager.register_anomaly_investigator(ALIEN_ARTIFACT, investigate_alien_artifact)

# Example usage
resources = {"energy": 2000, "minerals": 1500}
tech_level = 0.7

# Generate a random event
event_type, event_data = event_manager.generate_event(resources, tech_level)

# Handle the event
event_manager.handle_event(event_type, event_data)

# Investigate an anomaly
if event_type in [WORMHOLE, ALIEN_ARTIFACT]:
    resource_impact, tech_impact = event_manager.investigate_anomaly(event_type, event_data)
    print(f"Anomaly impact: Resource={resource_impact}, Tech={tech_impact}")