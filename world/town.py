"""Town world state: locations and occupants."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

__all__ = ["Town", "Location"]

logger = logging.getLogger(__name__)

LOCATIONS = ["Home", "Library", "Cafe", "Park", "Office", "Market", "Town Square"]


@dataclass
class Location:
    name: str
    capacity: int = 10
    occupants: list[str] = field(default_factory=list)


class Town:
    def __init__(self) -> None:
        self.locations: dict[str, Location] = {
            name: Location(name=name) for name in LOCATIONS
        }

    def move_agent(self, agent_id: str, from_loc: str, to_loc: str) -> None:
        if from_loc in self.locations and agent_id in self.locations[from_loc].occupants:
            self.locations[from_loc].occupants.remove(agent_id)
        if to_loc in self.locations:
            self.locations[to_loc].occupants.append(agent_id)
            logger.debug(f"[Town] {agent_id} moved {from_loc} -> {to_loc}")

    def agents_at(self, location: str) -> list[str]:
        return list(self.locations.get(location, Location(name=location)).occupants)
