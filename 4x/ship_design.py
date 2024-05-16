import random
from typing import List, Tuple

# Ship components
class ShipComponent:
    def __init__(self, name: str, weight: int, cost: int):
        self.name = name
        self.weight = weight
        self.cost = cost

class Engine(ShipComponent):
    def __init__(self, name: str, weight: int, cost: int, speed: int):
        super().__init__(name, weight, cost)
        self.speed = speed

class Weapon(ShipComponent):
    def __init__(self, name: str, weight: int, cost: int, damage: int, range: int):
        super().__init__(name, weight, cost)
        self.damage = damage
        self.range = range

# Ship design system
class ShipDesign:
    def __init__(self, name: str, components: List[ShipComponent]):
        self.name = name
        self.components = components
        self.weapons = [c for c in components if isinstance(c, Weapon)]
        self.speed = max((c.speed for c in components if isinstance(c, Engine)), default=0)
        self.armor = sum(c.weight for c in components)
        self.cargo_capacity = sum(c.weight for c in components) // 10  # Rough estimate

    def __repr__(self):
        return f"{self.name} (Weapons: {len(self.weapons)}, Armor: {self.armor}, Speed: {self.speed}, Cargo: {self.cargo_capacity})"

# Task assignment
def assign_ships_to_tasks(ships: List[ShipDesign], tasks: List[str]) -> dict:
    assignments = {}
    for task in tasks:
        suitable_ships = []
        if task == "Defense":
            suitable_ships = [ship for ship in ships if ship.weapons]
        elif task == "Exploration":
            suitable_ships = [ship for ship in ships if ship.speed > 5 and ship.cargo_capacity > 100]
        elif task == "Transport":
            suitable_ships = [ship for ship in ships if ship.cargo_capacity > 500]

        if suitable_ships:
            assignments[task] = random.sample(suitable_ships, min(len(suitable_ships), 3))
        else:
            assignments[task] = []

    return assignments

# Fleet management
class Fleet:
    def __init__(self, ships: List[ShipDesign]):
        self.ships = ships

    def move_to(self, destination: Tuple[int, int]):
        # Implement pathfinding and fleet movement logic here
        pass

    def attack(self, target: "Fleet"):
        # Implement fleet combat logic here
        pass

    def defend(self, attacker: "Fleet"):
        # Implement fleet defense logic here
        pass

# Documentation and usage examples
"""
This library provides a ship design system, task assignment functionality, and a fleet management system for a 4X strategy game.

Ship Design System:
- The `ShipComponent` class represents a base component for a ship, with properties like weight and cost.
- Derived classes `Engine` and `Weapon` represent specialized components with additional properties like speed and damage.
- The `ShipDesign` class allows for the creation of unique ship designs by combining different components.

Task Assignment:
- The `assign_ships_to_tasks` function takes a list of ship designs and a list of tasks, and assigns suitable ships to each task based on their capabilities.
- Multiple ships can be assigned to a single task, with a maximum of 3 ships per task.

Fleet Management:
- The `Fleet` class represents a collection of ship designs that can move together, attack other fleets, and defend against attacks.
- The `move_to` method allows you to move the fleet to a specified destination (implementation not provided).
- The `attack` method allows the fleet to engage in combat with another fleet (implementation not provided).
- The `defend` method allows the fleet to defend against an attacking fleet (implementation not provided).

Example usage:

# Create some ship components
engine1 = Engine("Ion Drive", 50, 100, 8)
engine2 = Engine("Fusion Drive", 80, 200, 12)
weapon1 = Weapon("Laser Cannon", 20, 50, 30, 10)
weapon2 = Weapon("Plasma Cannon", 40, 100, 50, 15)

# Create ship designs
combat_design = ShipDesign("Vanguard", [engine1, weapon1, weapon2])
exploration_design = ShipDesign("Pathfinder", [engine2, weapon1])
transport_design = ShipDesign("Cargo Hauler", [engine1] * 4)

# Assign ships to tasks
tasks = ["Defense", "Exploration", "Transport"]
assignments = assign_ships_to_tasks([combat_design, exploration_design, transport_design], tasks)
for task, assigned_ships in assignments.items():
    print(f"{task}: {', '.join(str(ship) for ship in assigned_ships)}")

# Create a fleet
fleet = Fleet([combat_design] * 5 + [exploration_design] * 3 + [transport_design] * 2)

# Move the fleet
fleet.move_to((100, 200))

# Engage in combat with another fleet
enemy_fleet = Fleet([combat_design] * 4)
fleet.attack(enemy_fleet)
enemy_fleet.defend(fleet)
"""