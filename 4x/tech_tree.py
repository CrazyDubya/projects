from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

# Technology Tree Data Structure
class TechnologyTree:
    def __init__(self):
        self.technologies = {}
        self.dependencies = defaultdict(list)
        self.dependents = defaultdict(list)

    def add_technology(self, tech_id: str, cost: int, requirements: List[str]):
        self.technologies[tech_id] = {"cost": cost, "unlocked": False}
        for req in requirements:
            self.dependencies[tech_id].append(req)
            self.dependents[req].append(tech_id)

    def unlock_technology(self, tech_id: str):
        if tech_id not in self.technologies:
            raise ValueError(f"Technology {tech_id} does not exist.")

        for dependency in self.dependencies[tech_id]:
            if not self.technologies[dependency]["unlocked"]:
                raise ValueError(f"Cannot unlock {tech_id} because {dependency} is not unlocked.")

        self.technologies[tech_id]["unlocked"] = True

    def is_unlocked(self, tech_id: str) -> bool:
        return self.technologies[tech_id]["unlocked"]

    def get_cost(self, tech_id: str) -> int:
        return self.technologies[tech_id]["cost"]

    def get_dependencies(self, tech_id: str) -> List[str]:
        return self.dependencies[tech_id]

    def get_dependents(self, tech_id: str) -> List[str]:
        return self.dependents[tech_id]

# Resource Allocation and Optimization
def allocate_resources(tech_tree: TechnologyTree, resources: int, priorities: Dict[str, float]) -> Dict[str, int]:
    allocated = {}
    remaining_resources = resources

    # Sort technologies by priority
    sorted_techs = sorted(priorities.items(), key=lambda x: x[1], reverse=True)

    for tech_id, priority in sorted_techs:
        if tech_id not in tech_tree.technologies:
            continue

        if tech_tree.is_unlocked(tech_id):
            continue

        dependencies = tech_tree.get_dependencies(tech_id)
        all_dependencies_unlocked = all(tech_tree.is_unlocked(dep) for dep in dependencies)

        if all_dependencies_unlocked:
            cost = tech_tree.get_cost(tech_id)
            if cost <= remaining_resources:
                allocated[tech_id] = cost
                remaining_resources -= cost

    return allocated

def optimize_research(tech_tree: TechnologyTree, resources: int, strategic_goals: Dict[str, float]) -> Dict[str, int]:
    priorities = calculate_priorities(tech_tree, strategic_goals)
    return allocate_resources(tech_tree, resources, priorities)

def calculate_priorities(tech_tree: TechnologyTree, strategic_goals: Dict[str, float]) -> Dict[str, float]:
    priorities = {}
    visited = set()
    queue = deque(strategic_goals.keys())

    while queue:
        tech_id = queue.popleft()
        if tech_id in visited:
            continue

        visited.add(tech_id)
        if tech_id in tech_tree.technologies:
            priorities[tech_id] = strategic_goals.get(tech_id, 0.0)

        for dependent in tech_tree.get_dependents(tech_id):
            queue.append(dependent)

    return priorities

# Technology Effects
def apply_technology_effects(tech_id: str, game_state: dict):
    # Apply the effects of the unlocked technology to the game state
    # This function would need to be implemented based on the specific game mechanics
    pass

# Documentation and usage examples
"""
This library provides a technology tree system for managing research and development in a 4X strategy game. It includes a data structure for representing the technology tree, algorithms for optimizing research priorities and resource allocation, and a system for applying the effects of unlocked technologies.

The TechnologyTree class is used to define the relationships and dependencies between different technologies, as well as their associated costs and requirements. You can add new technologies to the tree using the add_technology method, and unlock technologies using the unlock_technology method.

The allocate_resources function takes a TechnologyTree object, the available resources, and a dictionary of technology priorities, and returns a dictionary of technologies to be researched and the resources allocated to each one. The optimize_research function combines this with the calculate_priorities function to determine the optimal research priorities based on strategic goals and technological dependencies.

The apply_technology_effects function is a placeholder for applying the effects of unlocked technologies to the game state. This function would need to be implemented based on the specific game mechanics and technology effects.

Example usage:"""

# Define the technology tree
tech_tree = TechnologyTree()
tech_tree.add_technology("laser_weapons", 100, ["energy_weapons"])
tech_tree.add_technology("plasma_weapons", 200, ["laser_weapons"])
tech_tree.add_technology("energy_weapons", 50, [])
tech_tree.add_technology("warp_drive", 150, ["advanced_propulsion"])
tech_tree.add_technology("advanced_propulsion", 100, ["energy_weapons"])

# Unlock some technologies
tech_tree.unlock_technology("energy_weapons")
tech_tree.unlock_technology("advanced_propulsion")

# Define strategic goals
strategic_goals = {
    "plasma_weapons": 0.8,
    "warp_drive": 0.6,
}

# Optimize research and allocate resources
resources = 300
allocated_resources = optimize_research(tech_tree, resources, strategic_goals)

# Apply technology effects
for tech_id, cost in allocated_resources.items():
    tech_tree.unlock_technology(tech_id)
    apply_technology_effects(tech_id, game_state)
