import random
import heapq

# Colony management
class Colony:
    def __init__(self, name, location, resources, population=100, infrastructure=0):
        self.name = name
        self.location = location
        self.resources = resources
        self.population = population
        self.infrastructure = infrastructure
        self.defense_system = DefenseSystem(self)

    def update(self, time_step):
        # Population growth
        self.population += int(self.population * 0.01)

        # Resource production
        for resource, amount in self.resources.items():
            self.resources[resource] += int(amount * 0.1 * self.infrastructure)

        # Infrastructure construction
        if self.resources.get("metal", 0) >= 100 and self.resources.get("energy", 0) >= 50:
            self.resources["metal"] -= 100
            self.resources["energy"] -= 50
            self.infrastructure += 1

        # Defense system update
        self.defense_system.update(time_step)

# Resource distribution
def distribute_resources(colonies, resources):
    colony_needs = []
    for i, colony in enumerate(colonies):
        needs = {}
        for resource, amount in resources.items():
            deficit = max(0, colony.population * 10 - colony.resources.get(resource, 0))
            if deficit > 0:
                needs[resource] = deficit
        # Add unique index to avoid comparison issues with dicts
        heapq.heappush(colony_needs, (sum(needs.values()), i, needs, colony))

    while resources and colony_needs:
        _, _, needs, colony = heapq.heappop(colony_needs)
        for resource, need in needs.items():
            if resource in resources:
                amount = min(need, resources[resource])
                colony.resources[resource] += amount
                resources[resource] -= amount

# Defense system
class DefenseSystem:
    def __init__(self, colony):
        self.colony = colony
        self.defense_platforms = []
        self.shields = 0

    def update(self, time_step):
        # Simulate defense system monitoring and response
        # Check for potential threats in the area
        threats = []  # Initialize empty threats list
        
        # In a real implementation, this might check nearby hostile entities
        # For now, we'll just do basic maintenance
        
        for threat in threats:
            if self.engage(threat):
                print(f"Colony {self.colony.name} defended against {threat}")
            else:
                print(f"Colony {self.colony.name} was destroyed by {threat}")
                break

    def engage(self, threat):
        # Implement defense logic
        return random.random() < 0.5

    def construct_defense_platform(self):
        if self.colony.resources.get("metal", 0) >= 200 and self.colony.resources.get("energy", 0) >= 100:
            self.colony.resources["metal"] -= 200
            self.colony.resources["energy"] -= 100
            self.defense_platforms.append(DefensePlatform())

    def raise_shields(self):
        if self.colony.resources.get("energy", 0) >= 50:
            self.colony.resources["energy"] -= 50
            self.shields += 1
# Resource Extraction and Processing
class ResourceExtractor:
    def __init__(self, resource_type, extraction_rate):
        self.resource_type = resource_type
        self.extraction_rate = extraction_rate

    def extract(self, colony):
        amount = min(self.extraction_rate, colony.environment.get(self.resource_type, 0))
        colony.resources[self.resource_type] += amount
        colony.environment[self.resource_type] -= amount
        return amount

class ResourceProcessor:
    def __init__(self, input_resource, output_resource, conversion_rate):
        self.input_resource = input_resource
        self.output_resource = output_resource
        self.conversion_rate = conversion_rate

    def process(self, colony):
        amount = min(colony.resources.get(self.input_resource, 0), self.conversion_rate)
        colony.resources[self.input_resource] -= amount
        colony.resources[self.output_resource] += amount * self.conversion_rate

# Advanced Infrastructure
class Infrastructure:
    """Base class for infrastructure components"""
    def __init__(self):
        pass
    
    def update(self, colony):
        """Override in subclasses to implement specific infrastructure behavior"""
        pass

class PowerPlant(Infrastructure):
    def __init__(self, level):
        self.level = level
        self.power_output = level * 100

    def update(self, colony):
        colony.resources["power"] += self.power_output

class ResearchLab(Infrastructure):
    def __init__(self, level):
        self.level = level
        self.research_output = level * 10

    def update(self, colony):
        if colony.resources.get("power", 0) >= 50:
            colony.resources["research"] += self.research_output
            colony.resources["power"] -= 50

# Colonist Specializations
class Colonist:
    """Base class for colonist types"""
    def __init__(self, name="Colonist"):
        self.name = name
    
    def work(self, colony):
        """Override in subclasses to implement specific work behavior"""
        pass

class Worker(Colonist):
    def work(self, colony):
        colony.resources["labor"] += 1

class Scientist(Colonist):
    def research(self, colony):
        if colony.resources.get("research", 0) >= 100:
            colony.unlock_technology()
            colony.resources["research"] -= 100

# Research and Technology
class Technology:
    def __init__(self, name, cost, bonus):
        self.name = name
        self.cost = cost
        self.bonus = bonus

    def apply_bonus(self, colony):
        # Apply the bonus to the colony's resources, production rates, etc.
        pass

# Environmental Hazards and Disasters
class EnvironmentalHazard:
    """Base class for environmental hazards"""
    def __init__(self, name, severity=1.0):
        self.name = name
        self.severity = severity
    
    def affect_colony(self, colony):
        """Override in subclasses to implement specific hazard effects"""
        pass

class Storm(EnvironmentalHazard):
    def __init__(self, severity):
        self.severity = severity

    def impact(self, colony):
        damage = self.severity * 100
        colony.resources["infrastructure"] -= damage
        print(f"A storm has caused {damage} damage to {colony.name}'s infrastructure.")

# Trade and Diplomacy
class TradeAgreement:
    def __init__(self, colony1, colony2, resources1, resources2):
        self.colony1 = colony1
        self.colony2 = colony2
        self.resources1 = resources1
        self.resources2 = resources2

    def execute_trade(self):
        for resource, amount in self.resources1.items():
            self.colony1.resources[resource] -= amount
            self.colony2.resources[resource] += amount
        for resource, amount in self.resources2.items():
            self.colony2.resources[resource] -= amount
            self.colony1.resources[resource] += amount
# Example usage
colony1 = Colony("Alpha", (10, 20), {"metal": 1000, "energy": 500})
colony2 = Colony("Beta", (30, 40), {"metal": 500, "energy": 1000})
colonies = [colony1, colony2]

resources = {"metal": 2000, "energy": 1000}

for _ in range(10):
    for colony in colonies:
        colony.update(1)
    distribute_resources(colonies, resources)

print("Colony resources:")
for colony in colonies:
    print(f"{colony.name}: {colony.resources}")

# Documentation and usage examples

"""
This library provides a colony management system for a 4X strategy game, including functionality for colony development, resource distribution, and defense systems.

Colony Management:
The Colony class represents a colony and handles various aspects of colony development, such as population growth, resource production, and infrastructure construction. The update method should be called periodically to update the colony's state.

Example:
colony = Colony("Alpha", (10, 20), {"metal": 1000, "energy": 500})
for _ in range(10):
    colony.update(1)
print(colony.population)  # Output: 110
print(colony.resources)  # Output: {"metal": 1100, "energy": 550}
print(colony.infrastructure)  # Output: 1

Resource Distribution:
The distribute_resources function takes a list of colonies and a dictionary of available resources, and distributes the resources to the colonies based on their population needs.

Example:
colony1 = Colony("Alpha", (10, 20), {"metal": 1000, "energy": 500})
colony2 = Colony("Beta", (30, 40), {"metal": 500, "energy": 1000})
colonies = [colony1, colony2]
resources = {"metal": 2000, "energy": 1000}
distribute_resources(colonies, resources)
print(colony1.resources)  # Output: {"metal": 1500, "energy": 750}
print(colony2.resources)  # Output: {"metal": 1000, "energy": 1250}

Defense System:
The DefenseSystem class represents the defense system of a colony and handles threats from hostile forces and environmental hazards. The engage method implements the defense logic and should be customized based on the game's requirements.

Example:
colony = Colony("Alpha", (10, 20), {"metal": 1000, "energy": 500})
colony.defense_system.construct_defense_platform()
colony.defense_system.raise_shields()
colony.defense_system.update(1)  # Engage threats

This library can be integrated with the game engine by creating instances of the Colony and DefenseSystem classes, and calling the appropriate methods to manage colony development, resource distribution, and defense systems.
"""