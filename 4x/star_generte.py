import random
import math
from collections import namedtuple
from enum import Enum


# Star system components
StarSystem = namedtuple('StarSystem', ['name', 'coordinates', 'planets', 'asteroids', 'nebulae'])
Planet = namedtuple('Planet', ['name', 'radius', 'habitability', 'resources'])
Asteroid = namedtuple('Asteroid', ['radius', 'resources'])
Nebula = namedtuple('Nebula', ['radius', 'density'])

# Utility functions
def distance(p1, p2):
    """Calculate the distance between two points in 3D space."""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def random_coordinates(min_val, max_val):
    """Generate random 3D coordinates within a given range."""
    return (random.uniform(min_val, max_val),
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val))

# Star system generation
def generate_star_system(name, min_planets=1, max_planets=5, min_asteroids=0, max_asteroids=10, nebula_chance=0.2):
    """Generate a star system with random characteristics."""
    coordinates = random_coordinates(-1000, 1000)
    num_planets = random.randint(min_planets, max_planets)
    planets = [generate_planet(f"{name} {i+1}") for i in range(num_planets)]
    num_asteroids = random.randint(min_asteroids, max_asteroids)
    asteroids = [generate_asteroid() for _ in range(num_asteroids)]
    nebula = generate_nebula() if random.random() < nebula_chance else None
    return StarSystem(name, coordinates, planets, asteroids, nebula)

def generate_planet(name, min_radius=50, max_radius=200):
    """Generate a planet with random characteristics."""
    radius = random.randint(min_radius, max_radius)
    habitability = random.random()
    resources = random.random()
    return Planet(name, radius, habitability, resources)

def generate_asteroid(min_radius=5, max_radius=20):
    """Generate an asteroid with random characteristics."""
    radius = random.randint(min_radius, max_radius)
    resources = random.random()
    return Asteroid(radius, resources)

def generate_nebula(min_radius=100, max_radius=500):
    """Generate a nebula with random characteristics."""
    radius = random.randint(min_radius, max_radius)
    density = random.random()
    return Nebula(radius, density)

# Star system analysis
def analyze_star_system(star_system, priorities):
    """Analyze a star system based on user-defined priorities."""
    priority_values = []
    for priority, weight in priorities.items():
        if priority == 'habitability':
            value = sum(planet.habitability for planet in star_system.planets) / len(star_system.planets)
        elif priority == 'resources':
            planet_resources = sum(sum(resource.rarity for resource in planet.resources) for planet in star_system.planets) / len(star_system.planets)
            asteroid_resources = sum(asteroid.resources for asteroid in star_system.asteroids)
            if star_system.asteroids:
                value = planet_resources + asteroid_resources / len(star_system.asteroids)
            else:
                value = planet_resources
        elif priority == 'strategic_location':
            # Calculate the average distance to other star systems (not implemented)
            value = 0
        priority_values.append(value * weight)
    return sum(priority_values)

# Exploration target prioritization
def prioritize_exploration_targets(star_systems, priorities):
    """Prioritize a list of star systems based on user-defined criteria."""
    scored_systems = [(analyze_star_system(system, priorities), system) for system in star_systems]
    scored_systems.sort(reverse=True)
    return [system for _, system in scored_systems]

# Planetary types
class PlanetType(Enum):
    TERRESTRIAL = 1
    GAS_GIANT = 2
    ICE_GIANT = 3

# Resource types
ResourceType = namedtuple('ResourceType', ['name', 'rarity'])
MINERALS = ResourceType('Minerals', 0.8)
GASES = ResourceType('Gases', 0.6)
RARE_ELEMENTS = ResourceType('Rare Elements', 0.2)

# Planet characteristics
Planet = namedtuple('Planet', ['name', 'radius', 'mass', 'type', 'habitability', 'resources'])

def generate_planet(name, min_radius=50, max_radius=200):
    """Generate a planet with random characteristics."""
    radius = random.randint(min_radius, max_radius)
    mass = random.uniform(0.1, 5.0)  # Earth masses
    type = random.choice(list(PlanetType))
    habitability = random.random()
    resources = []
    for resource_type in [MINERALS, GASES, RARE_ELEMENTS]:
        if random.random() < resource_type.rarity:
            resources.append(resource_type)
    return Planet(name, radius, mass, type, habitability, resources)

# Exploration and colonization
def explore_system(star_system):
    """Explore a star system and gather information."""
    # Perform exploration activities (e.g., scan planets, survey resources)
    # Update star system information
    pass

def colonize_planet(planet):
    """Colonize a planet and establish a colony."""
    # Check if planet is habitable
    if planet.habitability < 0.5:
        print(f"Planet {planet.name} is not suitable for colonization.")
        return

    # Establish a colony
    print(f"Establishing a colony on {planet.name}.")
    # Perform colonization activities (e.g., build infrastructure, extract resources)

# Example usage
if __name__ == "__main__":
    # Generate a star system
    star_system = generate_star_system("Alpha Centauri", min_planets=3, max_planets=5)

    # Explore the star system
    explore_system(star_system)

    # Colonize a suitable planet
    for planet in star_system.planets:
        if planet.type == PlanetType.TERRESTRIAL:
            colonize_planet(planet)
            break
# Example usage
if __name__ == "__main__":
    # Generate a galaxy with 1000 star systems
    galaxy = [generate_star_system(f"System {i+1}") for i in range(1000)]

    # Define exploration priorities
    priorities = {
        'habitability': 0.5,
        'resources': 0.3,
        'strategic_location': 0.2
    }

    # Prioritize exploration targets
    prioritized_targets = prioritize_exploration_targets(galaxy, priorities)

    # Print the top 10 exploration targets
    print("Top 10 exploration targets:")
    for target in prioritized_targets[:10]:
        print(f"- {target.name} at {target.coordinates}")

# Documentation and usage examples

"""
This library provides functionality for procedurally generating star systems, analyzing their characteristics, and prioritizing exploration targets based on user-defined criteria.

Star System Generation:
- The `generate_star_system` function generates a star system with random characteristics, including planets, asteroids, and (optionally) a nebula.
- The `generate_planet`, `generate_asteroid`, and `generate_nebula` functions generate individual components with random characteristics.

Star System Analysis:
- The `analyze_star_system` function analyzes a star system based on user-defined priorities (habitability, resources, strategic location).
- The priorities and their weights are specified as a dictionary, where the keys are the priority names, and the values are the corresponding weights.

Exploration Target Prioritization:
- The `prioritize_exploration_targets` function takes a list of star systems and a dictionary of priorities, and returns a list of star systems sorted by their overall priority score.

Example usage:

# Generate a galaxy with 1000 star systems
galaxy = [generate_star_system(f"System {i+1}") for i in range(1000)]

# Define exploration priorities
priorities = {
    'habitability': 0.5,
    'resources': 0.3,
    'strategic_location': 0.2
}

# Prioritize exploration targets
prioritized_targets = prioritize_exploration_targets(galaxy, priorities)

# Print the top 10 exploration targets
print("Top 10 exploration targets:")
for target in prioritized_targets[:10]:
    print(f"- {target.name} at {target.coordinates}")
"""