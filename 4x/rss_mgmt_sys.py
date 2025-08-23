import heapq

class Resource:
    def __init__(self, name, units):
        self.name = name
        self.units = units

    def __repr__(self):
        return f"{self.name}: {self.units}"

class ResourceNode:
    def __init__(self, location, resources):
        self.location = location
        self.resources = resources
        self.facilities = []

    def add_facility(self, facility):
        self.facilities.append(facility)

    def extract_resources(self):
        extracted = {}
        for resource in self.resources:
            amount = sum(facility.extract(resource) for facility in self.facilities)
            if amount > 0:
                extracted[resource.name] = amount
        return extracted

    def process_resources(self, inputs):
        processed = {}
        for facility in self.facilities:
            output = facility.process(inputs)
            for resource, amount in output.items():
                processed[resource] = processed.get(resource, 0) + amount
        return processed

class MiningFacility:
    def __init__(self, resource, extraction_rate):
        self.resource = resource
        self.extraction_rate = extraction_rate

    def extract(self, resource):
        if resource.name == self.resource.name:
            return self.extraction_rate
        return 0

    def process(self, inputs):
        # Mining facilities don't process resources, they extract them
        return {}

class ProcessingPlant:
    def __init__(self, inputs, outputs, conversion_rate):
        self.inputs = inputs
        self.outputs = outputs
        self.conversion_rate = conversion_rate

    def extract(self, resource):
        # Processing plants don't extract raw resources
        return 0

    def process(self, inputs):
        outputs = {}
        for resource, amount in inputs.items():
            if resource in self.inputs:
                output_amount = amount * self.conversion_rate
                output_resource = self.outputs[self.inputs.index(resource)]
                outputs[output_resource.name] = output_amount
        return outputs

class ResourceManager:
    def __init__(self):
        self.nodes = []
        self.colonies = []
        self.fleets = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_colony(self, colony):
        self.colonies.append(colony)

    def add_fleet(self, fleet):
        self.fleets.append(fleet)

    def construct_facility(self, node, facility):
        node.add_facility(facility)

    def optimize_extraction(self):
        # Optimize resource extraction across all nodes
        total_extracted = {}
        
        for node in self.nodes:
            extracted = node.extract_resources()
            
            for resource_name, amount in extracted.items():
                # Distribute extracted resources to colonies and fleets based on demand and distance
                
                # Calculate demand from colonies
                colony_demands = []
                for colony in self.colonies:
                    # Simple demand calculation based on population and current resources
                    current_amount = colony.resources.get(resource_name, 0)
                    population_need = colony.population * 2  # Basic need per person
                    demand = max(0, population_need - current_amount)
                    
                    if demand > 0:
                        # Calculate distance-based priority (closer = higher priority)
                        distance = ((node.location[0] - colony.location[0])**2 + 
                                   (node.location[1] - colony.location[1])**2)**0.5
                        priority = demand / max(distance, 1.0)  # Avoid division by zero
                        colony_demands.append((priority, colony, min(demand, amount)))
                
                # Sort by priority and allocate resources
                colony_demands.sort(reverse=True, key=lambda x: x[0])
                remaining_amount = amount
                
                for priority, colony, needed in colony_demands:
                    if remaining_amount <= 0:
                        break
                    
                    allocated = min(needed, remaining_amount)
                    colony.resources[resource_name] = colony.resources.get(resource_name, 0) + allocated
                    remaining_amount -= allocated
                
                # Track total extraction
                total_extracted[resource_name] = total_extracted.get(resource_name, 0) + amount
        
        return total_extracted

    def optimize_processing(self):
        # Optimize resource processing across all nodes
        total_processed = {}
        
        for node in self.nodes:
            # Get available raw materials from extraction
            raw_materials = node.extract_resources()
            
            if raw_materials:
                # Process the raw materials
                processed = node.process_resources(raw_materials)
                
                for resource_name, amount in processed.items():
                    # Distribute processed resources similar to extraction optimization
                    
                    # Calculate strategic value of processed resources
                    strategic_demands = []
                    for colony in self.colonies:
                        # Processed resources are often more valuable for infrastructure
                        infrastructure_need = colony.infrastructure * 10  # Infrastructure consumption
                        current_amount = colony.resources.get(resource_name, 0)
                        demand = max(0, infrastructure_need - current_amount)
                        
                        if demand > 0:
                            distance = ((node.location[0] - colony.location[0])**2 + 
                                       (node.location[1] - colony.location[1])**2)**0.5
                            priority = demand / max(distance, 1.0)
                            strategic_demands.append((priority, colony, min(demand, amount)))
                    
                    # Allocate processed resources
                    strategic_demands.sort(reverse=True, key=lambda x: x[0])
                    remaining_amount = amount
                    
                    for priority, colony, needed in strategic_demands:
                        if remaining_amount <= 0:
                            break
                        
                        allocated = min(needed, remaining_amount)
                        colony.resources[resource_name] = colony.resources.get(resource_name, 0) + allocated
                        remaining_amount -= allocated
                    
                    # Track total processing
                    total_processed[resource_name] = total_processed.get(resource_name, 0) + amount
        
        return total_processed

    def find_optimal_locations(self, facility_type, resource_types, max_facilities=5):
        # Find optimal locations for placing facilities based on resource availability
        locations = []
        for node in self.nodes:
            node_resources = [resource for resource in node.resources if resource.name in resource_types]
            if node_resources:
                score = sum(resource.units for resource in node_resources)
                locations.append((score, node.location))
        
        locations.sort(reverse=True)
        # Return top locations up to max_facilities limit
        return [location for score, location in locations[:max_facilities]]

# Documentation and usage examples

# Creating resources
minerals = Resource("minerals", 1000)
gas = Resource("gas", 500)
energy = Resource("energy", 200)

# Creating resource nodes
node1 = ResourceNode((10, 20), [minerals, gas])
node2 = ResourceNode((30, 40), [gas, energy])

# Creating facilities
mining_facility = MiningFacility(minerals, 100)
processing_plant = ProcessingPlant([minerals, gas], [energy], 0.5)

# Adding facilities to nodes
node1.add_facility(mining_facility)
node2.add_facility(processing_plant)

# Creating a resource manager
manager = ResourceManager()
manager.add_node(node1)
manager.add_node(node2)

# Extracting and processing resources
extracted = node1.extract_resources()
print("Extracted resources:", extracted)

# Create Resource objects from resource names
resource_objects = {name: Resource(name, units) for name, units in extracted.items()}

# Use Resource objects as keys in the inputs dictionary
inputs = {resource_objects[name]: amount for name, amount in extracted.items()}
processed = node2.process_resources(inputs)
print("Processed resources:", processed)

# Optimizing resource extraction and processing
extracted_totals = manager.optimize_extraction()
processed_totals = manager.optimize_processing()

# Finding optimal locations for new facilities
optimal_locations = manager.find_optimal_locations(mining_facility, ["minerals"])
print("Optimal locations for mining facilities:", optimal_locations)