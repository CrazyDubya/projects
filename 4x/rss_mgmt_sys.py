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

class ProcessingPlant:
    def __init__(self, inputs, outputs, conversion_rate):
        self.inputs = inputs
        self.outputs = outputs
        self.conversion_rate = conversion_rate

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
        for node in self.nodes:
            extracted = node.extract_resources()
            for resource, amount in extracted.items():
                # Distribute extracted resources to colonies and fleets based on demand and transportation costs
                pass

    def optimize_processing(self):
        for node in self.nodes:
            inputs = node.extract_resources()
            processed = node.process_resources(inputs)
            for resource, amount in processed.items():
                # Distribute processed resources to colonies and fleets based on demand and transportation costs
                pass

    def find_optimal_locations(self, facility_type, resource_types):
        locations = []
        for node in self.nodes:
            node_resources = [resource for resource in node.resources if resource.name in resource_types]
            if node_resources:
                score = sum(resource.units for resource in node_resources)
                locations.append((score, node.location))
        locations.sort(reverse=True)
        return [location for score, location in locations[:facility_type.max_facilities]]

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
manager.optimize_extraction()
manager.optimize_processing()

# Finding optimal locations for new facilities
optimal_locations = manager.find_optimal_locations(mining_facility, ["minerals"])
print("Optimal locations for mining facilities:", optimal_locations)