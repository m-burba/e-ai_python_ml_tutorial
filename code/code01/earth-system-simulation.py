
            
class EarthSystemComponent:
    def __init__(self, name):
        self.name = name

    def simulate(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class Atmosphere(EarthSystemComponent):
    def simulate(self):
        return f"Simulating {self.name}: Temperature, pressure, and wind patterns"

class Ocean(EarthSystemComponent):
    def simulate(self):
        return f"Simulating {self.name}: Currents, salinity, and sea surface temperatures"

class Land(EarthSystemComponent):
    def simulate(self):
        return f"Simulating {self.name}: Soil moisture, vegetation, and surface temperature"

class EarthSystemModel:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run_simulation(self):
        for component in self.components:
            print(component.simulate())

atmosphere = Atmosphere("Global Atmosphere")
ocean = Ocean("Global Ocean")
land = Land("Global Land")

model = EarthSystemModel()
model.add_component(atmosphere)
model.add_component(ocean)
model.add_component(land)

model.run_simulation()