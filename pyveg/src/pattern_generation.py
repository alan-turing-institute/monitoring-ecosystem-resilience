"""
Translation of Matlab code to model patterned vegetation in semi-arid landscapes.
"""
import json
import os
import random

import numpy as np
from matplotlib import pyplot as plt


class PatternGenerator(object):
    """
    Class that can generate simulated veget  ation patterns, optionally
    from a loaded starting pattern, and propagate through time according
    to various amounts of rainfall and/or surface and soil water density.
    """

    def __init__(self):
        default_config_file = os.path.join(
            os.path.dirname(__file__), "..", "testdata", "patternGenConfig.json"
        )
        self.load_config(default_config_file)
        self.plant_biomass = None
        # remember the starting pattern in case we want to compare
        self.starting_pattern = None
        self.rainfall = None  # needs to be set explicitly
        self.configure()
        self.initialize()
        self.time = 0
        pass

    def print_config(self):

        params = [
            "m",
            "delta_x",
            "delta_y",
            "diffusion_plant",
            "diffusion_soil",
            "diffusion_surface",
            "surface_water_frac",
            "bare_soil_infiltration",
            "grazing_loss",
            "soil_water_loss",
            "plant_uptake",
            "plant_growth",
            "plant_senescence",
            "plant_uptake_saturation",
            "water_infilt_saturation",
            "veg_mass_per_cell",
            "fraction_plant_cells",
        ]
        print("\nCurrent configuration:\n======================\n")

        for param in params:
            print("{}: {}".format(param, self.__getattribute__(param)))

    def set_rainfall(self, rainfall):
        """
        Rainfall in mm
        """
        self.rainfall = rainfall

    def set_starting_pattern_from_file(self, filename):
        """
        Takes full path to a CSV file containing m rows
        of m comma-separated values, which are zero (bare soil)
        or not-zero (vegetation covered).
        """
        try:
            pattern = np.genfromtxt(filename, delimiter=",")
        except OSError:
            raise RuntimeError("File {} not found".format(filename))
        # Crop it to m*m if necessary
        pattern = pattern[: self.m, : self.m]

        # rescale non-zero values
        try:
            pattern = pattern * (self.veg_mass_per_cell / pattern.max())
        except ZeroDivisionError:
            print("Pattern is all zeros")
            pass
        self.starting_pattern = pattern
        self.plant_biomass = pattern

    def set_random_starting_pattern(self):
        """
        Use the frac from config file to randomly cover
        some fraction of cells.
        """
        if len(self.config.items()) == 0:
            raise RuntimeError("Need to set config file first")
        # loop over all cells and
        for ix in range(self.m):
            for iy in range(self.m):
                self.surface_water[ix, iy] = self.rainfall / (
                    self.surface_water_frac * self.bare_soil_infiltration
                )
                # Homogeneous equilibrium soil water in absence of plants
                self.soil_water[ix, iy] = self.rainfall / self.soil_water_loss
                if random.random() > self.config["frac"]:
                    self.starting_pattern[ix, iy] = self.config[
                        "vmass"
                    ]  # Initial plant biomass
                else:
                    self.starting_pattern[ix, iy] = 0  # Initial plant biomass
        self.plant_biomass = self.starting_pattern

    def configure(self):
        """
        Set initial parameters, loaded from JSON.
        """

        # number of cells along each of the (x,y) directions
        self.m = self.config["m"]

        # System discretisation
        self.delta_x = self.config["DeltaX"]
        self.delta_y = self.config["DeltaY"]

        # Diffusion constants for plants, soil water, surface water
        self.diffusion_plant = self.config["DifP"]
        self.diffusion_soil = self.config["DifW"]
        self.diffusion_surface = self.config["DifO"]

        # Parameter values
        self.surface_water_frac = self.config[
            "alpha"
        ]  # proportion of surface water available for infiltration (d-1)
        # Bare soil infiltration (-)
        self.bare_soil_infiltration = self.config["W0"]
        # Plant loss rate due to grazing (d-1)
        self.grazing_loss = self.config["beta"]
        self.soil_water_loss = self.config[
            "rw"
        ]  # Soil water loss rate due to seepage and evaporation (d-1)
        # Plant uptake constant (g.mm-1.m-2)
        self.plant_uptake = self.config["c"]
        self.plant_growth = self.config[
            "gmax"
        ]  # Plant growth constant (mm.g-1.m-2.d-1)
        self.plant_senescence = self.config["d"]  # Plant senescence rate (d-1)
        self.plant_uptake_saturation = self.config[
            "k1"
        ]  # Half saturation constant for plant uptake and growth (mm)
        self.water_infilt_saturation = self.config[
            "k2"
        ]  # Half saturation constant for water infiltration (g.m-2)

        # Starting biomass in a vegetation-covered cell.
        self.veg_mass_per_cell = self.config[
            "vmass"
        ]  # how much biomass in vegetation-covered cells?
        self.fraction_plant_cells = self.config[
            "frac"
        ]  # fraction of starting cells with plants

    def initialize(self):
        """
        Set initial values to zero, and boundary conditions.
        """
        # Initialize arrays with zeros
        self.starting_pattern = np.zeros((self.m, self.m))
        self.plant_biomass = np.zeros((self.m, self.m))
        self.soil_water = np.zeros((self.m, self.m))
        self.surface_water = np.zeros((self.m, self.m))
        self.d_plant = np.zeros((self.m, self.m))
        self.d_surf = np.zeros((self.m, self.m))
        self.d_soil = np.zeros((self.m, self.m))
        self.net_flow_plant = np.zeros((self.m, self.m))
        self.net_flow_surf = np.zeros((self.m, self.m))
        self.net_flow_soil = np.zeros((self.m, self.m))

        # Boundary conditions - no flow in/out to x, y directions
        self.y_flow_plant = np.zeros((self.m + 1, self.m))
        self.x_flow_plant = np.zeros((self.m, self.m + 1))
        self.y_flow_soil = np.zeros((self.m + 1, self.m))
        self.x_flow_soil = np.zeros((self.m, self.m + 1))
        self.y_flow_surf = np.zeros((self.m + 1, self.m))
        self.x_flow_surf = np.zeros((self.m, self.m + 1))

    def initial_conditions(self):
        """
        Set initial arrays of soil and surface water.
        """
        if not self.rainfall:
            raise RuntimeError("Need to call set_rainfall() first")
        # Initial conditions for soil and surface water
        for ix in range(self.m):
            for iy in range(self.m):
                self.surface_water[ix, iy] = self.rainfall / (
                    self.surface_water_frac * self.bare_soil_infiltration
                )
                # Homogeneous equilibrium soil water in absence of plants
                self.soil_water[ix, iy] = self.rainfall / self.soil_water_loss

    def evolve_pattern(self, steps=10000, dt=1):
        """
        Run the code to converge on a vegetation pattern
        """
        print("Doing {} steps with rainfall {}mm".format(steps, self.rainfall))

        # assume symmetrix m*m arrays
        nx = self.m
        ny = self.m

        #  Timesteps
        # snapshots = []
        for step in range(steps):

            # Changes over each cell
            d_surf = PatternGenerator.calc_surface_water_change(
                self.surface_water,
                self.plant_biomass,
                self.rainfall,
                self.surface_water_frac,
                self.bare_soil_infiltration,
                self.water_infilt_saturation,
            )

            d_soil = PatternGenerator.calc_soil_water_change(
                self.soil_water,
                self.surface_water,
                self.plant_biomass,
                self.surface_water_frac,
                self.bare_soil_infiltration,
                self.water_infilt_saturation,
                self.plant_growth,
                self.soil_water_loss,
                self.plant_uptake_saturation,
            )

            d_plant = PatternGenerator.calc_plant_change(
                self.plant_biomass,
                self.soil_water,
                self.plant_uptake,
                self.plant_uptake_saturation,
                self.plant_growth,
                self.plant_senescence,
                self.grazing_loss,
            )

            # Diffusion
            # calculate Flow in x - direction: Flow = -D * d_pattern / dx;
            self.x_flow_plant[0:ny, 1:nx] = (
                -1
                * self.diffusion_plant
                * (self.plant_biomass[:, 1:nx] - self.plant_biomass[:, 0 : (nx - 1)])
                * self.delta_y
                / self.delta_x
            )
            self.x_flow_soil[0:ny, 1:nx] = (
                -1
                * self.diffusion_soil
                * (self.soil_water[:, 1:nx] - self.soil_water[:, 0 : (nx - 1)])
                * self.delta_y
                / self.delta_x
            )
            self.x_flow_surf[0:ny, 1:nx] = (
                -1
                * self.diffusion_surface
                * (self.surface_water[:, 1:nx] - self.surface_water[:, 0 : (nx - 1)])
                * self.delta_y
                / self.delta_x
            )

            # calculate Flow in y - direction: Flow = -D * d_pattern / dy;
            self.y_flow_plant[1:ny, 0:nx] = (
                -1
                * self.diffusion_plant
                * (self.plant_biomass[1:ny, :] - self.plant_biomass[0 : (ny - 1), :])
                * self.delta_x
                / self.delta_y
            )
            self.y_flow_soil[1:ny, 0:nx] = (
                -1
                * self.diffusion_soil
                * (self.soil_water[1:ny, :] - self.soil_water[0 : (ny - 1), :])
                * self.delta_x
                / self.delta_y
            )
            self.y_flow_surf[1:ny, 0:nx] = (
                -1
                * self.diffusion_surface
                * (self.surface_water[1:ny, :] - self.surface_water[0 : (ny - 1), :])
                * self.delta_x
                / self.delta_y
            )

            # calculate netflow
            net_plant = (
                self.x_flow_plant[:, 0:nx]
                - self.x_flow_plant[:, 1 : (nx + 1)]
                + self.y_flow_plant[0:ny, :]
                - self.y_flow_plant[1 : ny + 1, :]
            )
            net_soil = (
                self.x_flow_soil[:, 0:nx]
                - self.x_flow_soil[:, 1 : (nx + 1)]
                + self.y_flow_soil[0:ny, :]
                - self.y_flow_soil[1 : ny + 1, :]
            )
            net_surf = (
                self.x_flow_surf[:, 0:nx]
                - self.x_flow_surf[:, 1 : (nx + 1)]
                + self.y_flow_surf[0:ny, :]
                - self.y_flow_surf[1 : ny + 1, :]
            )

            # Update
            self.soil_water = (
                self.soil_water
                + (d_soil + (net_soil / (self.delta_x * self.delta_y))) * dt
            )
            self.surface_water = (
                self.surface_water
                + (d_surf + (net_surf / (self.delta_x * self.delta_y))) * dt
            )
            self.plant_biomass = (
                self.plant_biomass
                + (d_plant + (net_plant / (self.delta_x * self.delta_y))) * dt
            )

            self.time += dt

    # static methods for evolving the simulation
    # --------------------------------------------------------------------------
    @staticmethod
    def calc_plant_change(
        plant_biomass,
        soil_water,
        uptake,
        uptake_saturation,
        growth_constant,
        senescence,
        grazing_loss,
    ):
        """
        Change in plant biomass as a function of available soil water
        and various constants.
        """
        relative_growth = (
            growth_constant
            * uptake
            * np.divide(soil_water, (soil_water + uptake_saturation))
        )
        relative_loss = senescence + grazing_loss

        change = (relative_growth - relative_loss) * plant_biomass
        return change

    @staticmethod
    def calc_surface_water_change(
        surface_water,
        plant_biomass,
        rainfall,
        frac_surface_water_available,
        bare_soil_infilt,
        infilt_saturation,
    ):
        """
        Change in surface water as a function of rainfall, plant_biomass,
        and various constants.
        """
        absorb_numerator = plant_biomass + bare_soil_infilt * infilt_saturation
        absorb_denominator = plant_biomass + infilt_saturation
        rel_loss = frac_surface_water_available * np.divide(
            absorb_numerator, absorb_denominator
        )
        change = rainfall - surface_water * rel_loss
        return change

    @staticmethod
    def calc_soil_water_change(
        soil_water,
        surface_water,
        plant_biomass,
        frac_surface_water_available,
        bare_soil_infilt,
        infilt_saturation,
        plant_growth,
        soil_water_evap,
        uptake_saturation,
    ):
        """
        Change in soil water as a function of surface water, plant_biomass,
        and various constants.
        """
        lost_to_plants = (
            plant_growth
            * np.divide(soil_water, (soil_water + uptake_saturation))
            * plant_biomass
        )

        surface_water_available = surface_water * frac_surface_water_available
        rel_absorbed_from_surface = np.divide(
            (plant_biomass + infilt_saturation * bare_soil_infilt),
            (plant_biomass + infilt_saturation),
        )

        absorbed_from_surface = surface_water_available * rel_absorbed_from_surface
        lost_to_evaporation = soil_water * soil_water_evap
        change = absorbed_from_surface - lost_to_plants - lost_to_evaporation
        return change

    # --------------------------------------------------------------------------

    def make_binary(self, threshold=None):
        """
        if not given a threshold to use,  look at the (max+min)/2 value
        - for anything below, set to zero, for anything above, set to 1
        """
        if not threshold:
            threshold = (
                np.quantile(self.plant_biomass, 0.9)
                + np.quantile(self.plant_biomass, 0.1)
            ) / 2.0
        #            threshold = (self.plant_biomass.max() + self.plant_biomass.min()) / 2.
        new_list_x = []
        for row in self.plant_biomass:
            new_list_y = np.array([255 * int(val < threshold) for val in row])
            new_list_x.append(new_list_y)
        return np.array(new_list_x)

    def load_config(self, config_filename):
        """
        Load a set of configuration parameters from a JSON file
        """
        if not os.path.exists(config_filename):
            raise RuntimeError("Config file {} does not exist".format(config_filename))
        self.config = json.load(open(config_filename))
        self.configure()

    def save_as_csv(self, filename):
        """
        Save the image as a csv file
        """
        print(f'Saving file "{filename}"')
        np.savetxt(filename, self.plant_biomass, delimiter=",", newline="\n", fmt="%f")

    def save_as_matlab(self, filename):
        """
        Save the image as a matlab file
        """
        from scipy.io import savemat

        print(f'Saving file "{filename}"')
        savemat(filename, mdict={"plant_biomass": self.plant_biomass})

    def save_as_png(self, filename):
        """
        Save the image as a png file
        """
        print(f'Saving file "{filename}"')
        im = plt.imshow(self.plant_biomass)
        plt.savefig(filename)

    def plot_image(self):
        """
        Display the current pattern.
        """
        im = plt.imshow(self.plant_biomass)
        plt.colorbar()
        plt.show()
