import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d


class Fundamental:

    @staticmethod
    def plot(mp_density, mp_flow, mp_ID):
        # Remove duplicates by averaging flow values for each unique density
        unique_density = np.unique(mp_density)
        mean_flow = np.array([np.mean(mp_flow[mp_density == d]) for d in unique_density])

        # Smooth density range for interpolation
        density_smooth = np.linspace(unique_density.min(), unique_density.max(), 200)

        # Interpolation choice based on number of unique points
        if len(unique_density) < 3:
            # Use linear interpolation if there are fewer than 3 unique points
            interp_function = interp1d(unique_density, mean_flow, kind='linear', fill_value="extrapolate")
            flow_smooth = interp_function(density_smooth)
        else:
            # Use spline interpolation for more unique points
            flow_smooth = make_interp_spline(unique_density, mean_flow)(density_smooth)

        # Plotting the result
        plt.figure(figsize=(8, 6))
        plt.plot(density_smooth, flow_smooth, color='b', linestyle='-', label='Interpolated Curve')
        plt.scatter(unique_density, mean_flow, color='r', marker='o', label='Averaged Data')
        plt.xlabel("Density (pedestrians per measuring area cell)")
        plt.ylabel("Flow (speed * density)")
        plt.title(f"Flow vs. Density Fundamental Diagram of Measuring Point {mp_ID}")
        plt.legend()
        plt.grid(True)
        plt.show()










