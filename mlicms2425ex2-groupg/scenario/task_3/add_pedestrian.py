import json
import numpy as np
import argparse

def read_scenario(file_path):
    """
    Loads a scenario from a JSON file.

    Parameters:
    ----------
    file_path : str
        The file path to the scenario JSON file.
    
    Returns:
    -------
    dict
        A dictionary containing the scenario data.
    """
    with open(file_path, 'r') as f:
        scenario = json.load(f)
    return scenario

def is_pedestrian_id_valid(id, scenario):
    """
    Checks if the input pedestrian ID is valid by verifying it has not been used in the scenario.

    Parameters:
    ----------
    id : int
        The ID to check for validity.
    scenario : dict
        A dictionary containing the scenario data, which includes lists of 
        pedestrians, targets, sources, and obstacles to determine used IDs.

    Returns:
    -------
    bool
        True if the ID is valid (not used), False otherwise. If False, 
        prints a sorted list of already used IDs in the scenario.
    """
    invalid_ids = []

    # Collect IDs from pedestrians, targets, sources, and obstacles
    dynamic_elems = scenario['scenario']['topography']['dynamicElements']
    for e in dynamic_elems:
        if e['type'] == 'PEDESTRIAN':
            invalid_ids.append(e['attributes']['id'])

    targets = scenario['scenario']['topography']['targets']
    for t in targets:
        invalid_ids.append(t['id'])

    sources = scenario['scenario']['topography']['sources']
    for s in sources:
        invalid_ids.append(s['id'])

    obstacles = scenario['scenario']['topography']['obstacles']
    for o in obstacles:
        invalid_ids.append(o['id'])

    if id in invalid_ids:
        print("IDs already in use: ", sorted(invalid_ids))
        raise Exception("Please choose another ID")
    else:
        print(f"{id} is a valid pedestrian ID!")
        return True

def are_target_ids_valid(ids, scenario):
    valid_ids = []

    targets = scenario['scenario']['topography']['targets']
    for t in targets:
        valid_ids.append(t['id'])

    is_valid = all(elem in valid_ids for elem in ids)

    if not is_valid:
        raise Exception("Specified target IDs do not match with existing target IDs")
    else:
        print(f"Specified target IDs are valid!")
        return True
    
def is_coordinate_inside_obstacles(point, obstacles):
    """
    Check if a point is inside any obstacles using the Ray-Casting algorithm

    Parameters:
    ----------
    point : dict
        A dictionary with 'x' and 'y' coordinates of the point to check.
    obstacles : list of dict
        A list of obstacle dictionaries

    Returns:
    -------
    bool
        True if the point is inside any obstacle, False otherwise.
    """

    def point_in_polygon(point, polygon):
        """Determine if a point is inside a polygon using the Ray-Casting algorithm."""
        x = point['x']
        y = point['y']
        inside = False

        n = len(polygon)
        p1x, p1y = polygon[0]['x'], polygon[0]['y']

        for i in range(n + 1):
            p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def point_in_rectangle(point, rectangle):
        """Determine if a point is inside a rectangle by boundary checking."""
        x = point['x']
        y = point['y']
        rect_x = rectangle['x']
        rect_y = rectangle['y']
        rect_width = rectangle['width']
        rect_height = rectangle['height']
        
        return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

    for obstacle in obstacles:
        if obstacle['visible']:
            shape = obstacle['shape']
            if shape['type'] == 'POLYGON':
                if point_in_polygon(point, shape['points']):
                    raise Exception("Pedestrian coordinates are inside obstacles")
            elif shape['type'] == 'RECTANGLE':
                if point_in_rectangle(point, shape):
                    raise Exception("Pedestrian coordinates are inside obstacles")

    return False

def speed_inference(mean, std_dev):
    """
    Generates a random speed value based on a Gaussian distribution.

    Parameters:
    ----------
    mean : float
        The mean value of the speed distribution.
    std_dev : float
        The standard deviation of the speed distribution.

    Returns:
    -------
    float
        A randomly generated speed value following a Gaussian distribution.
    """
    random_speed = np.random.normal(mean, std_dev)
    return random_speed

def add_pedestrian(scenario, id, targetIds, pos_x, pos_y, scenario_name):
    """
    Adds a pedestrian with specified attributes to a given scenario.

    Parameters
    ----------
    scenario : dict
        The scenario data dictionary to which the pedestrian will be added.
    id : int
        Unique identifier for the pedestrian.
    targetIds : list of int
        List of target IDs that the pedestrian will follow.
    pos_x : float
        The x-coordinate of the pedestrian's starting position.
    pos_y : float
        The y-coordinate of the pedestrian's starting position.
    scenario_name : str
        Name to assign to the scenario after adding the pedestrian.

    Returns
    -------
    dict
        Updated scenario data with the newly added pedestrian.
    """
    if is_pedestrian_id_valid(id, scenario) and are_target_ids_valid(targetIds, scenario) and not is_coordinate_inside_obstacles({"x": pos_x, "y": pos_y}, scenario['scenario']['topography']['obstacles']):
        attributesPedestrian = scenario['scenario']['topography']['attributesPedestrian'].copy()
        attributesPedestrian["id"] = id

        speed_mean = attributesPedestrian["speedDistributionMean"]
        speed_std_dev = attributesPedestrian["speedDistributionStandardDeviation"]

        pedestrian = {
            "attributes": attributesPedestrian,
            "source" : None,
            "targetIds" : targetIds,
            "position" : {
                "x" : pos_x,
                "y" : pos_y
            },
            "freeFlowSpeed": speed_inference(speed_mean, speed_std_dev),
            "type" : "PEDESTRIAN"
        }

        scenario["scenario"]["topography"]["dynamicElements"].append(pedestrian)
        scenario["name"] = scenario_name
        print("Successfully add a new pedestrian!")
        return scenario

def save_scenario(file_path, scenario):
    """
    Saves the scenario data to a specified JSON file.

    Parameters
    ----------
    file_path : str
        The file path where the scenario data will be saved.
    scenario : dict
        The scenario data dictionary to save, including the newly added pedestrian.
    """
    with open(file_path, 'w') as f:
        json.dump(scenario, f, indent = 4)

def main():
    """
    Main function that loads a scenario, adds a pedestrian, and saves the updated scenario.
    """
    parser = argparse.ArgumentParser(description='Add a pedestrian to a scenario.')
    parser.add_argument('file_path', type=str, help='Path to the scenario JSON file.')
    parser.add_argument('pedestrian_id', type=int, help='ID for the new pedestrian.')
    parser.add_argument('target_ids', type=str, help='Comma-separated list of target IDs for the pedestrian (can be a single ID).')
    parser.add_argument('pos_x', type=float, help='X-coordinate of the pedestrian.')
    parser.add_argument('pos_y', type=float, help='Y-coordinate of the pedestrian.')
    parser.add_argument('scenario_name', type=str, help='Name of the updated scenario.')
    parser.add_argument('output_path', type=str, help='Path where the updated scenario will be saved.')

    args = parser.parse_args()

    target_ids_list = list(map(int, args.target_ids.split(',')))

    scenario = read_scenario(args.file_path)
    updated_scenario = add_pedestrian(scenario, args.pedestrian_id, target_ids_list, args.pos_x, args.pos_y, args.scenario_name)

    save_scenario(args.output_path, updated_scenario)

# Example call (can also replace python with python3):
#                       python [input_file_path] [pedestrian_ID] [target_IDs] [x_coordinate] [y_coordinate] [scenario_name] [output_file_path] 
# For multiple targets: python add_pedestrian.py "../task_1/corner_scenario6.scenario" 100 "3,4,5" 9.0 9.0 "task3" "./corner_scenario6_updated.scenario"
# For 1 target        : python add_pedestrian.py "../task_1/corner_scenario6.scenario" 100 "3" 9.0 9..0 "task3" "./corner_scenario6_updated.scenario"
if __name__ == "__main__":
    main()