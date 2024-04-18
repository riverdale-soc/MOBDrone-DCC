"""
Uses Djikstra Algorithm to find shortest path between
two waypoints of mission plan represented as Weighted Graph
"""
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappush, heappop
# from dronekit import LocationGlobal    
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LocationGlobal(object):
    """
    A global location object.

    The latitude and longitude are relative to the `WGS84 coordinate system <http://en.wikipedia.org/wiki/World_Geodetic_System>`_.
    The altitude is relative to mean sea-level (MSL).

    For example, a global location object with altitude 30 metres above sea level might be defined as:

    :param lat: Latitude.
    :param lon: Longitude.
    :param alt: Altitude in meters relative to mean sea-level (MSL).
    """

    def __init__(self, lat, lon, alt=None):
        self.lat = lat
        self.lon = lon
        self.alt = alt

        # This is for backward compatibility.
        self.local_frame = None
        self.global_frame = None

    def __str__(self):
        return "LocationGlobal:lat=%s,lon=%s,alt=%s" % (self.lat, self.lon, self.alt)


@dataclass
class Waypoint:
    point: LocationGlobal
    index: int

    def __str__(self) -> str:
        return f"{self.point.lat} {self.point.lon} {self.point.alt}, {self.index}"


@dataclass
class Edge:
    u: int  # index of starting state
    v: int  # index of ending state
    weight: int = 1  # Unit time to travel from u to v in seconds

    def reversed(self):
        return Edge(self.v, self.u, self.weight)

    def __lt__(self, other) -> bool:
        return self.weight < other.weight

    def __str__(self) -> str:
        return f"{self.u} -> {self.v}"


class Mission:
    def __init__(self, home: LocationGlobal, TargetAltitude: int, Area: int, Cam_FOV: int,  MAX_RANGE: int, states=[]):
        self._waypoints = states
        self._edges = [[]]
        self.distance = 0
        self.physical_length = 0
        self.physical_area = 0
        self.mission_time = 0
        self.next = 0
        # Create default survey mission at 30 meters covering 160 meters
        # self.build_mission(home.lat, home.lon, TargetAltitude, Area, Cam_FOV, MAX_RANGE)
        print(f"Mission Plan Created...")
        print(f"Mission Length: {self.physical_length}m")
        print(f"Mission Area: {self.physical_area}m^2")
        print(f"Mission Time: {self.mission_time} minutes")


    @property
    def waypoint_count(self):
        return len(self._waypoints)

    @property
    def edge_count(self):
        return sum(map(len, self._edges))

    def index_of(self, waypoint) -> int:
        return self._waypoints.index(waypoint)

    def add_edge(self, edge: Edge):
        self._edges[edge.u].append(edge)

    def state_at(self, index: int):
        return self._waypoints[index]

    def add_waypoint(self, state):
        self._waypoints.append(state)
        self._edges.append([])
        return self.waypoint_count - 1

    def add_edge_by_indices(self, u, v, weight=1):
        edge = Edge(u, v, weight)
        self.add_edge(edge)

    def add_edge_by_vertices(self, first, second):
        u = self._waypoints.index(first)
        v = self._waypoints.index(second)
        self.add_edge_by_indices(u, v)

    def edges_for_index(self, index: int):
        return self._edges[index]

    def neighbors_for_index_with_weights(self, index):
        distance_tuples = []
        for edge in self.edges_for_index(index):
            distance_tuples.append((self.state_at(edge.v)))
        return distance_tuples

    def __str__(self):
        desc: str = ''
        for i in range(self.waypoint_count):
            desc += f"{self.waypoint_count(i)} -> {self.neighbors_for_index_with_weights(i)} \n"
        return desc

    # Follow the path from the first waypoint to the last
    def go_to_next(self):
        if self.next < self.waypoint_count:
            self.next += 1
            return self.state_at(self.next)
        else:
            return None

    @lru_cache(maxsize=None)
    def get_min_path(self, start, end):
        assert start in self._waypoints and end in self._waypoints
        distances, path_dict = self.dijkstra(start)
        path = path_dict_to_path(self.index_of(start),
                                 self.index_of(end), path_dict)
        return get_weighted_path(self, path)

    def dijkstra(self, root):
        # Find starting index
        first: int = self.index_of(root)
        # Distances are unknown first
        distances = [None] * self.waypoint_count
        # Root is 0 away from root
        distances[first] = 0
        path_dict = {}
        pq = PriorityQueue()
        pq.push(DijkstraNode(first, 0))
        while not pq.empty:
            u: int = pq.pop().state
            dist_u = distances[u]
            for we in self.edges_for_index(u):
                dist_v = distances[we.v]
                if dist_v is None or dist_v > we.weight + dist_u:
                    distances[we.v] = we.weight + dist_u
                    path_dict[we.v] = we
                    pq.push(DijkstraNode(we.v, we.weight + dist_u))
        return distances, path_dict

    # Start at first waypoint and traverse along the path, printing out the neighbors
    def traverse_along_path(self):
        for i in range(self.waypoint_count):
            # yield self.state_at(i)
            # yield self.state_at(i)
            print(f"{self.state_at(i)} -> {self.neighbors_for_index_with_weights(i)}")

    def display_mission(self):
        for i in range(self.waypoint_count):
            print(f"{self.state_at(i)} -> {self.neighbors_for_index_with_weights(i)}")

    # save gps coordinates to a csv file
    def save_mission_to_csv(self, filename: str):
        df = pd.DataFrame(columns=["Latitude", "Longitude", "Altitude"])
        for i in range(self.waypoint_count):
            df = df._append({"Latitude": self.state_at(i).point.lat, "Longitude": self.state_at(i).point.lon, "Altitude": self.state_at(i).point.alt}, ignore_index=True)
        df.to_csv(filename, index=False)

    # Display mission waypoints (longitude and latitude) and edges on plot
    def display_mission_on_plot(self):
        index = 0
        for i in range(self.waypoint_count):
            plt.plot(self.state_at(i).point.lon, self.state_at(i).point.lat, 'bo')
            # print(f"{self.state_at(i).point.lon} {self.state_at(i).point.lat}")
            for edge in self.edges_for_index(i):
                # label each point with index
                # get index of the next waypoint
                plt.text(self.state_at(i).point.lon, self.state_at(i).point.lat, f'{index}', fontsize=12, fontweight='bold', ha='right')
                plt.plot([self.state_at(i).point.lon, self.state_at(edge.v).point.lon], [self.state_at(i).point.lat, self.state_at(edge.v).point.lat], 'r-')
            index += 1
        # Add title including self.physical_length and self.physical_area, waypoint number, and home location
        plt.title(f"Mission: {self.physical_length}m, {self.physical_area}m^2, {self.waypoint_count} waypoints\n Home: {self.state_at(0).point.lat}, {self.state_at(0).point.lon}", fontsize=8, fontweight='bold')
        plt.show()

    def build_mission(self, home_lat: float, home_long: float, TargetAltitude: int, Area: int, Cam_FOV: int, MAX_RANGE: int):
        # We need to build a Mission with least waypoints holding GPS coordinates for each waypoint. 
        # Starting at home location, build LocationGlobal waypoints for each state in the mission plan
        # Allow for all area to be covered by the camera at a given altitude. 
        # We will use a 160 degree camera FOV and 30 meters altitude to cover 160 square meters of area
        # calculate area covered by camera at given altitude, assuming camera is facing directly down

        # Calculate area of 0 altitude image captured by camera at given altitude
        # in square meters using camera FOV in degrees and altitude in meters
        camera_area = ((math.tan(math.radians(Cam_FOV/2)) * TargetAltitude)*2)**2

        MAX_RANGE = MAX_RANGE / math.sqrt(camera_area)

        # width of the grid area to be covered by the camera in meters
        L = math.sqrt(Area)
        # How many waypoints to cover this width 
        waypoint_width = int(L / math.sqrt(camera_area)) + 1
        print(Area)
        print(L)
        # Make sure waypoint_width is odd, so center home location is in the middle of the grid
        if waypoint_width % 2 == 0:
            waypoint_width += 1
        
        # Furthest distance from the origin
        reach = L // 2
        if reach > MAX_RANGE:  # Change max range units of camera area
            reach = MAX_RANGE
        reach = int(reach)

        start = LocationGlobal(home_lat, home_long, TargetAltitude)
        # Create np Grid matrix of waypoints
        # Create waypoint_witdth x waypoint_width grid of waypoints.
        # Home coordinates will be in the middle of the grid. (0, 0)
        # Expand grid from center in all four quadrants.
        # If Reach = 2: Matrix looks like:
        #   (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2)
        #   (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1)
        #   (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)
        #   (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)
        #   (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2)
        #         
        # Create matrix based on reach
        # Create 2D matrix 
        transform = np.zeros((reach * 2 + 1, reach * 2 + 1), dtype=object)
        for y in range(-reach, reach + 1):
            # Iterate in reverse direction as done for x
            for x in range(-reach, reach + 1):
                transform[x + reach, y + reach] = (x, -y)

        # self.physical_length = ((reach * 2 + 1) * math.sqrt(camera_area))
        print(reach * 2 + 1)
        print(math.sqrt(camera_area))
        self.physical_area = self.physical_length ** 2

        # Generate edges for the graph
        path_matrix, indices = create_spiral_matrix((reach * 2) + 1)

        # Now build mission plan waypoints and assign edges in order of path_matrix indices
        # Add home location as first waypoint
        home = Waypoint(start, 0)
        self.add_waypoint(home)
        # Since we are starting at home, we can pop first index from indices
        indices = indices[1:]
        previous = 0
        for index in indices:
            # Get x and y coordinates
            point = transform_coordinate(home_lat, home_long, TargetAltitude, math.sqrt(waypoint_width), transform.flatten()[index])
            # print(point)
            # Create waypoint object
            waypoint = Waypoint(point, previous + 1)
            # Add waypoint to mission plan, and connect to next waypoint as defined in path_matrix
            self.add_waypoint(waypoint)
            # Add edges to the graph, connecting waypoints based on path_matrix,  and distance between waypoints
            # print(self._waypoints)
            self.add_edge_by_vertices(self._waypoints[previous], self._waypoints[previous + 1])
            self.distance += haversine_distance(self._waypoints[previous].point, self._waypoints[previous + 1].point)
            previous += 1
        # physical length is column 0 point 0 to last column last point distance
        self.physical_length = haversine_distance(self._waypoints[0].point, self._waypoints[-1].point)
        self.physical_area = self.physical_length ** 2
        print(f'physical length: {self.physical_length}')
        print(f'physical area: {self.physical_area}')
        print(f"distance: {self.distance}")
        # assign last waypoint to home
        self.add_edge_by_vertices(self._waypoints[previous], self._waypoints[0])
        # if UAV traveling at 3 m/s calculate time to complete mission, find time to travel entire distance in minutes
        self.mission_time = self.distance / 3 / 60



# Function that returns new GPS coordinates when adding some meters to longitude and latitude
def new_gps_coords(lat, lon, dNorth, dEast):
    earth_radius = 6378137.0
    # Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*lat/180))

    # New coordinates in degrees
    newlat = lat + dLat * 180/math.pi
    newlon = lon + dLon * 180/math.pi
    return LocationGlobal(newlat, newlon, 0)


# Get distance in meters between two GPS coordinates
def haversine_distance(aLocation1, aLocation2):
    # Radius of the Earth in meters
    lat1 = aLocation1.lat
    lon1 = aLocation1.lon
    lat2 = aLocation2.lat
    lon2 = aLocation2.lon
    R = 6371000.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Calculate differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    distance = R * c * 1000

    return distance


@dataclass
class DijkstraNode:
    state: int
    distance: float

    def __lt__(self, other) -> bool:
        return self.distance < other.distance

    def __eq__(self, other) -> bool:
        return self.distance == other.distance


class PriorityQueue:
    def __init__(self):
        self._container = []

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, item) -> None:
        heappush(self._container, item)

    def pop(self):
        return heappop(self._container)

    def __repr__(self):
        return repr(self._container)


def path_dict_to_path(start, end, path_dict):
    if len(path_dict) == 0:
        return []
    waypoint_path = []
    wp = path_dict[end]
    waypoint_path.append(wp)
    while wp.u != start:
        wp = path_dict[wp.u]
        waypoint_path.append(wp)
    return list(reversed(waypoint_path))


def get_weighted_path(wg, wp):
    path = []
    for waypoint in wp:
        print(f"{wg.state_at(waypoint.u)} {waypoint.TMS} > {wg.state_at(waypoint.v)}")
        path.append(waypoint.point)
    print(path)
    return path


def haversine(long1, lat1, long2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    long1, lat1, long2, lat2 = map(math.radians, [long1, lat1, long2, lat2])

    # haversine formula
    dlon = long2 - long1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    # Return distance in meters
    return c * r * 1000


def transform_coordinate(home_lat, home_long, targetAlt, unit, direction: (int, int)):
    """
    Transform GPS coordinates by adding some meters to longitude and latitude
    """
    x_scale, y_scale = direction

    new_long = home_long + (x_scale * unit) / (haversine(home_long, home_lat, home_long + 1, home_lat) * 1000)
    new_lat = home_lat + (y_scale * unit) / (haversine(home_long, home_lat, home_long, home_lat + 1) * 1000)
    
    return LocationGlobal(new_lat, new_long, targetAlt)


def create_spiral_matrix(n):
    """
    Create a 2D numpy array with a spiral pattern of incremental numbers.

    Parameters:
    - n: Size of the square matrix (n x n). n must be odd.

    Returns:
    - A 2D numpy array with a spiral pattern.
    """
    if n % 2 == 0:
        raise ValueError("n must be odd")

    matrix = np.zeros((n, n), dtype=int)

    mid = n // 2
    current_value = 0

    for layer in range(mid + 1):
        # Top row
        for i in range(mid - layer, mid + layer + 1):
            matrix[mid - layer, i] = current_value
            current_value += 1

        # Right column
        for i in range(mid - layer + 1, mid + layer + 1):
            matrix[i, mid + layer] = current_value
            current_value += 1

        # Bottom row
        for i in range(mid + layer - 1, mid - layer, -1):
            matrix[mid + layer, i] = current_value
            current_value += 1

        # Left column
        for i in range(mid + layer , mid - layer, -1):
            matrix[i, mid - layer] = current_value
            current_value += 1

    # Get list of indices that are arranged in increasing order of the elements in the matrix
    indices = np.argsort(matrix.flatten())

    return matrix, indices


if __name__ == "__main__":
    # home_location, target_altitude, area, camera_fov, max_range
    mission = Mission(LocationGlobal(40.919681, -73.352823, 30), 30, 100, 160, 150*150)
    mission.build_mission(40.919681, -73.352823, 30,
                      Area=10*10,
                      Cam_FOV=160,
                      MAX_RANGE=150*150)
    mission.display_mission_on_plot()
    mission.save_mission_to_csv("points.csv")