import ast
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
import networkx as nx
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Disable logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Global variables
current_node = 0
dest_node = 0
arrives = 0
estimated_time_changes = 0
nav_path=[]
video_info = {}
estimated_times = {}
estimated_time = 0
start_timer = False
pre = 1
cur_speed = 0

def timer():
    global estimated_time
    global estimated_times
    global estimated_time_changes
    while True:
        if (current_node, dest_node) in estimated_times:
            break
    while not start_timer:
        pass
    estimated_time = estimated_times[(current_node, dest_node)]
    while True:
        if estimated_time <= 0:
            break
        if estimated_time_changes == 1:
                # Update estimated_time if estimated_time_changes
                estimated_time = estimated_times.get((current_node, dest_node))
                estimated_time_changes = 0  # Reset estimated_time_changes
        else:
            while estimated_time > 0:
                estimated_time -= 1
                if estimated_time_changes==1:
                    break
                time.sleep(1)
    estimated_time = 0    
    

def create_graph():
    G = nx.Graph()
    
    print("Enter nodes separated by spaces:")
    nodes = input().split()
    G.add_nodes_from(nodes)
    
    print("Enter edges separated by spaces (e.g., 'node1 node2 distance (m)'). Type 'done' when finished:")
    while True:
        edge_input = input()
        if edge_input.lower() == 'done':
            break
        else:
            node1, node2, distance = edge_input.split()
            G.add_edge(node1, node2, distance=float(distance), vehicle_num=0)
    return G

def assign_videos(G):
    for edge in G.edges():
        while True:
            print(f"Enter video for edge {edge[0]} - {edge[1]}:")
            video = input()
            if video:
                G.edges[edge]['video'] = video
                break
            else:
                print("Please enter a valid video name.")

def draw_graph(G):
    global estimated_time_changes
    global estimated_time
    plt.figure(num='Navigator')
    plt.clf()

    # Create a color map for the nodes
    node_colors = ['#00EC00' if node == current_node else '#FF2D2D' if node == dest_node else '#8E8E8E' for node in G.nodes()]
    # Draw the graph
    pos = nx.spring_layout(G, seed=0)  # Positions for all nodes with fixed seed
    nx.draw(G, pos, with_labels=True, node_color=node_colors)

    # Draw the labels
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Draw edge labels for 'vehicle_num' and 'distance' attributes
    edge_labels = {(u, v): f"Vehicle_num: {G.edges[u, v]['vehicle_num']}\nDistance: {G.edges[u, v]['distance']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw the navigation path
    if arrives == 0:
        nav_edges = [(nav_path[i], nav_path[i+1]) for i in range(len(nav_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=nav_edges, edge_color='#CA8EFF', width=2.0)

    # Add legend for node colors and navigation path
    plt.text(-0.1, 1.1, 'Legend:', transform=plt.gca().transAxes)
    red_patch = mpatches.Patch(color='#FF2D2D', label='Destination Node')
    green_patch = mpatches.Patch(color='#00EC00', label='Current Location')
    violent_line = mpatches.Patch(color='#CA8EFF', label='Navigation Path')
    plt.legend(handles=[red_patch, green_patch, violent_line], loc='upper left')

    if (current_node, dest_node) in estimated_times:
        plt.text(0.5, 0.9, f'Estimated arrival time to {dest_node}: {round(estimated_time, 3)} s', transform=plt.gca().transAxes, color='black')


    # Set the title for the plot
    plt.title('Navigator')

    # Show the plot
    plt.pause(0.001)

def perform_vehicle_detection(G):
    global arrives
    global video_info
    global pre
    model = YOLO("yolov8s.pt")
    desired_class_indices = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

    video_capture = {}
    for edge in G.edges():
        video_capture[edge] = cv2.VideoCapture(G.edges[edge]['video']) # can be changed to camera
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # mps for MacOS Metal Performance Shader
    while True:
        frames = {}
        for edge in G.edges():
            ret, frame = video_capture[edge].read()
            if not ret:
                break
            frames[edge] = frame

        if len(frames) == 0:  # All videos finished or arrived
            break
        
        for edge, frame in frames.items():
            rx, ry, rx2, ry2 = ast.literal_eval(video_info[G.edges[edge]['video']].get('region_of_interest'))
            roi_frame = frame[ry:ry2, rx:rx2]
            results = model(roi_frame, device=device)
            result = results[0]
            
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            # Only consider vehicles
            desired_class_boxes = []
            for desired_class_index in desired_class_indices:
                class_boxes = bboxes[classes == desired_class_index]
                desired_class_boxes.extend(class_boxes)

            num_rectangles = len(desired_class_boxes)
            # Update edge vehicle_num
            G.edges[edge]['vehicle_num'] = num_rectangles  
            
            for bbox in desired_class_boxes:
                (x, y, x2, y2) = bbox
                x = x + rx
                y = y + ry
                x2 = x2 + rx
                y2 = y2 + ry
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (rx, ry), (rx2, ry2), (255, 0, 0), 2)
            text = f"Number of Vehicles: {num_rectangles}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = frame.shape[0] - 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f"Edge {edge[0]} - {edge[1]}", frame)
            
            if (arrives == 0 or arrives == 1):
                draw_graph(G)
                if (arrives == 1):
                    draw_graph(G)
                    arrives = 2
                    break
        if (pre == 1):
            pre = 0
            return
        key = cv2.waitKey(1)
        if key == 27:
            break

    for cap in video_capture.values():
        cap.release()
    cv2.destroyAllWindows()

def simulate_car_movement(G, start_node, end_node):
    global nav_path
    global current_node 
    current_node = start_node
    global dest_node 
    dest_node = end_node
    global arrives
    global start_timer
    global cur_speed
    global estimated_time_changes
    
    total_time = 0
    # collect traffic information time.sleep(5) 
    
    heuristic_function(current_node, dest_node)
    start_timer = True
    while current_node != end_node:
        # Calculate shortest path using A* algorithm
        shortest_path = nx.astar_path(G, current_node, end_node, heuristic=heuristic_function , weight=weight_function)
        nav_path = shortest_path
        # Next node is the next node in the shortest path
        next_node = shortest_path[1]
        max_vehicle_length = 5
        edge = (current_node, next_node)
        speed_limit = int(video_info[G.edges[edge]['video']].get('speed_limit'))
        lanes = int(video_info[G.edges[edge]['video']].get('lanes'))
        cost_now = int(G.edges[edge]['vehicle_num'])/int(video_info[G.edges[edge]['video']].get('captured_lane_length'))
        # Calculate maximum density
        km = lanes / max_vehicle_length
        vf = speed_limit
        cur_speed = real_speed(vf, cost_now, km) # this should be based on your driving behavior or vehicle design
                                                 # currently use underwoods speed model to simulate
        time_to_next_node = G.edges[edge]['distance']/cur_speed
        
        estimated_times[current_node, dest_node] = sum(G[shortest_path[i]][shortest_path[i+1]]['distance'] for i in range(len(shortest_path)-1))/cur_speed
        estimated_time_changes = 1
        time.sleep(time_to_next_node)
        print(f"Move from node {current_node} to node {next_node} in {time_to_next_node} seconds")
        total_time += time_to_next_node
        current_node = next_node
    print(f"Total time taken: {total_time} seconds")
    arrives = 1

def heuristic_function(current_node, end_node):
    global estimated_time_changes
    global cur_speed
    length = nx.dijkstra_path_length(G, current_node, end_node, weight='distance')
    if (cur_speed!=0):
        estimated_times[current_node, end_node] = length/cur_speed
        estimated_time_changes = 1
    return length

def weight_function(current_node, neighbor_node, edge_attr):
    max_vehicle_length = 5
    speed_limit = int(video_info[edge_attr.get('video')].get('speed_limit'))
    lanes = int(video_info[edge_attr.get('video')].get('lanes'))
    cost_now = int(edge_attr.get('vehicle_num'))/int(video_info[edge_attr.get('video')].get('captured_lane_length'))
    # Calculate maximum density
    km = lanes / max_vehicle_length
    vf = speed_limit
    v = real_speed(vf, cost_now, km)
    apparent_distance = int(edge_attr.get('distance')) * vf/v
    return apparent_distance

def real_speed(vf, k, km):
    # Calculate speed v using Underwood R T.'s flow theory
    #####################################################################################
    v = vf * math.exp(-(k / km))                                                        #
    #  Underwood R T. Speed, volume and density relationships.                          #
    #  In: Quality and Theory of Traffic Flow. New Haven, Conn:                         #
    #  Bureau of Highway Traffic, Yale University, 1961. 141—187                        #
    ##################################################################################### 
    return v

def main():
    global G
    # Read configuration file
    # Initialize video_name
    video_name = None

    # Read configuration file
    with open('conf.dat', 'r') as file:
        lines = file.readlines()

    # Parse each line and store data in video_info
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().rstrip(',').split(':')
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                if key.endswith('.mp4'):
                    video_name = key
                    video_info[video_name] = {}
                else:
                    if video_name is not None:
                        if key and value:
                            video_info[video_name][key] = value
    G = create_graph()
    assign_videos(G)
    print("Enter Your Starting Node:")
    start = input()
    print("Enter Your Destination Node:")
    dest = input()
    # Threading for simultaneous execution
    thread = threading.Thread(target=simulate_car_movement, args=(G, start, dest))  # Start and end nodes, car speed
    thread2 = threading.Thread(target=timer) # Start timer
    perform_vehicle_detection(G) # get initial data
    thread.start()
    thread2.start()
    perform_vehicle_detection(G)
    thread.join()
    thread2.join()

if __name__ == "__main__":
    main()
