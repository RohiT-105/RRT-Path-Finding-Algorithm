import matplotlib.pyplot as plt
import numpy as np
import random


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def plot(self, start=None, end=None, nodes=None, parents=None, paths=None, new_window=True):
        if new_window:
            plt.figure()  # This creates a new figure window for each call
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height, edgecolor='black', facecolor='none'))
        for obstacle in self.obstacles:
            plt.gca().add_patch(
                plt.Rectangle((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], edgecolor='red', facecolor='red'))
        if start:
            plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        if end:
            plt.plot(end[0], end[1], 'bo', markersize=10, label='End')
        if nodes and parents:
            for node in nodes:
                if parents[node] is not None:
                    plt.plot([node[0], parents[node][0]], [node[1], parents[node][1]], 'k-', linewidth=0.5)
                plt.plot(node[0], node[1], 'ro', markersize=3)
        if paths:
            for path, color, label in paths:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], color, linewidth=2, label=label)
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Environment with Obstacles')
        plt.legend()
        if new_window:
            plt.show()

    def is_collision(self, point):
        for obstacle in self.obstacles:
            if (obstacle[0] <= point[0] <= obstacle[0] + obstacle[2]) and (
                    obstacle[1] <= point[1] <= obstacle[1] + obstacle[3]):
                return True
        return False


class RRT:
    def __init__(self, start, goal, environment, step_size=0.25, max_iter=2000, **kwargs):
        self.start = start
        self.goal = goal
        self.env = environment
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [start]
        self.parents = {start: None}

    def get_random_point(self, spacing=0):
        x_min, x_max = spacing, self.env.width - spacing
        y_min, y_max = spacing, self.env.height - spacing
        return random.uniform(x_min, x_max), random.uniform(y_min, y_max)

    def get_nearest_node(self, point):
        nodes = np.array(self.tree)
        distances = np.linalg.norm(nodes - point, axis=1)
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def steer(self, from_node, to_point):
        vector = np.array(to_point) - np.array(from_node)
        distance = np.linalg.norm(vector)
        direction = vector / distance
        new_point = np.array(from_node) + direction * min(distance, self.step_size)
        return tuple(new_point)

    def path_to_goal(self, node):
        path = [node]
        while node is not None:
            node = self.parents[node]
            if node is not None:
                path.append(node)
        path.reverse()
        return path

    def run(self):
        for i in range(self.max_iter):
            random_point = self.get_random_point()
            nearest_node = self.get_nearest_node(random_point)
            new_point = self.steer(nearest_node, random_point)

            if not self.env.is_collision(new_point):
                self.tree.append(new_point)
                self.parents[new_point] = nearest_node

                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) < self.step_size:
                    print("Goal reached!")
                    return self.path_to_goal(new_point)
        print("Goal not reached within the iteration limit.")
        return None

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(np.array(p2) - np.array(p1))


def ramer_douglas_peucker(points, epsilon, environment):
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        dist = np.abs(np.cross(np.array(end) - np.array(start), np.array(start) - np.array(points[i]))) / RRT.distance(
            start, end)
        if dist > max_dist:
            index = i
            max_dist = dist
    if max_dist > epsilon:
        results1 = ramer_douglas_peucker(points[:index + 1], epsilon, environment)
        results2 = ramer_douglas_peucker(points[index:], epsilon, environment)
        return results1[:-1] + results2
    else:
        if not any(environment.is_collision(point) for point in points):
            return [start, end]
        else:
            intersecting_points = [start]
            for i in range(1, len(points)):
                segment = [start, points[i]]
                while environment.is_collision(segment[1]):
                    mid_point = ((segment[0][0] + segment[1][0]) / 2, (segment[0][1] + segment[1][1]) / 2)
                    segment = [segment[0], mid_point]
                intersecting_points.extend(ramer_douglas_peucker(segment, epsilon, environment)[1:])
                start = segment[1]
            return intersecting_points


def generate_spacing_path(path, env, spacing):
    spacing_path = [path[0]]

    for i in range(1, len(path)):
        prev_point = path[i - 1]
        curr_point = path[i]

        direction_vector = np.array(curr_point) - np.array(prev_point)
        distance = np.linalg.norm(direction_vector)

        if distance > 0:
            direction_vector /= distance

        j = 0
        while j < distance:
            new_point = tuple(np.array(prev_point) + j * direction_vector)
            nearest_obstacle_dist = min(env.width - new_point[0], new_point[0], env.height - new_point[1], new_point[1])
            adjusted_spacing = min(spacing, nearest_obstacle_dist / 2)
            spacing_path.append(new_point)
            j += adjusted_spacing

    spacing_path.append(path[-1])
    return spacing_path


def rrt_with_spacing(start_point, end_point, env, step_size=0.25, max_iter=2000, spacing=1):
    rrt = RRT(start=start_point, goal=end_point, environment=env, step_size=step_size, max_iter=max_iter)
    path = rrt.run()

    if path:
        epsilon = 0.5

        spacing_path = generate_spacing_path(path, env, spacing)
        return spacing_path
    return None


def rrt_with_greedy(start_point, end_point, env, step_size=0.25, max_iter=2000):
    rrt = RRT(start=start_point, goal=end_point, environment=env, step_size=step_size, max_iter=max_iter)

    for i in range(rrt.max_iter):
        random_point = rrt.get_random_point()

        if random.random() < 0.2:
            random_point = rrt.goal

        nearest_node = rrt.get_nearest_node(random_point)
        new_point = rrt.steer(nearest_node, random_point)

        if not rrt.env.is_collision(new_point):
            rrt.tree.append(new_point)
            rrt.parents[new_point] = nearest_node

            if np.linalg.norm(np.array(new_point) - np.array(rrt.goal)) < rrt.step_size:
                print("Goal reached with greedy approach!")
                path = rrt.path_to_goal(new_point)
                return path
    print("Goal not reached with greedy approach within the iteration limit.")
    return None


start_point = (0.5, 0.5)
end_point = (9.5, 9.5)
env = Environment(10, 10)
env.add_obstacle((1, 5.5, 3.5, 3.5))
env.add_obstacle((0.5, 2, 3, 3))
env.add_obstacle((6, 5, 2.5, 3.5))
env.add_obstacle((6, 1, 3.5, 3))

rrt = RRT(start=start_point, goal=end_point, environment=env)

path = rrt.run()
paths = []
if path:
    epsilon = 0.5
    smoothed_path = ramer_douglas_peucker(path, epsilon, env)
    paths.append((path, 'g-', 'Path'))
    paths.append((smoothed_path, 'b-', 'Smoothed Path'))

spacing_path = rrt_with_spacing(start_point, end_point, env, step_size=0.25, max_iter=2000, spacing=1)
if spacing_path:
    paths.append((spacing_path, 'm-', 'Spacing Path'))

greedy_path = rrt_with_greedy(start_point, end_point, env, step_size=0.25, max_iter=2000)
if greedy_path:
    paths.append((greedy_path, 'c-', 'Greedy Path'))

# For plotting each path variant in a separate window
env.plot(start=start_point, end=end_point, nodes=rrt.tree, parents=rrt.parents, paths=paths, new_window=True)

# If you want to show all variants in one plot for comparison
env.plot(start=start_point, end=end_point, nodes=rrt.tree, parents=rrt.parents, paths=paths, new_window=False)

# Assuming paths for different methods are calculated
env.plot(start=start_point, end=end_point, nodes=rrt.tree, parents=rrt.parents, paths=[(path, 'g-', 'Path')],
         new_window=True)
env.plot(start=start_point, end=end_point, nodes=rrt.tree, parents=rrt.parents,
         paths=[(spacing_path, 'm-', 'Spacing Path')], new_window=True)
env.plot(start=start_point, end=end_point, nodes=rrt.tree, parents=rrt.parents,
         paths=[(greedy_path, 'c-', 'Greedy Path')], new_window=True)


def average_nodes_for_variants(env, start_point, end_point, num_runs=20, step_size=0.25, max_iter=2000, spacing=1):
    rrt_nodes_counts = []
    spacing_nodes_counts = []
    greedy_nodes_counts = []

    for _ in range(num_runs):
        rrt = RRT(start=start_point, goal=end_point, environment=env, step_size=step_size, max_iter=max_iter)
        rrt.run()
        rrt_nodes_counts.append(len(rrt.tree))

        rrt_spacing = rrt_with_spacing(start_point, end_point, env, step_size=step_size, max_iter=max_iter, spacing=spacing)
        if rrt_spacing:
            spacing_nodes_counts.append(len(rrt_spacing))

        rrt_greedy = rrt_with_greedy(start_point, end_point, env, step_size=step_size, max_iter=max_iter)
        if rrt_greedy:
            greedy_nodes_counts.append(len(rrt_greedy))

    avg_rrt_nodes = np.mean(rrt_nodes_counts) if rrt_nodes_counts else None
    avg_spacing_nodes = np.mean(spacing_nodes_counts) if spacing_nodes_counts else None
    avg_greedy_nodes = np.mean(greedy_nodes_counts) if greedy_nodes_counts else None

    return avg_rrt_nodes, avg_spacing_nodes, avg_greedy_nodes


avg_rrt_nodes, avg_spacing_nodes, avg_greedy_nodes = average_nodes_for_variants(env, start_point, end_point)

# Plot the average number of nodes needed to reach the goal
labels = ['RRT', 'RRT with Spacing', 'Greedy RRT']
averages = [avg_rrt_nodes, avg_spacing_nodes, avg_greedy_nodes]

plt.figure()
plt.bar(labels, averages, color=['green', 'magenta', 'cyan'])
plt.xlabel('RRT Variants')
plt.ylabel('Average Number of Nodes')
plt.title('Average Number of Nodes to Reach Goal for Different RRT Variants')
plt.show()



