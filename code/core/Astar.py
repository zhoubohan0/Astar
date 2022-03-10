from queue import PriorityQueue
import numpy as np

'''
[伪代码]
1.首先把起始位置点加入到一个称为“open List”的列表，
    在寻路的过程中，目前，我们可以认为open List这个列表
    会存放许多待测试的点，这些点是通往目标点的关键，
    以后会逐渐往里面添加更多的测试点，同时，为了效率考虑，
    通常这个列表是个已经排序的列表。

2.如果open List列表不为空，则重复以下工作：
（1）找出open List中通往目标点代价最小的点作为当前点；
（2）把当前点放入一个称为close List的列表；
（3）对当前点周围的4个点每个进行处理（这里是限制了斜向的移动），
    如果该点是可以通过并且该点不在close List列表中，则处理如下；
（4）如果该点正好是目标点，则把当前点作为该点的父节点，并退出循环，设置已经找到路径标记；
（5）如果该点也不在open List中，则计算该节点到目标节点的代价，把当前点作为该点的父节点，并把该节点添加到open List中；
（6）如果该点已经在open List中了，则比较该点和当前点通往目标点的代价，
    如果当前点的代价更小，则把当前点作为该点的父节点，
    同时，重新计算该点通往目标点的代价，并把open List重新排序；
3.完成以上循环后，如果已经找到路径，则从目标点开始，依次查找每个节点的父节点，直到找到开始点，这样就形成了一条路径。 
'''

inf = 1e8


class Map:
    def __init__(self, width, height, start, end, obstacles, mode):
        assert mode == 4 or mode == 8
        self.OBSTACLE = -1
        self.START = 1
        self.END = 2
        self.start = start
        self.end = end
        self.height = height
        self.width = width
        self.mode = mode
        # --------------------------------------------------
        self.mp = np.zeros((height, width))
        # set begin and end
        self.mp[start] = self.START
        self.mp[end] = self.END
        # set obstacles
        for x, y in obstacles:
            self.mp[x, y] = self.OBSTACLE


class Solver(Map):
    def __init__(self, width, height, start, end, obstacles, mode):
        super(Solver, self).__init__(width, height, start, end, obstacles, mode)
        self.mindistance = inf
        self.path = []

    def within(self, x, y):  # border detection
        return 0 <= x < self.height and 0 <= y < self.width

    def neighbors(self, node):  # get neighbors
        if self.mode == 4:
            direction = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if self.mode == 8:
            direction = [(-1, 0), (0, -1), (0, 1), (1, 0),
                         (-1, -1), (1, -1), (-1, 1), (1, 1)]
        return [(node[0] + x, node[1] + y) for (x, y) in direction if
                self.within(node[0] + x, node[1] + y) and self.mp[node[0] + x, node[1] + y] != self.OBSTACLE]

    def movecost(self, cur, near):  # move cost，移动距离由mode决定
        if self.mode == 8:
            ord = np.inf
        if self.mode == 4:
            ord = 1
        return np.linalg.norm(np.array(cur) - np.array(near), ord=ord)

    def heuristic(self, near, end):  # heuristic distance,启发式距离可人为设定，默认曼哈顿距离
        # 当mode = 4, ord = 1 / 2 / inf
        # 当mode = 8, ord = inf
        if self.mode == 8:
            ord = np.inf
        if self.mode == 4:
            ord = np.random.choice([1, 2, np.inf])
        return np.linalg.norm(np.array(end) - np.array(near), ord=ord)

    def A_star(self):  # search
        # init priority-queue
        q = PriorityQueue()
        q.put(self.start, int(0))
        # init path recorder
        comeFrom = {self.start: None}
        # init current cost recorder
        costSoFar = {self.start: 0}
        # searching
        while q.qsize():
            cur = q.get()
            if cur == self.end:
                break
            for near in self.neighbors(cur):
                newCost = costSoFar[cur] + self.movecost(cur, near)
                if near not in costSoFar or newCost < costSoFar[near]:  # 没有搜过的点相当于距离无穷大
                    costSoFar[near] = newCost
                    comeFrom[near] = cur
                    q.put(near, costSoFar[near] + self.heuristic(near, self.end))

        # terminate,find path recursively
        terminal = self.end
        path = [self.end]
        while comeFrom.get(terminal, None) is not None:
            path.append(comeFrom[terminal])
            terminal = comeFrom[terminal]
        path.reverse()
        self.mindistance = costSoFar.get(self.end, inf)
        self.path = path

    def outputresult(self):
        mindistance = self.mindistance if self.mindistance != inf else '∞'
        print(f'从{self.start}到{self.end}最短距离：{mindistance}')
        print('最短路径如下：')
        if len(self.path) == 1 and self.path[0] == end:
            print('empty path')
        else:
            for i, node in enumerate(self.path):
                print(node, end='')
                if i != len(self.path) - 1:
                    print('->', end='')
                else:
                    print()


def loadTestData(n=1):
    if n == 1:
        # 起始点
        start = (2, 2)
        end = (6, 12)
        # 创建障碍
        obstacle_y = [i for i in range(5, 10)]
        obstacle_x = [2] * len(obstacle_y)
        tmp = [i for i in range(3, 6)]
        obstacle_x.extend(tmp)
        obstacle_y.extend([9] * len(tmp))
        tmp = [i for i in range(5, 10)]
        obstacle_y.extend(tmp)
        obstacle_x.extend([6] * len(tmp))
        obstacles = zip(obstacle_x, obstacle_y)
    if n == 2:
        start = (0, 0)
        end = (3, 3)
        obstacles = [(3, 2), (3, 4), (2, 3), (4, 3)]
    return start, end, obstacles


if __name__ == '__main__':
    # 初始化地图基本属性
    WIDTH = 15
    HEIGHT = 10
    mode = 4
    start, end, obstacles = loadTestData(n=2)
    print(f'起点：{start} 终点：{end}')
    print('障碍：', *obstacles)
    print('----------------------------------------------------------------------------------------------------------------')

    # A*最短路径求解
    A_star_solver = Solver(WIDTH, HEIGHT, start, end, obstacles, mode)
    A_star_solver.A_star()
    A_star_solver.outputresult()
