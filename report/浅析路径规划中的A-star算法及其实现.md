#! https://zhuanlan.zhihu.com/p/478858388
# 浅析路径规划中的A-star算法及其实现

## 图搜索简介

图搜索总能产生一棵搜索树，高效+最优构建搜索树为算法核心。

<center><img src="https://pic4.zhimg.com/80/v2-692f206531ace157b607d611ad0354c5.png" style="zoom:80%;" /></center>

图搜索算法一般框架如下所示：

<center><img src="https://pic4.zhimg.com/80/v2-545c69adec35f25ccb93ae7b77b08e2c.png" style="zoom: 25%;" /></center>

### 盲目搜索方法

所有的图搜索算法都具有一种**容器(container)**和一种**方法(algorithm)**。

- “容器”在一定意义上就是开集，确定了算法的数据结构基础，以起始节点$S$​初始化，定义了结点进出的规则，深搜就是**栈(stack)**，广搜就是**队列(queue)**。
- “方法”确定了结点弹出的顺序，深搜(Depth First Search)中是“**弹出最深的节点**”，广搜(Breadth First Search)中是“**弹出最浅的节点**”（在树中表现为由根向叶层序推进）。

<center><img src="https://pic4.zhimg.com/80/v2-923a33fc1c088324a3dfd560bf9240b2.png" style="zoom:67%;" /></center>

需要注意的是，**DFS不能保证在一定的时空复杂度限制下寻找到最短路径**。因此，图搜索的基础是**BFS**。

### 启发式方法

一般地，BFS只适用于任意两点距离为1的图搜索找最短路径，而且属于“撒网式”的没有明确目标方向的盲目尝试。在BFS的基础上，重新定义结点出栈的顺序“具有最优属性的结点先弹出容器”，并升级容器为“优先队列”，就形成了具有启发式的路径搜索算法。
在正边权有向图中，每个节点的距离代价评估可用估价函数${f(n)}$来建模。
$$
f(n)=g(n)+h(n)\\
$$
其中$g(n)$是在状态空间中从初始节点到节点$n$的实际代价，$h(n)$是从节点$n$​到目标节点最佳路径的启发式估计代价，即“**启发式(heuristic)距离**”，“猜测”当前节点距离目标节点还有多远。
- Greedy：$f(n)=h(n)$​​​​
   - `策略`：不断访问**距终点启发距离**最小的邻点（默认当前点到所有邻点距离相同，不同则加上当前点到邻点距离代价）
   - 无障碍情况下比BFS高效；在最短路径上出现障碍则极大可能找不到最优解。 
- Dijkstra：$f(n)=g(n)$​
   - `策略`：不断访问**距原点累计距离**最小的邻点，邻点若未扩展过直接加入**优先队列**，若已扩展过（即在优先队列中）则进行**松弛**
   - `最优性保证`：已扩展点存储的一定是距离起始点的最短距离
   - 搜索过程均匀扩展（与边权相关），若任意两点距离为1退化为BFS
   - 伪代码如下：
     <center><img src="https://pic4.zhimg.com/80/v2-e97115807d457c29e837880ad08cefe3.png" style="zoom:80%;" /></center>
   - Dijkstra与Greedy算法的对比如下：
<center><img src="https://pic4.zhimg.com/80/v2-fe618a76b5abce1d3e0943e091ba4ed9.png" style="zoom: 33%;" /></center>

- A-star：${f(n)=g(n)+h(n)}$	
   - A*算法与Dijstra等一致代价搜索算法的主要区别在于启发项${h(n)}$的存在将优先队列的排序依据由$g(n)$变成$f(n)$。
     
      > A-star编程注意更新时要同步更新优先队列中每个节点的$g(n)$。

   - 估价距离${h(n)}$​**不大于**节点$n$​​到目标节点的距离时，搜索的点数多、范围大、效率低，保证得到最优解；若估价距离**大于**实际距离, 搜索的点数少、范围小、效率高，但不能保证得到最优解。估价值与实际值越接近，估价函数质量越高。
   - $h\le h^*$​​时保证算法完备性，举例如下：
     
      <center><img src="https://pic4.zhimg.com/80/v2-37476af9f9571ace194ed7282001353c.png" style="zoom: 33%;" /></center>
   - 伪代码如下：
     
     <center><img src="https://pic4.zhimg.com/80/v2-169286f32de835c8cb488240ac7015c1.png" style="zoom:67%;" /></center>

## A-star算法流程

**A***算法是静态路网中求解最短路最有效的方法之一，主要搜索过程伪代码示意如下：
```c++
//step 1
创建两个表，OPEN表保存所有已生成而未考察的节点，CLOSED表中记录已访问过的节点。
//step 2
遍历当前节点的各个节点，将n节点放入CLOSE中，取n节点的子节点X,算X的估价值
//step 3
While(OPEN!=NULL)
   {
    从OPEN表中取估价值f最小的节点n;
   	if(n节点==目标节点) break;
   	else
   	{
   		if(X in OPEN) 
            比较两个X的估价值f //注意是同一个节点的两个不同路径的估价值
   		if( X的估价值小于OPEN表的估价值 )
            更新OPEN表中的估价值; //取最小路径的估价值
		if(X in CLOSE) 
            比较两个X的估价值 //注意是同一个节点的两个不同路径的估价值
		if( X的估价值小于CLOSE表的估价值 )
　　　		  更新CLOSE表中的估价值; 把X节点放入OPEN //取最小路径的估价值
		if(X not in both)
			求X的估价值;并将X插入OPEN表中;　//还没有排序
	}
	将n节点插入CLOSE表中;按照估价值将OPEN表中的节点排序; 
    //(实际上是比较OPEN表内节点f的大小，从最小路径的节点向下进行。)
}
```
**A***算法框图展示如下：
<center><img src="https://pic4.zhimg.com/80/v2-495ed5d80efc31f23a7e62d0b26a99db.png" alt="Astar算法框图" style="zoom: 50%;" /></center>

## A-star算法实现

> [编译环境]
>
> Windows 系统|PyCharm 编译器|python 3.8.11

### 定义地图类
由长度、宽度、起点坐标、终点坐标、障碍坐标列表、地图模式（4邻接模式/8邻接模式）唯一确定一个地图类。

> [4邻接模式]：
>
> agent所有可能的移动范围包括上、下、左、右四个方向，一步行进一个单位长度
>
> [8邻接模式]：
>
> agent所有可能的移动范围包括上、下、左、右、左上、左下、右上、右下八个方向，一步行进一个单位长度

```python
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
```

### A*算法类

继承地图类的信息，类内成员变量和函数具体阐释如下：

<center><img src="https://pic4.zhimg.com/80/v2-80ba77945713f267a73cc5c847d1c0c9.png" style="zoom:80%;" /></center>

<center><img src="https://pic4.zhimg.com/80/v2-4ac191217c862f85e60fdcf671af484e.png" style="zoom:80%;" /></center>

```python
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
```

### 数据导入

```python
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

```

### 主函数

```python
if __name__ == '__main__':
    # 初始化地图基本属性
    WIDTH = 15
    HEIGHT = 10
    mode = 4
    start, end, obstacles = loadTestData(n=1)
    print(f'起点：{start} 终点：{end}')
    print('障碍：', *obstacles)
    print('------------------------------------------------------')

    # A*最短路径求解
    A_star_solver = Solver(WIDTH, HEIGHT, start, end, obstacles, mode)
    A_star_solver.A_star()
    A_star_solver.outputresult()
```

### 控制台测试

- 正常情况测试(存在最短路径)

<center><img src="https://pic4.zhimg.com/80/v2-065b1c0fc90c27bad6c9182458960e7a.png" alt="image-20211104163311758" style="zoom:80%;" /></center>

- 异常情况测试(4-邻接下无最短路径)

<center><img src="https://pic4.zhimg.com/80/v2-e4f4b33c55ebe7c21872343e0ec929de.png" alt="image-20211104163900320" style="zoom: 50%;" /></center>


## A-star算法可视化呈现

引入Python`PyQt5`第三方库，主要通过自行实现`GameBoard`类搭建窗口程序，完成A*算法在地图寻路上的应用(具体代码详见附件)。

窗口的主要区域为地图可视化显示，地图右侧分别展示窗口的使用说明、地图的颜色说明、操作功能键以及信息输出。通过加载预测地图或者根据使用说明设置地图后即可点击“开始搜索”进行寻路结果演示，算法寻找到的最优路径以及路径的最短距离在寻路演示之后会呈现在信息输出区域。

> [使用说明]
>
> 右键 : 首次单击格子选定起始点，第二次单击格子选定终点
>
> 左键 : 选定格子为墙壁，单击墙壁则删除墙壁
>
> [颜色说明]
>
> 黄色 : 代表起点
>
> 绿色 : 代表终点
>
> 黑色 : 代表墙壁
>
> 灰色 : 代表可行区域
>
> 红色 : 闪烁，代表最短路径上的每个节点

<center><img src="https://pic4.zhimg.com/80/v2-8f79f4b5de147350dd3ebfb7c4728487.gif" alt="demo" style="zoom:80%;" /></center>

视频演示中我们分别使智能体以4邻接和8邻接方式进行寻路，所使用的地图如上所示。结果比较如下：

```python
# 4邻接
从(0, 0)到(12, 15)最短距离：41
最短路径如下：
(0, 0)->(1, 0)->(2, 0)->(3, 0)->(4, 0)->(5, 0)->(6, 0)->(6, 1)->(6, 2)->(5, 2)->(4, 2)->(3, 2)->(2, 2)->(1, 2)->(0, 2)->(0, 3)->(0, 4)->(0, 5)->(0, 6)->(0, 7)->(0, 8)->(0, 9)->(1, 9)->(2, 9)->(3, 9)->(4, 9)->(5, 9)->(6, 9)->(6, 10)->(6, 11)->(5, 11)->(5, 12)->(5, 13)->(5, 14)->(5, 15)->(6, 15)->(7, 15)->(8, 15)->(9, 15)->(10, 15)->(11, 15)->(12, 15)
```
----
```python
# 8邻接
从(0, 0)到(12, 15)最短距离：21
最短路径：
(0, 0)->(1, 0)->(2, 0)->(3, 0)->(4, 0)->(5, 0)->(6, 0)->(7, 1)->(8, 2)->(9, 3)->(10, 4)->(11, 5)->(10, 6)->(11, 7)->(12, 8)->(12, 9)->(12, 10)->(12, 11)->(12, 12)->(12, 13)->(12, 14)->(12, 15)
```

应用A*算法在自己设计的游戏界面上运行顺利，我们继续探索，将算法应用在真实游戏中，实现功能：通过鼠标点击目标位置使游戏人物以最短路径到达指定位置。结果呈现见视频演示。

<center><img src="https://pic4.zhimg.com/80/v2-9b498e4e3349290ff76028df68fdc7c5.gif" alt="demo" style="zoom:80%;" /></center>

## 评价
- A*算法的核心代码部分主要基于**优先队列**的数据结构实现（底层结构为二叉堆），既凸显启发式算法的特征，在代码效率方面相比其他数据结构又有一定的提升；同时考虑到**无最短路径**的特殊情况，算法鲁棒性强。
- A*算法的核心代码以及可视化代码通过类进行封装并形成一个完整模块，便于改变地图模式，也便于代码的维护与调试。

## 扩展
### 地图路标形式
将地图的拓扑特征抽取出，使用路标形式存储地图可以有效提高算法寻路的效率。

<center><img src="https://pic4.zhimg.com/80/v2-26a0d5e56aed007b2ece0b47ccf731ee.png" alt="waypoints" style="zoom: 67%;" /></center>

### A-star算法工程应用

​	从更加宏观和一般的角度看待含有启发式信息的寻路算法：

<center><img src="https://pic4.zhimg.com/80/v2-ce67a1cf6afbe6d892900372469d3bb3.png" style="zoom:100%;" /></center>

> Weighted A-star：$f(n)=g(n)+\epsilon h(n),\epsilon > 1$
>
> - 用次优解换取更少的搜索时间，高估的启发距离使其更偏向贪心算法，可证明次优解质量满足：$cost\le \epsilon·cost^*$​
> - 还可以使$\epsilon$随搜索越来越接近1，在最优性和时间成本之间权衡
>

- 最合适的启发式函数

   由于h越接近h*越好，而在无障碍的栅格地图中最短路径一定沿**以起点终点确定的矩形的对角线**，因此可定义Diagonal Heuristic：

   ```python
   # D为水平/竖直移动代价；D2为斜线移动代价
   def heuristic(node,goal):
       dx = abs(node.x - goal.x)
       dy = abs(node.y - goal.y)
       return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
   ```

- 打破路径的对称性以减少搜索次数
   对于f相等的路径，A*中是无差别探索，结果趋向于找到多条最优路径。但实际上只需要一条，因此在搜索的时候可以设置“倾向”，仅找一条最短路径，思路可以选择如下几种：
   - 在f相同时选择h大/小的路线
   - 构建一张仅与坐标关联的随机数表，$h=h+\epsilon$
   - 趋向选择更接近对角线的路线
     ```python
     def h_(start,node,goal):
     	dx1 = abs(node.x - goal.x)
         dy1 = abs(node.y - goal.y)
         dx2 = abs(start.x - goal.x)
         dy2 = abs(start.y - goal.y)
         cross = abs(dy2*dx1-dx2*dy1)
         return h(node,goal) + cross * 0.001 
     	# h为原启发式函数，cross愈大相当于当前点离对角线上的点越远对原		 h给与更高的惩罚，cross的系数必须小不要逾越h_<=h*的最优条件
     ```
   - 稍微“打破”完备性条件

    <center><img src="https://pic4.zhimg.com/80/v2-9a1b4d739c5c3385aaa97ecdc52c21a6.jpg" style="zoom:50%;" /></center>

### D-star算法浅谈

**A***算法是静态路网中有效的寻路算法，而**D***算法是不断变化的动态环境下采用的有效寻路算法，其主要算法流程如下：

```c++
//step 1
先用Dijstra算法从目标节点G向起始节点搜索。储存路网中目标点到各个节点的最短路和该位置到目标点的实际值h,k。(k为所有变化h之中最小的值,当前为k=h。每个节点包含上一节点到目标点的最短路信息1(2),2(5),5(4)，4(7)。则1到4的最短路为1-2-5-4)
原OPEN和CLOSE中节点信息保存。
//step 2
/机器人沿最短路开始移动，在移动的下一节点没有变化时，无需计算，利用上一步Dijstra计算出的最短路信息从出发点向后追述即可，当在Y点探测到下一节点X状态发生改变(如堵塞)。机器人首先调整自己在当前位置Y到目标点G的实际值h(Y)，h(Y)=X到Y的新权值c(X,Y)+X的原实际值h(X).X为下一节点(到目标点方向Y->X->G），Y是当前点。k值取h值变化前后的最小。
//step 3
用A*或其它算法计算，这里假设用A*算法,遍历Y的子节点，点放入CLOSE,调整Y的子节点a的h值，h(a)=h(Y)+Y到子节点a的权重C(Y,a),比较a点是否存在于OPEN和CLOSE中，
//伪码示意
while()
{
 从OPEN表中取k值最小的节点Y;
 遍历Y的子节点a,计算a的h值 h(a)=h(Y)+Y到子节点a的权重C(Y,a)
 {
     if(a in OPEN)     
         比较两个a的h值 
     if( a的h值小于OPEN表a的h值 )
     {
      	更新OPEN表中a的h值;k值取最小的h值
         有未受影响的最短路经存在
         break; 
     }
     if(a in CLOSE) 
         比较两个a的h值 //注意是同一个节点的两个不同路径的估价值
     if( a的h值小于CLOSE表的h值 )
     {
      	更新CLOSE表中a的h值; k值取最小的h值;将a节点放入OPEN表
         有未受影响的最短路经存在
         break;
     }
     if(a not in both)
         将a插入OPEN表中;　//还没有排序
 }
 放Y到CLOSE表；
 OPEN表比较k值大小进行排序；
}
机器人利用第一步Dijstra计算出的最短路信息从a点到目标点的最短路经进行。
```

## 总结

本文从图搜索到A*算法从理论到实践分析比较了**A-star**算法的优势并给出其代码实现，并且进一步探讨了路标形式表示优化算法效率的方法以及了解了应用于动态环境下的D-star算法，为更复杂问题的寻路搜索提供了思路。

