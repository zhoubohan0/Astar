# -*- coding: UTF-8 -*-
'''*******************************
@ 开发人员：Mr.Zs
@ 开发时间：2020/5/1710:34
@ 开发环境：PyCharm
@ 项目名称：A-star算法->my_Astar_test.py
******************************'''
print(r'''**********算法伪代码*****************
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
**********算法伪代码*****************
''')
import random
#将地图中的点抽象化成类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other): #函数重载  #判断两个坐标  的  值 是否一样
        if((self.x == other.x )and (self.y == other.y)):
            return  True
        else:
            return False
    def __ne__(self, other):
        pass

class Map():          #首先定义地图类 创建地图
    obstacle_num = 100  # 障碍物的个数 实际障碍可能小于50
    def __init__(self,row,col):#构造方法  参数为行和列
        self.row = row
        self.col = col
        self.data = []
        self.data = [[0 for i in range(col)]for j in range(row)]   #列表推导式 创建地图的初始值为0
        # print(self.deta)
    def map_show(self):     #显示地图
        for i in range(self.row):
            for j in range(self.col):
                print(self.data[i][j],end =' ')
            print(' ')
    def obstacle(self,x,y):#在地图中设置障碍物
        self.data[x][y] = "|"     #障碍用1
    def map_obstacleshow(self,user,usec):#显示带有障碍的地图
        for x in range(self.obstacle_num):  # 循环obstacle_num次  随机生成障碍
            i = random.randint(0, (user-1))
            j = random.randint(0, (usec-1))
            self.obstacle(i,j)
        # self.map_show()

    def draw(self,point):
        self.data[point.x][point.y] = "*"
class Astar:    #A*算法
    class Node:
        def __init__(self,point,end_point,g):   #节点坐标
            '''
            :当前节点 current_point:#节点信息  包括坐标 父节点 移动代价g 估算代价h 总代价f
            :终点 end_point:
            :g  初始代价
            '''
            self.point = point  #当前位置
            self.end_point = end_point          #终点
            self.father = None      #父节点
            self.g = g              #g 为从起点到当前点的移动代价
            self.h = (abs(end_point.x - point.x) + abs(end_point.y - point.y)) * 10  # 计算h值 #曼哈顿算法算出当前点到终点的估算代价
            self.f = self.g + self.h  #计算总代价
        def near_Node(self,ur,ul):  #附近节点
            '''
            :左右移动 ur:
            :上下移动 ul:
            :返回值   return: 附近节点信息 继承Node 的信息
            '''
            nearpoint = Point(self.point.x+ur,self.point.y+ul)
            if abs(ur)==1 and abs(ul) == 1 :
                self.g = 5         #对角线 移动代价 为14
            else:                   #横着  或者 竖着走  代价为10
                self.g = 10

            nearnode = Astar.Node(nearpoint,self.end_point,self.g) #计算临近节点
            return nearnode  #返回临近节点信息
    def __init__(self,start_point,end_point,map):
        '''
        :起始坐标 start_point:
        :终点 end_point:
        :地图信息 map:
        '''
        self.start_point = start_point  #起点
        self.end_point = end_point  #终点
        self.current = 0        #当前节点
        self.map = map          #地图
        self.openlist = []      #打开节点  待测试的节点
        self.closelist = []     #已经测试过的节点
        self.path = []          #路径  存储每次选择的路径信息
    def select_node(self):  #选择一个代价最小的节点作为当前点
        '''
        在openlist中选择代价最小的节点  存放进closelist中
        :return:当前节点信息
        '''
        f_min = 1000   #初始设置  代价为1000
        node_temp = 0  #缓存节点
        for each in self.openlist:      #在openlist 中遍历 找出最小的代价节点
            if each.f < f_min:
                f_min = each.f
                node_temp = each
        self.path.append(node_temp)         #路径信息中存入这个路径
        self.openlist.remove(node_temp)     #将节点从待测试节点中删除
        self.closelist.append(node_temp)    #将节点加入到closelist中 表示已测试过
        return node_temp    #返回当前选择的节点   下一步开始寻找附近节点
    def isin_openlist(self,node):
        '''
        判断节点是否存在于openlist中 存在返回openlist中的原来的节点信息   不存在返回0
        :节点 node:
        :return:存在返回openlist中的原来的节点信息   不存在0
        '''
        for opennode in self.openlist:
            if opennode.point == node.point:
                return opennode
        return 0
    def isin_closelist(self,node):
        '''
        判断节点是否存在于closelist中 存在返回1   不存在返回0
        :节点 node:
        :return:存在返回1   不存在0
        '''
        for closenode in self.closelist:
            if closenode.point == node.point:
                return 1
        return 0
    def is_obstacle(self,node):#判断是否是障碍物
        if self.map.data[node.point.x][node.point.y]==1 :
            return  1
        return  0
    def search_nextnode(self,node):

        ud = 1
        rl = 0

        node_temp = node.near_Node(ud,rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = -1
        rl = 0
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = 0
        rl = 1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = 0
        rl = -1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = 1
        rl = 1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = 1
        rl = -1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = -1
        rl = 1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        ud = -1
        rl = -1
        node_temp = node.near_Node(ud, rl)  # 在调用另一个类的方法时（不论是子类还是在类外定义的类），都要进行实例化才能调用函数
        if node_temp.point == end_point:
            return 1
        elif self.isin_closelist(node_temp):
            pass
        elif self.is_obstacle(node_temp):
            pass
        elif self.isin_openlist(node_temp) == 0:
            node_temp.father = node
            self.openlist.append(node_temp)
        else:
            if node_temp.f < (self.isin_openlist(node_temp)).f:
                self.openlist.remove(self.isin_openlist(node_temp))
                node_temp.father = node
                self.openlist.append(node_temp)

        return 0

start_point = Point(10,13)
end_point = Point(1,0)


userow = 15     #实际使用的行
usecol = 15     #实际使用的列
mapshow = Map(userow,usecol)        #地图 实例化
# mapshow.map_show()#显示地图
# print('__________________________')
mapshow.map_obstacleshow(userow,usecol)
mapshow.draw(start_point)
mapshow.draw(end_point)
# mapshow.map_show()#显示地图
#初始化设置
astar = Astar(start_point,end_point,mapshow)  #实例化 A*算法
start_node = astar.Node(start_point,end_point,0)  #获取到当前节点  也就是起始点的所有信息
astar.openlist.append(start_node)           #将起始点添加进openlist中

flag = 0
while flag!=1:
    astar.current = astar.select_node()#从openlist中选取一个代价最小的node节点 作为当前节点
    flag=astar.search_nextnode(astar.current)#对选中的当前node进行周边探索
#画出地图路径
for node_path in astar.closelist:
    mapshow.draw(node_path.point)
mapshow.map_show()#显示地图

