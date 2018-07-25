from random import choice
import matplotlib.pyplot as plt
import time


class RandomWalk:
    number_rw = 0
    def __init__(self,steps):
        self.steps = steps
        self.x_value = [0]
        self.y_value = [0]
        RandomWalk.number_rw += 1
        print(self.number_rw,end='/')
    
    def fill_walk(self):
        while len(self.x_value) < self.steps:
            x_direction = choice([1,-1])
            x_distance = choice(range(1,2))
            x_step = x_direction * x_distance

            y_direction = choice([1])
            y_distance = choice(range(1,2))
            y_step = y_direction * y_distance

            next_x = self.x_value[-1] + x_step
            next_y = self.y_value[-1] + y_step

            self.x_value.append((next_x))
            self.y_value.append((next_y))

    def plot_walk(self):
        point_numbers = list(range(self.steps))
        fig = plt.figure(dpi=1000)
        plt.scatter(self.x_value,self.y_value,c=point_numbers,cmap=plt.cm.Blues,edgecolor=None,s=1)
        plt.scatter(0,0,marker='p',c='green',edgecolor=None,s=10)
        plt.scatter(self.x_value[-1],self.y_value[-1],marker='p',c='red',edgecolor=None,s=15)
        # plt.axes().get_xaxis().set_visible(False)
        # plt.axes().get_yaxis().set_visible(False)
        plt.axis('off')
        fig.savefig('d:/Documents/GitHub/practice/数据可视化/%s' % self.number_rw)
        # plt.show()
        plt.close()

class Brownian_motion(RandomWalk):
    '''布朗运动'''
     
    def __init__(self,steps):
        super(Brownian_motion,self).__init__(steps)
        
    def plot_walk(self):
        '''绘制分子运动轨迹'''
        fig = plt.figure(dpi=1000)
        plt.plot(self.x_value,self.y_value,linewidth=0.5)
        plt.plot(0,0,marker='p',c='g')
        plt.plot(self.x_value[-1],self.y_value[-1],marker='p',c='r')
        plt.axis('off')
        fig.savefig('d:/Documents/GitHub/practice/数据可视化/%s' % self.number_rw)
        plt.close()

        

if __name__=='__main__':
    for i in range(1):
        rw = RandomWalk(50000)
        rw.fill_walk()
        rw.plot_walk()

    # bm = Brownian_motion(5000)
    # bm.fill_walk()
    # bm.plot_walk()
    





    


