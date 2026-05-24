import torch
import numpy as np
import pygame
import threading


# pygame setup
pygame.init()
size = width, height = 1280, 720
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
running = True





#Make our data first lets say we want abunch of random points 
N = 10

#how many features the system has
dn_in = 1
dn_out = 1

#input data
data = torch.randn(N,dn_in)

#sets what the system is trying to find
true_x = torch.tensor([[2.0]])
true_b = torch.tensor(6.9)
true_y = data @ true_x + true_b  +torch.randn(N,dn_out) * 0.1 #adds noise so that the output varies

X = torch.randn(dn_in,dn_out, requires_grad= True)
B = torch.randn(1, requires_grad= True)

tpy_X : float
tpy_B : float
py_X  : float
py_B  : float

def model(X,B):   
    y = 0
    #how much we wanna nudge the data the smaller this is the more accurate
    learningrate =0.01
    #learning time 
    epoch = 100

        
    # converts tensor to primitive types

    tpy_X = true_x.item()
    tpy_B = true_b.item()
    #intial guess and front port
    for epoch in range(epoch):
       
        
        #this function mimics the actual function without the noise lol  forward propagation
        y = data @ X + B
        loss = torch.mean((y - true_y) **2)

        #this is our back propgation
        loss.backward()

        with torch.no_grad():
             X -= learningrate * X.grad; B -= learningrate * B.grad

        #resets the gradient
        X.grad.zero_(); B.grad.zero_()
        py_X = X.item()
        py_B = B.item()
    return y

# starting points
a = 0
b = height
dt : float

#function that calculates next poisition
# will draw a function accurately and at the bottom left like normal graphs
#acts as a function translator to pygame
def calculatepos(a,b,color,dt):
   
    a += dt * 0.5 # is input for the function

    #actual function is here
    if color == "blue":
        b = a*py_X + py_B
    else:
        b = a*tpy_X +tpy_B # will allow multiple color to runn different function simultanously

    b = -b  # reverse a so that the input can stay bottome left and move properly
    
    return[a,b+height]

#simplifies the draw cricle method into one function
def draw_circle(circle,dt):
    a = circle.a
    b = circle.b
    color = circle.color
    size = circle.size
    if a > width or b < 0:
       pygame.draw.line(screen,color,[0,height],[a,b],1)
       return 0,height
    NextPos = calculatepos(a,b,color,dt)   
    print(NextPos)
    a,b = NextPos[0],NextPos[1]
    pygame.draw.circle(screen,color,NextPos,size)
    return a,b

class Circle(object):
    a = 0.0
    b = 0.0
    color = ""
    size = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, a, b, color,size):
        self.a = a
        self.b = b
        self.color = color
        self.size = size

def make_circle(color,size):
    circle = Circle(0,height,color,size)
    return circle
#initalize the circles 
circle1 = make_circle("green",3)
circle2 = make_circle("blue",3)




#how much we wanna nudge the data the smaller this is the more accurate
learningrate =0.1
#learning time 
epoch = 100

        
# converts tensor to primitive types
tpy_X = true_x.item()
tpy_B = true_b.item()


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

    dt = clock.tick(120) # limits FPS to 60
    #the first epoch the game will render the first line 
    #after the line is rendered go to the next epoch

    #where the game renders
    for epoch in range(epoch):
        #this function mimics the actual function without the noise lol  forward propagation
        y = data @ X + B
        loss = torch.mean((y - true_y) **2)

        #this is our back propgation
        loss.backward()

        with torch.no_grad():
                X -= learningrate * X.grad; B -= learningrate * B.grad

        #resets the gradient
        X.grad.zero_(); B.grad.zero_()
        py_X = X.item()
        py_B = B.item()
        for i in range(100):
            screen.fill("black")
            circle1.a, circle1.b = draw_circle(circle1,dt) #one line draw circle function to keep code clean
            circle2.a, circle2.b = draw_circle(circle2,dt) #one line draw circle function to keep code clean
            # flip() the display to put your work on screen
            pygame.display.flip()
    
    



pygame.quit()




