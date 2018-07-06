import pygame
import random
import math
import os
import numpy as np

'''
Game Variables
'''

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,100,0)

display_width = 1400
display_height = 800
FPS = 45
pipe_gap = 200
between_pipe = 300
pipe_width = 100
pipe_speed = 4
score = 0
velocity = 10
pipe_count = display_width//(pipe_width+between_pipe)+2

'''
Game Variables
'''

'''
Genetic Variables
'''

population = 350	# Total Population
hidden_nodes = 16	# Number of Nodes inside the hidden layer
inp = 4				# Number of inputs + Bias
bias1 = np.random.uniform()	# Input Layer Bias
bias2 = np.random.uniform()	# Hidden Layer Bias
'''
Inputs :
1. Velocity of the Bird
2. Distance between the center of the bird and the center of the gap of the pipe vertically
3. Distance between the center of the bird and the center of the gap of the pipe horizontally
4. Bias
'''

game_folder = os.path.dirname(__file__)

master_parameters = [np.zeros(shape=(inp,hidden_nodes)),np.zeros(shape=hidden_nodes+1)]
master_parameters = np.asarray(master_parameters)  # Will be used to store the best bird and mutate it.

'''
Genetic Variables
'''

'''
Genetic Algorithm
'''

def sigmoid(value):
	value = float(math.exp(-value))
	value = float(value + 1)
	value = float(1/value)
	return value

def nn(arr,paras,bias):
	hidden_activations = np.dot(arr,paras[0])
	hidden_activations = [bias] + list(map(sigmoid,hidden_activations))
	return sigmoid(np.dot(hidden_activations,paras[1]))
def mutate(master):
	mutation = np.random.normal(scale=3)
	return (master+mutation)

def make_parameters(master,population):
	para_list = [master]
	for make in range(population-1):
		para_list.append(mutate(master))
	return para_list

'''
Genetic Algorithm
'''

'''
Pygame Sprite Classes
'''

class Bird(pygame.sprite.Sprite):
	""" Will contain the bird attributes.
		Args:
		x_loc (int): X - coordinate of the center of the bird sprite
		y_loc (int): Y - coordinate of the center of the bird sprite
		velocity (int): Velocity of the bird sprite. """
	def __init__(self, x_loc, y_loc, velocity):
		super(Bird, self).__init__()
		self.check = 0
		self.velocity = velocity
		self.x_loc = x_loc
		self.y_loc = y_loc
		self.image = pygame.image.load(os.path.join(game_folder,"index.png")).convert()
		self.image.set_colorkey(WHITE)
		self.image = pygame.transform.scale(self.image,(40,40))
		self.rect = self.image.get_rect()
		self.rect.center = (x_loc,y_loc)
		self.mask = pygame.mask.from_surface(self.image)
	def update(self):
		self.rect.y += self.velocity
		self.velocity = self.velocity+1
	def jump(self):
		self.velocity = -10
	def boundary_collison(self):
		if self.rect.bottom+100>=display_height or self.rect.top<=0:
			return True
	def bird_center(self):
		return self.rect.center
	def vel(self):
		return velocity


class UpperPipe(pygame.sprite.Sprite):
	""" Will contain the upper pipe's attributes.
		Args:
		pipe_x (int): X - coordinate of the starting of the pipe
		pipe_height (int): Height of the upper pipe
		pipe_speed (int): Pipe speed with which they pipe's will move horizontally. """
	def __init__(self, pipe_x, pipe_height, pipe_speed):
		super(UpperPipe, self).__init__()
		self.pipe_speed = pipe_speed
		self.pipe_height = pipe_height
		self.image = pygame.Surface((pipe_width, pipe_height))
		self.image.fill(GREEN)
		self.image.set_colorkey(WHITE)
		self.rect = self.image.get_rect()
		self.rect.x = (pipe_x)
		self.rect.y = (0)
		self.mask = pygame.mask.from_surface(self.image)
	def update(self):
		self.rect.x -= self.pipe_speed
	def x_cord(self):
		return self.rect.x
	def y_cord(self):
		return (self.rect.y+self.pipe_height)

class LowerPipe(pygame.sprite.Sprite):
	""" Will contain the lower pipe's attributes.
		Args:
		pipe_x (int): X - coordinate of the starting of the pipe
		pipe_height (int): Height of the lower pipe
		pipe_speed (int): Pipe speed with which they pipe's will move horizontally. """
	def __init__(self, pipe_x, pipe_height, pipe_speed):
		super(LowerPipe, self).__init__()
		self.pipe_speed = pipe_speed
		self.image = pygame.Surface((pipe_width, display_height-(pipe_gap+pipe_height)))
		self.image.fill(GREEN)
		self.image.set_colorkey(WHITE)
		self.rect = self.image.get_rect()
		self.rect.x = (pipe_x)
		self.rect.y = (pipe_height+pipe_gap)
		self.mask = pygame.mask.from_surface(self.image)
	def update(self):
		self.rect.x -= self.pipe_speed
	def x_cord(self):
		return self.rect.x
	def y_cord(self):
		return self.rect.y

'''
Pygame Sprite Classes
'''

# Will create a new set of upper and lower pipe everytime an existing
# pipe moves off the screen.
def new_pipe(pipe):
	pipe_x = pipe[0].x_cord()+between_pipe+pipe_width
	pipe_height = (round(np.random.uniform(0.15,0.85), 2))*(display_height-pipe_gap)
	upper = UpperPipe(pipe_x,pipe_height,pipe_speed)
	lower = LowerPipe(pipe_x,pipe_height,pipe_speed)
	add_pipe = [upper,lower]
	pipe_group.add(upper)
	pipe_group.add(lower)
	return(add_pipe)

def init_para():
	parameter_list = []
	for iii in range(population):
		m_parameters = [np.random.normal(size=(inp,hidden_nodes)),np.random.normal(size=hidden_nodes+1)]
		m_parameters = np.asarray(m_parameters)
		parameter_list.append(m_parameters)
	return parameter_list

'''
Main Game Run Function
'''

def run_game(generation,score):
	myfont = pygame.font.SysFont("monospace", 16)
	run_score = 0
	best_score = np.zeros(shape=population)
	global master_parameters
	global check
	global gameExit
	global pipe_collision
	global pipe
	global bias1
	global bias2
	cur_index = 0
	while not gameExit:
		clock.tick(FPS)
		gameDisplay.fill(WHITE)
		for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						gameExit = True
		for bird_index in range(len(bird_list)):
			if check[bird_index]==0:
				bird = bird_list[bird_index]
				arr = [bias1]
				bird_x,bird_y = bird.bird_center()
				arr.append(bird.vel())
				xpip = (pipe_collision[0].x_cord()+(pipe_width//2))/(display_width)
				ypip = (pipe_collision[0].y_cord()+(pipe_gap//2))/(display_height)
				arr.append((bird_x/display_width)-xpip)
				arr.append((bird_y/display_height)-ypip)
				direct_distance = (arr[1]**2)+(arr[2]**2)
				direct_distance = math.sqrt(direct_distance)
				fitness = run_score - direct_distance
				out = nn(arr,parameter_list[bird_index],bias2)
				if out>0.5:
					bird.jump()
				if bird.boundary_collison():
					pygame.sprite.Sprite.kill(bird)
					if fitness>score:
						best_score[bird_index] = (fitness)
					check[bird_index] = 1
				for x in pipe_collision:
					c=0
					if pygame.sprite.collide_rect(bird,x):
						bird_hits = pygame.sprite.spritecollide(bird,pipe_group,False,pygame.sprite.collide_mask)
						if bird_hits:
							c = 1
							pygame.sprite.Sprite.kill(bird)
							if fitness>score:
								best_score[bird_index] = (fitness)
							check[bird_index] = 1
					if c==1:
						break
						
			if sum(check)==len(check):
				if max(best_score)>score:
					master_parameters = parameter_list[list(best_score).index(max(best_score))]
					score = max(best_score)
				return master_parameters,score
		
		sprites.update()
		pipe_group.update()
		sprites.draw(gameDisplay)
		pipe_group.draw(gameDisplay)

		if (pipe[0].x_cord())+pipe_width <= 0:
			for k in pipe:
				pygame.sprite.Sprite.kill(k)
			pipe = pipe_list[1]
			del pipe_list[0]
			pipe_list.append(new_pipe(pipe_list[-1]))


		if ((pipe_collision[0].x_cord()+pipe_width)<x_loc):
			pipe_collision = pipe_list[1]


		gen = myfont.render("Genertation {0}".format(generation), 1, (0,0,0))
		highest = myfont.render("Highest Score {0}".format(int(round(score))), 1, (0,0,0))
		current = myfont.render("Current Score {0}".format(run_score), 1, (0,0,0))
		gameDisplay.blit(gen, (5, 10))
		gameDisplay.blit(highest, (5, 35))
		gameDisplay.blit(current, (5, 60))
		run_score += 1
		pygame.display.flip()

'''
Main Game Run Function
'''

score = 0	# Starting Score Initialization.
generation = 1 	# Generation Initialization.

parameter_list = init_para() #Describing unique parameters for every bird in the population.


'''
Main control loop -
Will be responsible to run the game multiple times and
mutate the best bird after the end of each game
'''

while True:
	pygame.init()
	gameDisplay = pygame.display.set_mode((display_width,display_height))
	clock = pygame.time.Clock()
	pygame.display.set_caption("Flappy Bird")

	x_loc = display_width//8
	y_loc = display_height//2
			
	sprites = pygame.sprite.Group()
	pipe_group = pygame.sprite.Group()
	y_locations = np.random.randint(low = 0,high=display_height,size=population)
	bird_list = []
	for make_bird in range(population):
		bird = Bird(x_loc,y_locations[make_bird],velocity)
		bird_list.append(bird)
		sprites.add(bird)

	pipe_list = []
	init_pipe_x = 500
	for make in range(pipe_count):
		pipe_x = init_pipe_x+((between_pipe+pipe_width)*make)
		pipe_height = (round(np.random.uniform(0.15,0.85), 2))*(display_height-pipe_gap)
		upper = UpperPipe(pipe_x,pipe_height,pipe_speed)
		lower = LowerPipe(pipe_x,pipe_height,pipe_speed)
		add_pipe = [upper,lower]
		pipe_list.append(add_pipe)
		pipe_group.add(upper)
		pipe_group.add(lower)
	pipe = pipe_list[0]
	pipe_collision = pipe_list[0]
	gameExit = False
	check=[]
	for i in range(population):
		check.append(0)
	master_parameters,score = run_game(generation,score)
	pygame.quit()
	print("Generation =",generation,"---------- Highest Score =",int(score))
	parameter_list1 = [master_parameters]*int(population*0.1)
	parameter_list2 = make_parameters(master_parameters,int(population*0.9))
	parameter_list = parameter_list1+parameter_list2
	generation+=1