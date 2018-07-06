import pygame
import random
import math
import os
pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,100,0)

display_width = 1400
display_height = 800
FPS = 45
pipe_gap = 180
between_pipe = 150
pipe_width = 100
pipe_speed = 2
score = 0
velocity = 10
pipe_count = display_width//(pipe_width+between_pipe)+1


game_folder = os.path.dirname(__file__)

class Bird(pygame.sprite.Sprite):
	"""docstring for Bird"""
	def __init__(self, x_loc, y_loc, velocity):
		super(Bird, self).__init__()
		self.velocity = velocity
		self.x_loc = x_loc
		self.y_loc = y_loc
		self.image = pygame.image.load(os.path.join(game_folder,"index2.png")).convert()
		self.image.set_colorkey(WHITE)
		self.image = pygame.transform.scale(self.image,(60,65))
		self.rect = self.image.get_rect()
		self.rect.center = (x_loc,y_loc)
	def update(self):
		self.rect.y += self.velocity
		self.velocity = self.velocity+1
	def jump(self):
		self.velocity = -10
	def boundary_collison(self):
		if self.rect.bottom+100>=display_height or self.rect.top<=0:
			return True


class UpperPipe(pygame.sprite.Sprite):
	"""docstring for UpperPipe"""
	def __init__(self, pipe_x, pipe_height, pipe_speed):
		super(UpperPipe, self).__init__()
		self.pipe_speed = pipe_speed
		self.image = pygame.Surface((pipe_width, pipe_height))
		self.image.fill(GREEN)
		self.rect = self.image.get_rect()
		self.rect.x = (pipe_x)
		self.rect.y = (0)
	def update(self):
		self.rect.x -= self.pipe_speed
	def x_cord(self):
		return self.rect.x

class LowerPipe(pygame.sprite.Sprite):
	"""docstring for UpperPipe"""
	def __init__(self, pipe_x, pipe_height, pipe_speed):
		super(LowerPipe, self).__init__()
		self.pipe_speed = pipe_speed
		self.image = pygame.Surface((pipe_width, display_height-(pipe_gap+pipe_height)))
		self.image.fill(GREEN)
		self.rect = self.image.get_rect()
		self.rect.x = (pipe_x)
		self.rect.y = (pipe_height+pipe_gap)
	def update(self):
		self.rect.x -= self.pipe_speed
	def x_cord(self):
		return self.rect.x
		
def new_pipe(pipe):
	pipe_x = pipe[0].x_cord()+pipe_gap+pipe_width
	pipe_height = (round(random.uniform(0.2,0.8), 2))*(display_height-pipe_gap)
	upper = UpperPipe(pipe_x,pipe_height,pipe_speed)
	lower = LowerPipe(pipe_x,pipe_height,pipe_speed)
	add_pipe = [upper,lower]
	sprites.add(upper)
	sprites.add(lower)
	return(add_pipe)

gameDisplay = pygame.display.set_mode((display_width,display_height))
myfont = pygame.font.SysFont("monospace", 16)
clock = pygame.time.Clock()
pygame.display.set_caption("Flappy Bird")

x_loc = display_width//8
y_loc = display_height//2
		
sprites = pygame.sprite.Group()

bird = Bird(x_loc,y_loc,velocity)
sprites.add(bird)

pipe_list = []
init_pipe_x = 500
for make in range(pipe_count):
	pipe_x = init_pipe_x+((between_pipe+pipe_width)*make)
	pipe_height = (round(random.uniform(0.2,0.8), 2))*(display_height-pipe_gap)
	upper = UpperPipe(pipe_x,pipe_height,pipe_speed)
	lower = LowerPipe(pipe_x,pipe_height,pipe_speed)
	add_pipe = [upper,lower]
	pipe_list.append(add_pipe)
	sprites.add(upper)
	sprites.add(lower)


pipe = pipe_list[0]
pipe_collision = pipe_list[0]
gameExit = False
while not gameExit:
	clock.tick(FPS)
	gameDisplay.fill(WHITE)

	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_q:
				gameExit = True
			if event.key == pygame.K_SPACE:
				bird.jump()


	if bird.boundary_collison():
		gameExit = True
	
	sprites.update()
	sprites.draw(gameDisplay)

	if (pipe[0].x_cord())+pipe_width <= 0:
		for k in pipe:
			pygame.sprite.Sprite.kill(k)
		pipe = pipe_list[1]
		del pipe_list[0]
		pipe_list.append(new_pipe(pipe_list[-1]))


	if ((pipe_collision[0].x_cord()+pipe_width)<x_loc):
		pipe_collision = pipe_list[1]

	for x in pipe_collision:
		if pygame.sprite.collide_rect(bird,x):
			gameExit = True

	scoretext = myfont.render("Score {0}".format(score), 1, (0,0,0))
	gameDisplay.blit(scoretext, (5, 10))
	score += 1

	pygame.display.flip()

pygame.quit()
quit()