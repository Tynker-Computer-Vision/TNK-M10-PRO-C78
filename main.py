'''
SA1: Load configuration file and run the ganme for each genome in the generation 
SA2: Stop the game if the car goes out of the track
SA3: Control the game using NN    
'''
import pygame,math
import neat

pygame.init()
screen = pygame.display.set_mode((800,600))

pygame.display.set_caption("Car racing")
background_image = pygame.image.load("track.png").convert()
player_image = pygame.image.load("car.png").convert_alpha()

player=pygame.Rect(60,300,20,20)

WHITE=(255,255,255)
xvel=2
yvel=3
angle=0
change=0

distance=2
forward=False

font = pygame.font.Font('freesansbold.ttf', 12)

def newxy(x,y,distance,angle):
  angle=math.radians(angle+90)
  xnew=x+(distance*math.cos(angle))
  ynew=y-(distance*math.sin(angle))
  return xnew,ynew

# SA2: Check if car is outside the track
def checkOutOfBounds(car):
  x = car.x
  y = car.y
  width = car.width
  height = car.height

  if(checkPixel(x,y) or checkPixel(x+width, y) or checkPixel(x, y+height) or checkPixel(x+width, y+height)):
      return True
  
# SA2: Check color of the pixel    
def checkPixel(x, y):
    global screen
    try:
        color = screen.get_at((x, y))
    except:
        return 1
    if(color == (163,171,160,255)):
        return 0
    return 1

gen=0
angle =0

# SA1: Create a function eval_fitness which is called automaticalling   
def eval_fitness(generation, config):
    # SA1: Use global keyword as we are writing the game code inside a function, and in python global is required when accessing the global variables inside a function
    global angle, gen, forward, change
    
    # SA1: Gen count to show which generation is this (Can be given as boiler to uncomment)
    gen = gen+1
    # SA1: Genome count to show which genome is running (can be given as boiler to uncomment)
    genomeCount = 1
    
    # SA1: Run the game code for each genome in the generation
    for gid, genome in generation:
        
        # SA3: Create a neural network using current genome and configuration
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # SA1: Print the text on game screen (Can be given as boiler to uncommnet)
        infoText = font.render('Generation :'+ str(gen)+  ' genomecount: '+str(genomeCount)+"/"+str(len(generation)) , True, (255,255,0))
        
        while True:
          screen.blit(background_image,[0,0])
          screen.blit(infoText, (320, 20))
         
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              pygame.quit()
              
            if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  change = 5
               if event.key ==pygame.K_RIGHT:
                change = -5 
               if event.key == pygame.K_UP:
                forward = True
                
            if event.type == pygame.KEYUP:
              if event.key ==pygame.K_LEFT or event.key == pygame.K_RIGHT:
                  change = 0
              if event.key == pygame.K_UP:
                forward = False 
            
          if forward:
              player.x,player.y=newxy(player.x, player.y, 3, angle)  
                          
          # SA2: Stop the game when player goes out of the race track    
          if(checkOutOfBounds(player)):
              player.x = 60
              player.y = 300
              angle =0
              genomeCount = genomeCount +1
              break
          
          angle = angle + change
          
          newimage=pygame.transform.rotate(player_image,angle) 
          pygame.draw.rect(screen,(0, 255, 0), player)
          screen.blit(newimage ,player)
            
          # Sa3: Make farward = True so that car always moves forward
          forward = True
          # Sa3: change controlls the left and right turn so set it to 0 so that car moves straight if not decided by NN
          change = 0
          # Sa3: Give input to neural network and get output
          output = net.activate((player.x,player.y))
          
          # Sa3: Change value of change variable to turn left or right depending on value of output[0] and output[1]
          if output[0] > 0.65:
              change = 3
          if output[1] > 0.65:
              change = -3
              
          
          pygame.display.update()
          pygame.time.Clock().tick(30)
    
# SA1: Load NN configuration    
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  
# SA1: Create a population according to the configuration
p = neat.Population(config)
# SA1: Run the genetic algorithm for 10 generations
winner = p.run(eval_fitness,10) 
