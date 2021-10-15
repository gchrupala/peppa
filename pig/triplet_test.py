import pygame
import pig.data as D
import moviepy.editor as E
import random
random.seed(123)


white = (255, 255, 255)
black = (0, 0, 0)
size   = (720, 400)
middle = (720//3, 400//3)


def main():

    data = D.PeppaPigIterableDataset(split='val', fragment_type='dialog', window=0,
                                     transform=None, duration=None, triplet=False)
    data._prepare_triplets()
    pygame.init()
    screen = pygame.display.set_mode(size)
    font = pygame.font.SysFont("monospace", 56)
    screen.fill(black)
    pygame.display.flip()
    ear = pygame.image.load('ear.png')
    ear = pygame.transform.scale(ear, (300, 300))
    for i, item in enumerate(data.raw_triplets()):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        if i > 5:
            break
        else:
        
            screen.fill(white)
            screen.blit(ear, (200, 50))
            pygame.display.flip()
            item.anchor.preview()
            pygame.time.wait(2000)
            
            
            clips = (item.positive, item.negative)
            index = random.randint(0, 1)
            
            
            screen.fill(black)
            screen.blit(font.render("1", False, white), middle)
            pygame.display.flip()
            pygame.time.wait(1000)
            clips[index].resize(width=720).preview(audio=False)
            screen.fill(black)
            pygame.display.flip()
            pygame.time.wait(2000)

            
            #pygame.time.wait(2000)
            
            screen.fill(black)
            screen.blit(font.render("2", False, white), middle)
            pygame.display.flip()
            pygame.time.wait(1000)
            clips[1-index].resize(width=720).preview(audio=False)
            screen.fill(black)
            pygame.display.flip()
            pygame.time.wait(3000)
            
            #screen.fill((0, 0, 0))
            #pygame.time.wait(4000)
            
    
    
    
