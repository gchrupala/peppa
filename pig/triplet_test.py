import pygame
import pig.data as D
import moviepy.editor as E
import random
#random.seed(123)


white = (255, 255, 255)
black = (0, 0, 0)
size   = (720, 400)
middle = (720//3, 400//3)


def main():
    records = []
    data = D.PeppaPigIterableDataset(split='val', fragment_type='dialog', window=0,
                                     transform=None, duration=None, triplet=False)
    data._prepare_triplets()
    pygame.init()
    screen = pygame.display.set_mode(size)
    font = pygame.font.SysFont("monospace", 56)
    screen.fill(white)
    pygame.display.flip()
    DIGIT_SIZE=(314//2, 512//2)
    ear = pygame.image.load('ear.png')
    ear = pygame.transform.scale(ear, (300, 300))
    
    one = pygame.image.load('one.png')
    one = pygame.transform.scale(one, DIGIT_SIZE)
    
    two = pygame.image.load('two.png')
    two = pygame.transform.scale(two, DIGIT_SIZE)
    
    que = pygame.image.load('que.png')
    que = pygame.transform.scale(que, DIGIT_SIZE)
    
    for i, item in enumerate(data.raw_triplets(shuffle=True)):
        if i > 5:
            break
        else:
            
            # screen.fill(white)
            # screen.blit(ear, (200, 50))
            # pygame.display.flip()
            # item.anchor.preview()
            # pygame.time.wait(2000)
            
            item.negative.audio = item.positive.audio
            clips = (item.positive, item.negative)
            index = random.randint(0, 1)
            
            
            screen.fill(white)
            screen.blit(one, (300, 50))
            pygame.display.flip()
            pygame.time.wait(1000)
            
            clips[index].resize(width=720).preview()
            screen.fill(white)
            pygame.display.flip()
            pygame.time.wait(2000)

            
            #pygame.time.wait(2000)
            
            screen.fill(white)
            screen.blit(two, (300, 50))
            pygame.display.flip()
            pygame.time.wait(1000)
            clips[1-index].resize(width=720).preview()
            screen.fill(white)
            pygame.display.flip()

            screen.fill(white)
            screen.blit(que, (300, 50))
            pygame.display.flip()
            choice = get_choice()
            records.append(dict(item=i, index=index, correct=index+1, choice=choice))
            screen.fill(black)
            pygame.display.flip()
            pygame.time.wait(500)

    print(f"Correct {sum(x['correct'] == x['choice'] for x in records)} out of {len(records)}")
    print(records)


            
            
def get_choice():
    pygame.event.clear()
    while True:
        event = pygame.event.wait()
        #print(event)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
            return 1
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_2:
            return 2
            
