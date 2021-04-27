import curses
import os
import time
import RPi.GPIO as GPIO

#setting output pins for the signal
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
servo_up_down=GPIO.PWM(11,50)
GPIO.setup(12,GPIO.OUT)
servo_left_right=GPIO.PWM(12,50)
#power of the servo motors
servo_up_down.start(0)
time.sleep(0.2)
servo_left_right.start(0)
time.sleep(0.2)

screen = curses.initscr()
# turn off input echoing
curses.noecho()
# respond to keys immediately
curses.cbreak()
# map arrow keys to special values
screen.keypad(True)

# duty cycle
DC=7

try:
    while(True):
        #reading the input from keyboard
        char = screen.getch()
        #in case s button has been pressed, exit from the program
        if char == ord('s'):
            servo_left_right.ChangeDutyCycle(7.5) #stop the servo motor
            servo_up_down.ChangeDutyCycle(7) #rotate the sevo motor to 90-degree position
            time.sleep(0.05)
            servo_up_down.stop()
            servo_left_right.stop()
            GPIO.cleanup()
            break
        #in case right arrow has been pressed, display “right” on the terminal and rotate the servo in clockwise direction

        elif char == curses.KEY_RIGHT:
            screen.addstr(0, 0, 'right    ')     
            servo_left_right.ChangeDutyCycle(6.7)
            time.sleep(0.005)
            servo_left_right.ChangeDutyCycle(0)

        #in case left arrow has been pressed, display “left” on the terminal and rotate the servo in anticlockwise direction
        elif char == curses.KEY_LEFT:
            screen.addstr(0, 0, 'left    ')
            servo_left_right.ChangeDutyCycle(7.3)
            time.sleep(0.005)
            servo_left_right.ChangeDutyCycle(0)

        #in case down arrow has been pressed, display “down” on the terminal and rotate the appropriate servo in a direction that camera rotates down.
        elif char == curses.KEY_DOWN:
            screen.addstr(0, 0, 'down    ')
            if DC>2:
                DC=DC-1
                servo_up_down.ChangeDutyCycle(DC)
                time.sleep(0.01)
                servo_up_down.ChangeDutyCycle(0)

        #in case up arrow has been pressed, display “up” on the terminal and rotate the appropriate servo in a direction that camera rotates up.
        elif char == curses.KEY_UP:
            screen.addstr(0, 0, 'up    ')
            if DC<12:
                DC=DC+1
                servo_up_down.ChangeDutyCycle(DC)
                time.sleep(0.01)
                servo_up_down.ChangeDutyCycle(0)

finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
