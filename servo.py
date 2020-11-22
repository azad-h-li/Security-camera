import curses
import os
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
servo_up_down=GPIO.PWM(11,50)
GPIO.setup(12,GPIO.OUT)
servo_left_right=GPIO.PWM(12,50)

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

DC1=7

try:
    while(True):
        char = screen.getch()
        if char == ord('0'):
            servo_left_right.ChangeDutyCycle(7)
            servo_up_down.ChangeDutyCycle(7)
            time.sleep(0.05)
            servo_up_down.stop()
            servo_left_right.stop()
            GPIO.cleanup()
            break
        elif char == curses.KEY_RIGHT:
            screen.addstr(0, 0, 'right    ')     
            servo_left_right.ChangeDutyCycle(6.8)
            time.sleep(0.01)
            servo_left_right.ChangeDutyCycle(0)
        elif char == curses.KEY_LEFT:
            screen.addstr(0, 0, 'left    ')
            servo_left_right.ChangeDutyCycle(7.3)
            time.sleep(0.01)
            servo_left_right.ChangeDutyCycle(0)
        elif char == curses.KEY_DOWN:
            screen.addstr(0, 0, 'down    ')
            if DC>2:
                DC=DC-1
                servo_up_down.ChangeDutyCycle(DC)
                time.sleep(0.02)
                servo_up_down.ChangeDutyCycle(0)
        elif char == curses.KEY_UP:
            screen.addstr(0, 0, 'up    ')
            if DC<12:
                DC=DC+1
                servo_up_down.ChangeDutyCycle(DC)
                time.sleep(0.02)
                servo_up_down.ChangeDutyCycle(0)
finally:
    # shut down cleanly
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()