import serial
import time
import os
from datetime import datetime
import yagmail
from gpiozero import CPUTemperature

#setup serial connection with Arduino Uno
ser=serial.Serial('/dev/ttyUSB0', 9600, timeout=1.0)
ser.reset_input_buffer()
time.sleep(3) #waiting for Arduino reboots

#setup e-mail for sending data
password = 'the password'
yag = yagmail.SMTP('first_email', password)
email_recipients = ['first_email','second_email']

#update data logger script currently running...
def data_logger_running():
    yag.send(to= email_recipients,
         subject='Data logger is running')
    print('data logger start')
            
#data dari Arduino masuk, data disimpan ke file (looping)
def update_data_logger(line_data):
    with open(csv_file_name, "a") as f:
        f.write(datetime.now().strftime("%H:%M:%S"))
        f.write(',')
        f.write(str(cpu.temperature))
        f.write(',')
        f.write(line_data)
        f.write('\n')

#set parameter for EC and pH
min_ec = 0.8
max_ec = 1.20
min_ph = 5.00
max_ph = 6.00
min_nt = 18.00
max_nt = 24.00 
crit_do = 4 #mg/l
min_do = 5 #mg/l

data_logger_running()

while True:
   #setup file path for data logger
   #make monthly directory if it does not exist   
    if os.path.exists("/home/pi/Data_Logger/"+"Data_"+str(datetime.now().strftime("%m_%Y")))==False:
        os.mkdir("/home/pi/Data_Logger/"+"Data_"+str(datetime.now().strftime("%m_%Y")))

    csv_file_name = "/home/pi/Data_Logger/"+"Data_"+str(datetime.now().strftime("%m_%Y"))+"/data_logger_"+ str(datetime.now().strftime("%d-%m-%Y"))+".csv"
    excel_file_name = "/home/pi/Data_Logger/"+"Data_"+str(datetime.now().strftime("%m_%Y"))+"/data_logger_"+ str(datetime.now().strftime("%d-%m-%Y"))+".xlsx"
    
    cpu = CPUTemperature()    
      
    while ser.in_waiting > 0:
        if cpu.temperature >= 50:
            yag.send(to= email_recipients,
            subject='CPU Temperature Warning '+str(time.strftime("%d-%m-%Y %H:%M:%S")),
            contents='CPU Temperature exceeds 50 degrees Celcius. Current temperature is {} degrees Celcius.'.format(str(cpu.temperature)),
            )    
                 
        line = ser.readline().decode('utf-8')
        print('byte received: ', ser.in_waiting)
        print(datetime.now().strftime("%H:%M:%S"), line, cpu.temperature)
        
        new_line = [float(i) for i in line.split(',')]
        current_ec = new_line[0]
        current_ph = new_line[1]
        current_nt = new_line[2]
        #current_do = new_line[3]
        
        optimal_ec = (current_ec < max_ec) and (current_ec > min_ec)
        optimal_ph = (current_ph < max_ph) and (current_ph > min_ph)
        optimal_nt = (current_nt < max_nt) and (current_nt > min_nt)
        #optimal_do = current_do > min_do
        
        if optimal_ec == False and optimal_ph==True and optimal_nt ==True:
            print("EC value is out of the optimal range. Current EC: {} mS/cm.".format(current_ec))
            yag.send(to= email_recipients,
                subject='EC value is out of the range',
                contents='EC value is out of the optimal range. Last EC detected: {} mS/cm.'.format(current_ec))
            print("e-mail sent")
        elif optimal_ec == True and optimal_ph==False and optimal_nt ==True:
            print("pH value is out of the optimal range. Last pH detected: {}.".format(current_ph))
            yag.send(to= email_recipients,
                subject='pH value is out of the range',
                contents='pH value is out of the optimal range. Last pH detected: {}.'.format(current_ph))
            print("e-mail sent")
        elif optimal_ec == True and optimal_ph==True and optimal_nt ==False:
            print("Nutrient temperature is out of the optimal range. Last temperature detected: {} C.".format(current_nt))
            yag.send(to= email_recipients,
                subject='Nutrient Temperature value is out of the range',
                contents='Nutrient Temperature is out of the optimal range. Last temperature detected: {} C.'.format(current_nt))
            print("e-mail sent")
        elif optimal_ec == False and optimal_ph==False and optimal_nt ==True:
            print("EC and pH is out of the optimal range. \n Last EC: %f mS/cm \n pH: %f C.".format(current_ec, current_ph))
            yag.send(to= email_recipients,
                subject='EC and pH are out of the range',
                contents= f"Last EC: {current_ec} mS/cm \n pH: {current_ph}")
            print("e-mail sent")
        elif optimal_ec == False and optimal_ph==True and optimal_nt ==False:
            print("EC and Nutrient temperature is out of the optimal range. \n Last EC: %f mS/cm \n pH: %f C.".format(current_ec, current_ph))
            yag.send(to= email_recipients,
                subject='EC and Nutrient Temperature values are out of the range',
                contents="Last EC: {} mS/cm \n Last temperature: {} C".format(current_ph, current_nt))
            print("e-mail sent")
        elif optimal_ec == True and optimal_ph==False and optimal_nt ==False:
            print("pH and Nutrient temperature is out of the optimal range.")
            yag.send(to= email_recipients,
                subject='pH and Nutrient Temperature values are out of the range',
                contents="Last pH: {} \n Last temperature: {} C".format(current_ph, current_nt))
            print("e-mail sent")
        update_data_logger(line)    
        time.sleep(3600)
