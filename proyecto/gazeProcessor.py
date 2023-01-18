# coding=utf-8
# -*- coding: utf-8 -*-
# native
import csv
import cv2
import os
import math
import os.path
from turtle import circle
import numpy as np
import cv2
import emoji
import pandas as pd


# external
from datetime import datetime

class gazeProcessor:

    def __init__(self, sensor, ruta, dispsize, header = True, savePath = "C:\\uxlab\\recordings\\"):
        #Esta variable guarda el tipo de sensor que se va a procesar
        # 3GP HD o Tobii
        self.sensor = sensor
        #tamano de la pantalla 
        self.dispsize = dispsize
        #altura y anchura
        self.height =  dispsize[0] #1080
        self.width = dispsize[1] #1920
        #nombres
        self.path = ruta #Path hacia arpta base de la prueba
        self.archivoVideo = os.path.join(self.path,"Video_display.mp4")
        if(self.sensor=="GP3"):
            archivoDatos = os.path.join(self.path,"Gaze.csv")
        elif(self.sensor=="Tobii"):
            archivoDatos = os.path.join(self.path,"eyectracking_data","FixationDataOutput.csv")
            archivoDatosParpadeos = os.path.join(self.path,"eyectracking_data","BlinkDataOutput.csv")
        elif("evento" in self.sensor):
            archivoDatos = os.path.join(self.path,self.sensor)
            self.archivoVideo = os.path.join(self.path,self.sensor.split(".")[0]+".mp4")        

        self.nombreVideo   = self.archivoVideo.split(".")[0]
        self.nombreArchivo = archivoDatos.split(".")[0]
        self.savePath = savePath

        self.data = self.load_data()
        self.data.to_csv("resultados.csv", index=False)
        
        #dataSet y blinkSet son donde se guarda la info del archivo
        with open(archivoDatos, 'r') as gaze_file:
            self.dataset = gaze_file.read().splitlines()
            if header: # si tiene una fila con nombres de columnas, eliminar
                del self.dataset[0]

        if(len(self.dataset)>1):
            if(self.sensor=="GP3"):
                self.gazepointSet = self.dataset.copy()
                self.convertGP3ToGeneral()
                with open(os.path.join(self.path,"FixationDataOutput.csv"), 'r') as gaze_file:
                    self.dataset = gaze_file.read().splitlines()
                    if header: # si tiene una fila con nombres de columnas, eliminar
                        del self.dataset[0]
            if(self.sensor=="Tobii"):
                with open(archivoDatosParpadeos, 'r') as blink_file:
                    self.blinkSet = blink_file.read().splitlines()
                    if header: # si tiene una fila con nombres de columnas, eliminar
                        del self.blinkSet[0]
            
            if(len(self.dataset)>1):
                #Frames por segundo
                self.fps, self.delay, self.raw_fps = self.getRealFps()    
                #variable gausiana (para el mapa de calor) 
                self.max_gaussian = 0
                #fijaciones
                self.fixations = self.getFixations()
                #fijaciones
                self.saccades = self.getSaccades()
            else:
                print("Los datos de la prueba no son validos")
        else:
            print("No se grabo correctamente los datos de la prueba")

        #Estado del proceso
        self.status = 0
        self.processLen = 1
        self.debugMessage = ""

    # # # # #
    # METHODS   
    def load_data(self):
        pos_data = pd.read_csv("E:\Erick\Documents\Erick\Nueva\Nueva\drivers\GazeDataOutput-Positivo2.csv")
        neg_data = pd.read_csv("E:\Erick\Documents\Erick\Nueva\Nueva\drivers\GazeDataOutput-Negativo2.csv")
        resultados = pd.read_csv("E:\Erick\Documents\Erick\Nueva\Nueva\drivers\datos.csv")
        # Manipular los datos y crear el nuevo vector de datos
        data = [['X Fixation Data', 'Y Fixation Data', 'Timestamp', 'Valencia', 'Emocion', 'Activacion']]

        # Aquí puedes usar los datos de pos_data y neg_data para manipularlos y crear el nuevo vector de datos
        # ...
        pos_data = pos_data[["X Gaze Data", "Y Gaze Data", "Timestamp"]]
        pos_data.columns = ["X Fixation Data", "Y Fixation Data", "Timestamp"]
        pos_data["Valencia"] = 1
        pos_data["Activacion"] = resultados.loc[resultados["Valencia"] == 1, "Activacion"]
        pos_data["Emocion"] = "Positiva"

        neg_data = neg_data[["X Gaze Data", "Y Gaze Data", "Timestamp"]]
        neg_data.columns = ["X Fixation Data", "Y Fixation Data", "Timestamp"]
        neg_data["Valencia"] = 0
        neg_data["Activacion"] = resultados.loc[resultados["Valencia"] == 0, "Activacion"]
        neg_data["Emocion"] = "Negativa"

        data = pos_data.append(neg_data)
        data.to_csv("resultados1.csv", index=False)
        return data

    def scanVideo(self, length = 10):
        """
        Método que genera el video de sacadas a paritr de un arreglo de datos del
        seguimiento ocular
        @Parameters:
            length      -   Optional: El número de fijaciones que aparecen durante el video al mismo tiempo (int)
        @Returns
            estado del método (string = "success" o "error")
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(self.archivoVideo)
        fps = cap.get(cv2.CAP_PROP_FPS)
        videoFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(videoFrames==0):
            print("el video esta corrupto o no fue grabado apropiadamente.")
            return "error"
        out = cv2.VideoWriter(os.path.join(self.savePath, self.nombreVideo, "{}_ScanPath.mp4".format(self.nombreArchivo)), fourcc, fps, (self.width, self.height))
        fixs = self.fixations.copy()
        saccs = self.saccades.copy()
        #[fixX, fixY, duracion.total_seconds(), inicio, fin]
        #[float,float,float,                  datetime, datetime]
        duracionTotal = (fixs[-1][4]-fixs[0][3]).total_seconds()

        try:
            fix = fixs.pop(0)
            sac = saccs.pop(0)
        except:
            print("No se pudo crear el mapa de rutas debido a que no hay suficientes datos para crearlo")
            cap.release()
            return "error"

        fixList = []
        sacList = []
        
        for v in range(videoFrames):
            flag, frame = cap.read()
            copyFrame = np.zeros_like(frame, np.uint8)
            if flag==True:
                #Calcula el segundo actual del video en el frame
                #devuelve el tiempo en milisegundos del frame actual
                time = cap.get(cv2.CAP_PROP_POS_MSEC)
                #sac[0] = inicio = fix[3]

                #generate timestamp:
                fixTimestamp = ( (fix[3].hour * 3600 * 10000) + (fix[3].minute * 60 * 10000) + (fix[3].second * 10000) + int(str(fix[3].microsecond)[:4]) )  
                sacTimestamp = ( (sac[0].hour * 3600 * 10000) + (sac[0].minute * 60 * 10000) + (sac[0].second * 10000) + int(str(sac[0].microsecond)[:4]) )

                #print("fix: ", fixTimestamp, "tiempo: ", time, time >= fixTimestamp)
                #print("Tiempo: ", time, "  Fix: ", fixTimestamp)               
                if( time >= fixTimestamp ):
                    #if fix[0] > 0 and fix[1] > 0 and fix[0] <= width and fix[1] <= height:
                    if(len(fixs) > 0):
                        fixList.append( (fix[0], fix[1], fix[2]) )                        
                        if( len(fixList) > length ):
                            fixList.pop(0)
                        fix = fixs.pop(0)

                if( time >= sacTimestamp ):
                    if(len(saccs) > 0):
                        sacList.append( (sac[3], sac[4], sac[5], sac[6]) )
                        if( len(sacList) > length-1 ):
                            sacList.pop(0)
                        sac = saccs.pop(0)

                for fig in sacList:
                    cv2.line(copyFrame, (int(fig[0]), int(fig[1])), (int(fig[2]), int(fig[3])), (0,0,255), 4)
#Desde aquí eh modificado el proyecto, para agregar los colores a los circulos, estos circulos son fijaciones en ellos se iran colocando el color de acuerdo a la emocion
#El objetivo es hacer que cada circulo demuestre una emocion y se pinte de acuerdo al color de la emoción, en la siguiente linea explico los colores
#El color rojo es para la emocion positiva, azul para la negativa y gris para la neutral.
#Como te expique antes, se usaran valores para la valencia y activacion, te dare detalles enseguida
                #Define el tamaño de los circulos dependiendo de su duración
                CIRCLE_ID = 0
                for fig in fixList:
                    duracion = fig[2] #Duración de la fijación en segundos
                    base = 50
                    size = int(np.ceil(((duracion * 100) / duracionTotal))) + base #tamaño en proporcion a la duración total
                    centro = (int(fig[0]), int(fig[1]))    
                    cv2.circle(copyFrame, centro, size, (56, 190, 224), cv2.FILLED)  
                    if CIRCLE_ID == 0:
                        cv2.circle(copyFrame, centro, size, (0,0,255), cv2.FILLED)
                    elif CIRCLE_ID == 1:
                        cv2.circle(copyFrame, centro, size, (155,155,155), cv2.FILLED)
                    elif CIRCLE_ID == 2:
                            cv2.circle(copyFrame, centro, size, (255,0,0), cv2.FILLED)   
                    CIRCLE_ID = CIRCLE_ID+1      
                    #
                    cv2.addWeighted(frame, 0.4, frame, 1 - 0.4, 0)

                TEXT = 0
                for fig in fixList:
                    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
                    TEXT_SCALE = 1
                    TEXT_SCALE_SUB = 0.75
                    TEXT_THICKNESS = 2
#Este codigo es para visualizar el color y numero de fijacion que va, de momento solo eh asignado a 3 circulos como ejemplo de ejecucion pero de esto no va el proyecto, solo es mero ilustrativo.
                    text_size, _ = cv2.getTextSize(str(TEXT), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
                    text_origin = (int(fig[0] - text_size[0] / 2), int(fig[1] + text_size[1] / 2))
                    text_origin_sub = (int(fig[0] - text_size[0] / 2), int((fig[1] + text_size[1] / 2)+30))
                    cv2.putText(copyFrame, str(TEXT), text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)
                    if TEXT == 0:
                        cv2.putText(copyFrame, str(TEXT), text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)
                        cv2.putText(copyFrame, "Positivo", text_origin_sub, TEXT_FACE, TEXT_SCALE_SUB, (255, 0, 255), TEXT_THICKNESS, cv2.LINE_AA)
                    elif TEXT == 1:
                        cv2.putText(copyFrame, str(TEXT), text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)
                        cv2.putText(copyFrame, "Neutral", text_origin_sub, TEXT_FACE, TEXT_SCALE_SUB, (255, 0, 255), TEXT_THICKNESS, cv2.LINE_AA)
                    elif TEXT == 2:
                        cv2.putText(copyFrame, str(TEXT), text_origin, TEXT_FACE, TEXT_SCALE, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)
                        cv2.putText(copyFrame, "Negativo", text_origin_sub, TEXT_FACE, TEXT_SCALE_SUB, 	(255, 0, 255), TEXT_THICKNESS, cv2.LINE_AA)                     
                    TEXT = TEXT+1

                copy = frame.copy()
                alpha = 0.5
                mask = copyFrame.astype(bool)
                copy[mask] = cv2.addWeighted(frame, alpha, copyFrame, 1 - alpha, 0)[mask]
                out.write(copy)
            else:
                break
#Hasta acá es donde eh editado el proyecto. 

        # Release everything if job is finished
        cap.release()
        out.release()
        return "success"

    # # # # #
    # HELPER FUNCTIONS
    def getRealFps(self):
        """
        Returns the number of frames per second of the dataset recorded for tobii 4c            
        @return:
            fps     -   Frames per second detected from the timestamps of the dataset
            delay   -   Leftover from rounded fps
            raw_fps -   lenght of the dataset
        """
        fps = len(self.dataset) 
        #convertir en objetos de tipo dateTime
        first_timestamp = datetime.strptime(self.dataset[0].split(',')[3], '%H:%M:%S:%f')
        last_timestamp = datetime.strptime(self.dataset[fps-1].split(',')[3], '%H:%M:%S:%f')

        fps /= (last_timestamp - first_timestamp).total_seconds()
        raw_fps = fps
        delay, fps = math.modf(fps+1)
        delay = (1 / (1 - delay)) * fps

        return fps, delay, raw_fps

    # Función para generar una lista de fijaciones para el mapa de rutas
    # Completado el 06/04/2022
    def getFixations(self):
        """
        Devuelve una lista de promedios de las fijaciones contenidas en los
        archivos de datos, donde los eventos omienzan en begin hasta end
        @return:
            fixations (list = [fixX, fixY, duracion.total_seconds(), inicio, fin])
        """
        #Definicion de variables auxiliares
        lista = []
        xPoints = []
        yPoints = []
        fixations = []
        inicio = datetime.strptime("00:00:00:0000", '%H:%M:%S:%f')
        fin = datetime.strptime("00:00:00:0000", '%H:%M:%S:%f')

        i = 1
        size = len(self.dataset)
        for line in self.dataset:  
            fix = line.split(',')
            evento = fix[0]
            tiempo = datetime.strptime(fix[3], '%H:%M:%S:%f')
            #codigo para trabajar las fijaciones
            #Si la lectura es el fin de una fijacón o la ultima entonces se calcula la media de la fijacion
            # y todas las lecturas de tipo data (Begin + data + end)
            if (evento == "End" or i == size): # calcular el punto central de la fijación y cuanto tiempo duró
                lista.append(line)
                fin = tiempo
                
                #calcular la duración de la fijación y el punto central
                duracion = (fin - inicio).total_seconds()
                
                #recorrer la lista auxiliar
                for row in lista:
                    data = row.split(',')
                    xPoints.append(float(data[1]))
                    yPoints.append(float(data[2]))
                
                fixX = np.median(xPoints)
                fixY = np.median(yPoints)

                # agregar al arreglo de fijaciones la nueva fijacion
                fixations.append([fixX, fixY, duracion, inicio, fin])

                #reiniciar las listas
                del lista[:]
                del xPoints[:]
                del yPoints[:]

            else: # Juntar todas las entradas en la lista auxiliar (tipo data y begin)
                lista.append(line)
                #sobreescribe la variable inicio si es una entrada "Begin"
                if (evento == "Begin"):
                    inicio = tiempo
            i += 1

        return fixations

    # Función para generar una lista de sacadas para el mapa de rutas
    def getSaccades(self):
        fixations = self.fixations.copy()
        #Definicion de variables auxiliares
        firstFix = []
        secondFix = []    
        saccades = []
        inicio = datetime.now()
        fin = datetime.now()
        for i in range(len(fixations)):        
            #codigo para trabajar las sacadas [fixX, fixY, duracion.total_seconds(), inicio, fin]
            #condicion, es el ultimo elemento de la lista?
            if (i != range(len(fixations))[-1]):
                firstFix = fixations[i]
                secondFix = fixations[i+1]
                #calcular la duración de la sacada
                inicio = firstFix[4]
                fin = secondFix[3]
                duracion = (fin - inicio)

                #fixations.append([fixX, fixY, duracion.total_seconds(), inicio, fin])   <-- refrencia de la estructura de fijacion
                saccades.append([inicio, fin, duracion, firstFix[0], firstFix[1], secondFix[0], secondFix[1]]) # <-- estructura de sacadas
            #else: #Ya no hay mas puntos, error i+1
                #print algo
        return saccades

    def convertGP3ToGeneral(self):
        # El formato general incluye: 
        # Event (Begin, Data, End), 
        # X Fixation Data (0-1920), 
        # Y Fixation Data (0-1080), 
        # Timestamp (00:00:00:0000) startTime = datetime.strptime(r[3], '%H:%M:%S:%f')
        data = self.dataset.copy()
        conversion = [] #Aqui iran las lineas del archivo CSV (lista de listas)
        lastID = 0 #Variable para controlar los eventos de fijación
        for i in range(len(data)):
            columna = data[i].split(',')         
            evento="Begin"
            
            if (columna[10] == "1"): #Si el valor de la fijacion es valido

                #Get evento                    
                actualID = columna[9]
                if (lastID == 0):
                    evento = "Begin"
                    lastID=actualID
                elif (lastID == actualID or lastID==1):
                    evento = "Data"
                    lastID=actualID
                elif (lastID != actualID):
                    arreglo = conversion.pop()
                    arreglo[0]="End"
                    conversion.append(arreglo)
                    evento="Begin"
                    lastID = 1
                if ((i+1)==len(data)):
                    evento="End"
                    lastID=0
                
                #Get Fixation points
                # FPOGX	(X- and Y-coordinates of the fixation POG, as a fraction of the screen size.)
                # FPOGY	((0,0) is top left, (0.5,0.5) is the screen center, and (1.0,1.0) is bottom right.)
                x = float(columna[5]) * self.width
                y = float(columna[6]) * self.height

                #Get timeStamp formateada
                tiempo = columna[0] #tiempo en segundos
                temp = self.cambiaTiempo(tiempo)

                validLeft = columna[24]
                validRight = columna[29]
                leftPupilSize = columna[22]
                leftPupilScale = columna[23]
                rightPupilSize = columna[27]
                rightPupilScale = columna[28]
                validLeftGaze = columna[13]
                validRightGaze = columna[16]
                leftXGaze = columna[11]
                leftYGaze = columna[12]
                rightXGaze = columna[14]
                rightYGaze = columna[15]
                validFixation = columna[10]
                
                conversion.append([
                    evento,             #0
                    x,                  #1
                    y,                  #2
                    temp,               #3
                    validLeft,          #4
                    validRight,         #5
                    leftPupilSize,      #6
                    leftPupilScale,     #7
                    rightPupilSize,     #8
                    rightPupilScale,    #9
                    validLeftGaze,      #10
                    validRightGaze,     #11
                    leftXGaze,          #12
                    leftYGaze,          #13
                    rightXGaze,         #14
                    rightYGaze,         #15
                    validFixation       #16
                    ])
        
        with open(os.path.join(self.path,"FixationDataOutput.csv"), 'w', newline='', encoding='utf-8') as archivo:
            wr = csv.writer(archivo)
            wr.writerow([
                "Event",
                "X Fixation Data",
                "Y Fixation Data",
                "Timestamp",
                "Valid Left Pupil",
                "Valid Right Pupil",
                "Left Pupil Size",
                "Left Pupil Scale",
                "Right Pupil Size",
                "Right Pupil Scale",
                "Valid Left Gaze",
                "Valid Right Gaze",
                "Left X Gaze",
                "Left Y Gaze",
                "Right X Gaze",
                "Right Y Gaze",
                "Valid Fixation"
                ])
            for line in conversion:
                wr.writerow(line)

    def cambiaTiempo(self, time):
        #funcion que cambia segundos a timestamp hh:mm:ss:ffff: 0.0007
        resultado = ""
        temps = float(time)

        horas = math.floor(temps / 3600)
        resto = temps % 3600
        minutos = math.floor(resto / 60)
        resto = resto % 60 #
        segundos = math.floor(resto)
        resto = resto % 1
        milisegundos = resto * 1000 #0.7

        if(horas <= 9):
            horas = "0"+str(horas)
        else:
            horas = str(horas)
        if(minutos <= 9):
            minutos = "0"+str(minutos)
        else:
            minutos = str(minutos)
        if(segundos <= 9):
            segundos = "0"+str(segundos)
        else:
            segundos = str(segundos)
        
        if(milisegundos < 10):
            milisegundos = "000"+str(milisegundos)
        elif(milisegundos < 100):
            milisegundos = "00"+str(milisegundos)
        elif(milisegundos < 1000):
            milisegundos = "0"+str(milisegundos)
        else:
            milisegundos = str(milisegundos)

        resultado = horas + ":" + minutos + ":" + segundos + ":" + milisegundos[:4]

        return resultado

    def normalizaTiempo(self, time, base):
        dateTime = datetime.strptime(time, '%H:%M:%S:%f')
        base     = datetime.strptime(base, '%H:%M:%S:%f')
        return self.cambiaTiempo((dateTime - base).total_seconds())