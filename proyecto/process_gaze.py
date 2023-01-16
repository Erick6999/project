# coding=utf-8
# -*- coding: utf-8 -*-
from concurrent.futures import thread
import os
import threading
import cv2
import sys


# Clase de procesamiento ocular de UXLab
from gazeProcessor import gazeProcessor

####Posprocesamiento
## De metricas oculares
class EyeProcess:
    
    def __init__(self,directorio, guardarEn, height, width, length, header):
        # Variables de instacia
        self.directorio = directorio
        self.guardarEn = guardarEn
        self.tipo, self.video = self.getTipoAndVideo()
        self.hilo = False
        self.debugger = []
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.procesoEnEjucion = False

        #Opciones (Deben venir de la GUI)
        self.height = height
        self.width = width
        self.mode = ""
        self.length = length
        self.header = header

        #Procesador de metricas:
        self.gazeP = gazeProcessor(self.tipo, self.directorio, (self.height, self.width), self.header, self.guardarEn)

    def modeSet(self, mode):
        self.mode = mode
    
    def getMode(self):
        return self.mode

    def analisisEyetracking(self):
        if(self.video != "/" and self.tipo != ""):
            #llamada a processGaze
            x = threading.Thread(target=self.processGaze, args=(self.mode, self.length))
            x.start()
            #self.processGaze(self.mode, self.length)
        else:
            print("Esta prueba no tiene las características necesarias para procesar seguimiento ocular, revisar los archivos de la grabación")
        

    def getTipoAndVideo(self):
        try:
            archivos = os.listdir(self.directorio + '/')
            for archivo in archivos:
                if (archivo == "Video_display.mp4"):
                    video = '/' + archivo 
                if (archivo == "events.txt"): #Checar si es Tobbi, Pytribe o GP3 HD (GazePoint)
                    file=open(self.directorio + '/' + archivo,'r')
                    for line in file:
                        if("Tobii" in line):
                            tipo="Tobii"
                        elif("GP3 HD" in line):
                            tipo="GP3"
                        elif("Eyetribe" in line):
                            tipo="Eyetribe"
            return (tipo,video)
        except Exception as e:
            self.debugger.append("Error al procesar Eye tracking, revisar archivos generados durante la prueba")
            self.debugger.append(str(e))

    def processGaze(self, mode, length):
        dataFrames = len(self.gazeP.dataset)
        if(dataFrames>1):
            self.procesoEnEjucion = True
            if mode == "all":
                self.gazeP.rawVideo(dataFrames, self.gazeP.rawArray, length)
                self.gazeP.scanVideo(3)
                self.gazeP.aoiVideo(500)
                if "evento" in self.tipo:
                    self.gazeP.pupilAnalysis(self.tipo.split(".")[0])

            if mode == "raw":
                self.gazeP.rawVideo(dataFrames, self.gazeP.rawArray, length)

            if mode == "heatmap":
                self.gazeP.heatmapVideo(fps=int(1), decay=int(20))

            if mode == "scanPath":
                self.gazeP.scanVideo(length)

            if mode == "AOI":
                self.gazeP.aoiVideo(500)

            if mode == "blink":
                self.gazeP.blinkAnalysis()

            if mode == "isolate":
                self.gazeP.detectaEventos()
                rutaEventos = os.path.join(self.directorio,'Eventos')
                archivos = os.listdir(rutaEventos)

                archivosDatos = [item for item in archivos if "csv" in item]
                for i in range(len(archivosDatos)-1):
                    evento = archivosDatos[i]
                    self.processGaze(evento, rutaEventos, 1080, 1920, "all", 10, True, rutaEventos)

            if mode == "pupil":
                self.gazeP.pupilAnalysis()

            if mode == "graficar":
                self.gazeP.plotEyeData()

            self.procesoEnEjucion = False
        else:
            return False

if __name__ == '__main__':

    if(len(sys.argv) < 3 or len(sys.argv) > 3):
        print("Error, ejecute el script con 2 parametros: 1) path a recordings y 2) tipo de sensor")
    else:
        directorio = sys.argv[1]   #C:\\uxlab\\recordings\\usuario\\fecha\\numPrueba\\
                                #session + \\eyectracking_data\\FixationDataOutput.csv
                                #session + \\eyectracking_data\\BlinkDataOutput.csv
        sensor = sys.argv[2]    # GP3 o Tobii o Eyetribe

        if(sensor=="Tobii"):
            parpadeos = directorio + "\\eyectracking_data\\BlinkDataOutput.csv"
        elif(sensor=="GP3"):            
            parpadeos = directorio + "\\Gaze.csv"
        elif(sensor=="Eyetribe"):
            print("Por el momento no tenemos como procesar esa informacion intente mas a adelante o contacte a d18ce078@cenidet.tecnm.mx")
        
        ##########EyeProcess(directorio, guardarEn, height, width, length, header)
        proceso = EyeProcess(directorio, directorio, 1080, 1920, 3, True)
        proceso.processGaze("scanPath", 6)

        