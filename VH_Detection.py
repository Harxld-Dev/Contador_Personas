#=========================================================================================================================
# Mejoras por   : Harold Alcantara
# Detalles      : Detección de personas en videos (Horizontal o vertical)
# Nota          : Requiere de instalar CUDA para ejecución con la tarjeta grafica, e importar con pip el txt llamado requirements 
#=========================================================================================================================

import numpy as np
import supervision as sv
import torch
import argparse

# Manejo de argumentos para obtener rutas de input y output (Videos)
parser = argparse.ArgumentParser(
                    prog='yolov5',
                    description='This program helps to detect and count people in the polygon region',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input', required=True)      # option that takes a value
parser.add_argument('-o', '--output', required=True)

args = parser.parse_args()

# Constructor para la detección de objetos,configuración de colores y obtención de info del video de entrada (Resolución)
class CountObject():

    def __init__(self, input_video_path, output_video_path) -> None:
        
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
        self.colors = sv.ColorPalette.default()

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        self.resolution_wh = self.video_info.resolution_wh
        self.is_vertical = self.resolution_wh[1] > self.resolution_wh[0]
        print("Tamanio del video - width :",self.resolution_wh[1]," Height :",self.resolution_wh[0])

        # Define poligonos en base a la orientacion del video
        if self.is_vertical:
            print("Ingreso a video vertical")
            self.polygons = [
                np.array([[0, 0], [self.resolution_wh[0], 0], [self.resolution_wh[0], self.resolution_wh[1]], [0, self.resolution_wh[1]]], np.int32)
            ]
        else:
            print("Ingreso a video horizontal")
            self.polygons = [
            np.array([
                [540,  985 ],
                [1620, 985 ],
                [2160, 1920],
                [1620, 2855],
                [540,  2855],
                [0,    1920]
            ], np.int32),
            np.array([
                [0,    1920],
                [540,  985 ],
                [0,    0   ]
            ], np.int32),
            np.array([
                [1620, 985 ],
                [2160, 1920],
                [2160,    0]
            ], np.int32),
            np.array([
                [540,  985 ],
                [0,    0   ],
                [2160, 0   ],
                [1620, 985 ]
            ], np.int32),
            np.array([
                [0,    1920],
                [0,    3840],
                [540,  2855]
            ], np.int32),
            np.array([
                [2160, 1920],
                [1620, 2855],
                [2160, 3840]
            ], np.int32),
            np.array([
                [1620, 2855],
                [540,  2855],
                [0,    3840],
                [2160, 3840]
            ], np.int32)
        ]
        #dibuja las zonas de cada poligono detectado y dibujar dentro del video
        self.zones = [
            sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=self.resolution_wh
            )
            for polygon in self.polygons
        ]

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone, 
                color=self.colors.by_idx(index), 
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(self.zones)
        ]

        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index), 
                thickness=4, 
                text_thickness=4, 
                text_scale=2
            )
            for index in range(len(self.polygons))
        ]

    def process_frame(self, frame: np.ndarray, i) -> np.ndarray:
        # Procesa cada Frame del video, detectando objetos usando modelo Yolov5 y filtra las detecciones para mantener solo personas
        
        results = self.model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            frame = zone_annotator.annotate(scene=frame)

        return frame
    
    def process_video(self):
        # Realiza la escritura del video por cada Frame 
        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame)


if __name__ == "__main__":
    # instancia para recibir los argumentos y generar el video final
    obj = CountObject(args.input, args.output)
    print(f"Creacion del video de salida {args.output}")
    obj.process_video()
    print("Se creo video Output")
