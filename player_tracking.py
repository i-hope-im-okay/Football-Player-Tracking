import cv2
import torch
import numpy as n
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

#setting up logging for de bugging
logging.basicConfig(level = logging. INFO)
logger = logging.getLogger(__name__)

#this is the main processing class of the code
class FootballClipAnalyzer:

    def __init__(self, video_path, yolomodel = "yolov8n.pt", max_age = 30):
        logger.info("Initializing FootballClipAnalyzer...........")
        try :
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened() :
                raise ValueError(f"failed to open the video : {video_path}")
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS)) or 30
            logger.info(f"Video loaded : {self.width} x {self.height}, {self.fps} FPS")
            self.model = YOLO(yolomodel)
            logger.info("YOLO model loaded")
            self.tracker = DeepSort(max_age = max_age)
            logger.info("DeepSort initialized")
            self.frame_count = 0
            #setting up the video output
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter("output.mp4", fourcc, self.fps, (self.width, self.height))
            if not self.out.isOpened() :
                raise ValueError("Failed to initialize video writer")
            logger.info("Video writer has been initialized")
        except Exception as e :
            logger.error(f"Error initializing FootballClipAnalyzer: {e}")
            raise

    def process_frame(self, frame) :
        try :
            self.frame_count += 1
            logger.info(f"Processing frames {self.frame_count}")
            #YOLO Detection
            results = self.model(frame, classes = [0]) #Class 0 for persons
            detections = []
            for box in results[0].boxes :
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                if conf > 0.5 :
                    #Since DeepSort expects [x, y, w, h] format
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, 0))
            logger.info(f"Detected {len(detections)} players with more than 0.5 confidence")
            
            # Fallback : Drawing raw YOLO detections just in case tracking fails
            for i, (bbox, conf, _) in enumerate(detections) :
                x, y, w, h = bbox
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2) #blue boxes
                cv2.putText(frame, f"Det_{i} : {conf : .2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                logger.info(f"Drew fallbacl YOLO detction {i} at {x}, {y}, {w}, {h}")

            #DeepSort Tracking
            tracks = self.tracker.update_tracks(detections, frame = frame)
            for track in tracks :
                if not track.is_confirmed() :
                    continue
                track_id = track.track_id
                try :
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)

                    #simulating Jersey Number
                    jersey_number = n.random.randint(1, 111)

                    #drawing on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #green boxes
                    cv2.putText(frame, f"ID : {track_id} No : {jersey_number}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    logger.info(f"Annotated track ID {track_id} with jersy {jersey_number} at {x1}, {y1}, {x2}, {y2}")
                except Exception as e :
                    logger.error(f"error while track annotation for ID {track_id} : {e}")
                    continue
            
            #adding a debug frame number for verification
            cv2.putText(frame, f"Frame {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        except Exception as e :
            logger.error(f"Frame processing error : {e}")
            return frame
    
    def run(self) :
        logger.info("Starting Video Processing...........")
        try :
            while self.video.isOpened() :
                ret, frame = self.video.read()
                if not ret :
                    logger.info("End of video")
                    break
                frame = self.process_frame(frame)
                self.out.write(frame)
                logger.info(f"Wrote frame {self.frame_count} to output")
            logger.info(f"Processed {self.frame_count} frames. Saving output...........")
        except Exception as e :
            logger.error(f"Run error : {e}")
        finally :
            self.video.release()
            self.out.release()
            logger.info("Resources released")

#running the pipeline
if __name__ == "__main__" :
    analyzer = FootballClipAnalyzer("/content/football_clip.mp4")
    analyzer.run()
