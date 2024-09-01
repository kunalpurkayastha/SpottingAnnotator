import numpy as np
import pickle
from detectron2.utils.visualizer import Visualizer
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm
import matplotlib as mpl
import matplotlib.figure as mplfigure
import random
from shapely.geometry import LineString
import math
import operator
from functools import reduce
import os

class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg, path):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.path = path
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        if self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        # voc_size includes the unknown class, which is not in self.CTABLES
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

    def draw_instance_predictions(self, predictions):
        ctrl_pnts = predictions.ctrl_points.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs
        bd_pts = np.asarray(predictions.bd)

        self.overlay_instances(ctrl_pnts, scores, recs, bd_pts)

        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def overlay_instances(self, ctrl_pnts, scores, recs, bd_pnts, alpha=0.4):
        colors = [(0,0.5,0),(0,0.75,0),(1,0,1),(0.75,0,0.75),(0.5,0,0.5),(1,0,0),(0.75,0,0),(0.5,0,0),
                (0,0,1),(0,0,0.75),(0.75,0.25,0.25),(0.75,0.5,0.5),(0,0.75,0.75),(0,0.5,0.5),(0,0.3,0.75)]
        
        # Ensure the "annotations" directory exists
        output_dir = "annotations"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):
            color = random.choice(colors)

            if bd is not None:
                # Reshape the boundary points to ensure correct dimensionality
                bd = bd.reshape(-1, 2)  # Ensuring bd is (N, 2) where N is the number of points
                
                # Sort points by y-coordinate to separate top and bottom parts
                bd_sorted = bd[np.argsort(bd[:, 1])]
                
                # Select top 50 points and bottom 50 points
                top_points = bd_sorted[:50]
                bottom_points = bd_sorted[-50:]
                
                # Extract the image name without extension
                image_name_with_ext = os.path.basename(self.path)
                image_name = os.path.splitext(image_name_with_ext)[0]
                
                # To get 8 equidistant points from top and bottom including the edge points
                if len(top_points) > 8:
                    top_indices = np.linspace(1, len(top_points) - 2, 6).astype(int)
                    top_points = np.vstack([top_points[0], top_points[top_indices], top_points[-1]])
                if len(bottom_points) > 8:
                    bottom_indices = np.linspace(1, len(bottom_points) - 2, 6).astype(int)
                    bottom_points = np.vstack([bottom_points[0], bottom_points[bottom_indices], bottom_points[-1]])

                # Reverse bottom points to maintain clockwise order
                bottom_points = bottom_points[::-1]
                
                # Combine points to form a polygon
                polygon_points = np.vstack([top_points, bottom_points])
                
                # Format polygon points and detected text
                formatted_points = ', '.join(f"{x},{y}" for x, y in polygon_points)
                detected_text = self._ctc_decode_recognition(rec)
                annotation_line = f"{formatted_points},{detected_text}\n"
                
                # Save annotation to a file
                output_file = os.path.join(output_dir, f"{image_name}.txt")
                with open(output_file, 'a') as f:
                    f.write(annotation_line)
                
                # Print the annotation (optional, can be removed if not needed)
                print(f"{formatted_points},{detected_text}")
                
                # Draw polygon (optional, can be removed if not needed)
                self.draw_polygon(polygon_points, color, alpha=alpha)

                # Draw text
                text = "{}".format(detected_text)
                lighter_color = self._change_color_brightness(color, brightness_factor=0)
                text_pos = polygon_points[0] - np.array([0,15])
                horiz_align = "left"
                font_size = self._default_font_size
                self.draw_text(
                            text,
                            text_pos,
                            color=lighter_color,
                            horizontal_alignment=horiz_align,
                            font_size=font_size,
                            draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
                        )




    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output