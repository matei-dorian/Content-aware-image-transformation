import random
import tkinter as tk
from copy import copy
from tkinter import filedialog

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageTk, Image
from torch import nn

import utils
from cycle_gan import Generator, predict_transform
from utils import TaskImage

WIDTH = 900
HEIGHT = 500


class BaseFrame(tk.Frame):
    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent)
        self.switch_frame = switch_frame

        # Add a background image to the frame
        self.canvas = tk.Canvas(self, width=WIDTH, height=HEIGHT, bg="red", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.place(x=0, y=0)

        # Display image
        self.bg = tk.PhotoImage(file="./assets/background.png")
        self.canvas.create_image(0, 0, image=self.bg, anchor="nw")

        # Cached info
        self.current_image = None if "cached_img" not in kwargs else kwargs["cached_img"]
        self.generator_file = None if "generator" not in kwargs else kwargs["generator"]

    def go_next(self, page, generator_file=None):
        kwargs = {}
        if self.current_image:
            kwargs["cached_img"] = self.current_image
        if generator_file:
            kwargs["generator"] = generator_file
        self.switch_frame(page, **kwargs)


class BaseFrameWithImage(BaseFrame):
    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent, switch_frame, **kwargs)
        self.switch_frame = switch_frame
        self.button_font = ("MontserratRoman Bold", 17 * -1)
        self.error_message = None
        self.drawings = []

        # Add slot for the image
        self.border = self.canvas.create_rectangle(35, 40, 450, 450, outline='#CBC7C7', width=3)
        self.image_placeholder = self.canvas.create_rectangle(40, 46, 445, 445, fill='#837F7F')
        self.placeholder_text = self.canvas.create_text(87, 215.0, anchor="nw", text="Your Image Here",
                                                        fill="#0A0A0A", font=("MontserratRoman Bold", 40 * -1))

        if self.current_image:
            self.draw_photo(self.current_image.photo_image)

    def draw_photo(self, photo_image):
        if self.current_image is not None:
            self.canvas.delete(self.current_image)
        if self.image_placeholder is not None:
            self.canvas.delete(self.image_placeholder)
        if self.placeholder_text is not None:
            self.canvas.delete(self.placeholder_text)
        if self.error_message is not None:
            self.canvas.delete(self.error_message)

        drawing_id = self.canvas.create_image(242.5, 245.0, image=photo_image)
        self.drawings.append(drawing_id)


class StartingPage(BaseFrame):
    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent, switch_frame, **kwargs)
        self.switch_frame = switch_frame

        # Add the title
        self.canvas.create_rectangle(
            0.0,
            0.0,
            900.0,
            71.0,
            fill="#CBC7C7",
            outline="")

        self.canvas.create_text(
            250,
            15.0,
            anchor="nw",
            text="Welcome to Toonify Generator ",
            fill="#0A0A0A",
            font=("MontserratRoman SemiBold", 30 * -1)
        )

        # Add instructions
        self.canvas.create_text(
            37.0,
            120.0,
            anchor="nw",
            text="Steps for creating your designs:\n\n",
            fill="#FDF4F4",
            font=("MontserratRoman Medium", 22 * -1)
        )

        text = "1. Upload an image\n"
        self.create_list_item(text, 47.0, 170.0)

        text = "2. Press Segment Image to select the\n    faces in the photo\n"
        self.create_list_item(text, 47.0, 205.0)

        text = "3. Review the result and edit the \n    selected area\n"
        self.create_list_item(text, 47.0, 260.0)

        text = "4. Press Generate and choose your \n    favorite version\n"
        self.create_list_item(text, 47.0, 315.0)

        text = "5. Download the result\n"
        self.create_list_item(text, 47.0, 370.0)

        self.canvas.create_rectangle(1, 50, 400, 499, outline='#CBC7C7', width=3)

        # Add example image
        example = Image.open("./assets/example_img.png").resize((250, 220))
        self.example = ImageTk.PhotoImage(example)
        self.canvas.create_image(523, 120, image=self.example, anchor="nw")

        # Add button
        button = tk.Button(self, text="Start Creating", font=("MontserratRoman Medium", 15 * -1),
                           command=lambda: self.go_next(SegmenterPage))
        button.place(
            x=580.0,
            y=373.0,
            width=135.0,
            height=40.0
        )

    def create_list_item(self, text, posx, posy):
        self.canvas.create_text(
            posx,
            posy,
            anchor="nw",
            text=text,
            fill="#FDF4F4",
            font=("MontserratRoman Medium", 18 * -1)
        )


class SegmenterPage(BaseFrameWithImage):

    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent, switch_frame, **kwargs)

        upload_button = tk.Button(self, text="Upload Image", font=self.button_font,
                                  command=self.upload_image)
        upload_button.place(x=555, y=150, width=135.0, height=45.0)

        next_button = tk.Button(self, text="Segment Image", font=self.button_font,
                                command=self.go_to_selection_editor)
        next_button.place(x=555, y=215, width=135.0, height=45.0)

        back_button = tk.Button(self, text="< Back", font=self.button_font,
                                command=lambda: self.go_next(StartingPage))
        back_button.place(x=555, y=280, width=135.0, height=45.0)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.current_image = TaskImage(file_path)

            for drawing in self.drawings:
                self.canvas.delete(drawing)
            self.draw_photo(self.current_image.photo_image)

    def go_to_selection_editor(self):
        if self.current_image is None:
            self.error_message = self.canvas.create_text(500, 350, anchor="nw", fill="red",
                                                         text="Error: No image selected!\n    Please upload an image.",
                                                         font=("MontserratRoman SemiBold", 22 * -1))
        else:
            self.go_next(VisualizeSegmentationPage)


class VisualizeSegmentationPage(BaseFrameWithImage):
    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent, switch_frame, **kwargs)

        if self.current_image is not None and self.current_image.seg_map is not None:
            self.draw_selected_area()
        else:
            self.perform_segmentation()

        edit_button = tk.Button(self, text="Edit SegMap", font=self.button_font,
                                command=self.edit_segmap)
        edit_button.place(x=555, y=150, width=135.0, height=45.0)

        refresh_image = Image.open("./assets/refresh.png")
        refresh_image = refresh_image.resize((32, 32))
        self.refresh_photo = ImageTk.PhotoImage(refresh_image)
        refresh_button = tk.Button(self, image=self.refresh_photo, command=self.refresh)
        refresh_button.place(x=703, y=153)

        generate_button = tk.Button(self, text="Generate Image", font=self.button_font,
                                    command=lambda: self.go_next(GeneratorPage, self.generator_version.get()))
        generate_button.place(x=555, y=215, width=135.0, height=45.0)

        options = ["Generator v1", "Generator v2", "Generator v3"]

        # Create a variable to store the selected option
        self.generator_version = tk.StringVar()

        # Set the default selected option
        self.generator_version.set(options[0])

        # Create the dropdown menu
        dropdown = tk.OptionMenu(self, self.generator_version, *options)
        dropdown.place(x=555, y=280, width=135.0, height=45.0)
        dropdown.config(font=self.button_font)

        back_button = tk.Button(self, text="< Back", font=self.button_font,
                                command=lambda: self.go_next(SegmenterPage))
        back_button.place(x=555, y=406, width=135.0, height=45.0)

    def perform_segmentation(self):
        image = self.current_image.original_image
        image = image.convert("RGB")
        device = utils.get_device()
        pixel_values = processer(image, return_tensors="pt").pixel_values.to(device)
        outputs = segmenter(pixel_values)
        logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                                           size=image.size[::-1],  # (height, width)
                                           mode='bilinear',
                                           align_corners=False)
        seg = logits.argmax(dim=1)[0]
        self.current_image.seg_map = utils.process_seg(seg)
        self.current_image.seg_backup = copy(self.current_image.seg_map)
        self.draw_selected_area()

    def draw_selected_area(self):
        result_image = self.current_image.original_image.copy()
        seg_map = np.uint8(self.current_image.seg_map) * 255
        seg_map_rgba = utils.convert_segmap_to_color(self.current_image.seg_map)

        img_rgba = result_image.convert('RGBA')
        img_rgba.paste(Image.fromarray(seg_map_rgba), box=None, mask=Image.fromarray(seg_map))

        self.current_image.segmented_image = img_rgba
        self.current_image.display_segmented = ImageTk.PhotoImage(utils.resize_image(img_rgba))
        self.draw_photo(self.current_image.display_segmented)

    def edit_segmap(self):
        image = np.array(self.current_image.original_image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        fuchsia_color = (153, 0, 153)
        seg = self.current_image.seg_map
        seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        seg[self.current_image.seg_map == 1] = fuchsia_color

        def draw_circle(x, y, color, radius=7):
            aux_mask = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
            cv.circle(aux_mask, (x, y), radius, 255, -1)
            seg[aux_mask != 0] = color

        def update_segmap(event, x, y, flags, params):
            if event == cv.EVENT_LBUTTONDOWN or flags == cv.EVENT_FLAG_LBUTTON:
                if 0 <= y <= seg.shape[0] and 0 <= x <= seg.shape[1]:
                    draw_circle(x, y, fuchsia_color, circle_radius)
            elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
                if 0 <= y <= seg.shape[0] and 0 <= x <= seg.shape[1]:
                    draw_circle(x, y, fuchsia_color, circle_radius)

            if event == cv.EVENT_RBUTTONDOWN or flags == cv.EVENT_FLAG_RBUTTON:
                if 0 <= y <= seg.shape[0] and 0 <= x <= seg.shape[1]:
                    draw_circle(x, y, (0, 0, 0), circle_radius)
            elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
                if 0 <= y <= seg.shape[0] and 0 <= x <= seg.shape[1]:
                    draw_circle(x, y, (0, 0, 0), circle_radius)

            new_image = cv.addWeighted(image, 1, seg, 0.7, 0)
            cv.imshow(params["window_name"], new_image)

        img_width, img_height = self.current_image.original_image.size
        circle_radius = max(7, int(min(img_width, img_height) * 0.01))

        displayed_image = cv.addWeighted(image, 1, seg, 0.7, 0)
        window_name = "Edit the selected area"
        cv.namedWindow(window_name, cv.WINDOW_FREERATIO)
        max_w, max_h = utils.resize_image(self.current_image.original_image, 700).size
        cv.resizeWindow(window_name, max_w, max_h)
        cv.imshow(window_name, displayed_image)
        cv.setMouseCallback(window_name, update_segmap, {"window_name": "Edit the selected area"})
        cv.waitKey(0)
        cv.destroyAllWindows()

        # revert to 2D seg map
        seg_map = np.zeros_like(seg[:, :, 0])
        mask = (seg[:, :, 2] == 153) & (seg[:, :, 1] == 0) & (seg[:, :, 0] == 153)
        seg_map[mask] = 1
        self.current_image.seg_map = seg_map
        self.draw_selected_area()

    def refresh(self):
        self.current_image.seg_map = copy(self.current_image.seg_backup)
        self.draw_selected_area()


class GeneratorPage(BaseFrameWithImage):
    def __init__(self, parent, switch_frame, **kwargs):
        super().__init__(parent, switch_frame, **kwargs)
        self.index = 0
        self.batch_size = 5

        self.first_anchor = None
        self.second_anchor = None
        self.cropped_image = None
        self.cropped_image_shape = None
        self.anime = None
        self.result = None
        self.success_text = None

        self.generator = self.load_weights()
        self.compute_anchors()
        self.generate()
        self.draw_generated_image(self.index)

        next_button = tk.Button(self, text="Prev", font=self.button_font,
                                command=lambda: self.change_index(-1))
        next_button.place(x=538.5, y=150, width=70.0, height=45.0)

        prev_button = tk.Button(self, text="Next", font=self.button_font,
                                command=lambda: self.change_index(1))
        prev_button.place(x=634.5, y=150, width=70.0, height=45.0)

        download_button = tk.Button(self, text="Download Image", font=self.button_font,
                                    command=self.save_result)
        download_button.place(x=555, y=220, width=135.0, height=45.0)

        home_button = tk.Button(self, text="Home page", font=self.button_font,
                                command=lambda: self.go_next(StartingPage))
        home_button.place(x=555, y=290, width=135.0, height=45.0)

        back_button = tk.Button(self, text="< Back", font=self.button_font,
                                command=lambda: self.go_next(VisualizeSegmentationPage))
        back_button.place(x=555, y=406, width=135.0, height=45.0)

    def load_weights(self):
        files = {
            "Generator v1": "10k_1.pth",
            "Generator v2": "6k_1.pth",
            "Generator v3": "myPortrait_1.pth",
        }
        generator = Generator(num_channels=3, num_residuals=9).to(utils.get_device())
        file_name = files[self.generator_file]
        generator.load_state(file_name)
        generator.eval()
        return generator

    def generate(self, batch_size=5):
        self.anime = []
        self.cropped_image_shape = self.cropped_image.size

        samples = []
        for i in range(batch_size):
            img = copy(self.cropped_image)
            if i > 0:
                tint_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = Image.blend(img, Image.new('RGB', img.size, tint_color), 0.08)
            samples.append(img)

        for i, img in enumerate(samples):
            tensor_image = predict_transform(img)
            batch = tensor_image.unsqueeze(0).to(utils.get_device())
            with torch.no_grad():
                result = self.generator.forward(batch) * 0.5 + 0.5
            self.anime.append(result[0])

    def compute_anchors(self):
        seg = self.current_image.seg_map
        width = seg.shape[1]
        height = seg.shape[0]
        min_x, min_y, max_x, max_y = width + 1, height + 1, -1, -1

        for i in range(width):
            for j in range(height):
                if seg[j][i] != 0:
                    if i < min_x:
                        min_x = i
                    if i > max_x:
                        max_x = i
                    if j < min_y:
                        min_y = j
                    if j > max_y:
                        max_y = j

        # # expand the bounding box
        min_x = max(0, min_x - 10)
        max_x = min(width - 1, max_x + 10)
        min_y = max(0, min_y - 10)
        max_y = min(height - 1, max_y + 10)
        self.first_anchor = (min_x, min_y)
        self.second_anchor = (max_x, max_y)
        self.cropped_image = self.current_image.original_image.crop((min_x, min_y, max_x, max_y))

    def draw_generated_image(self, index):
        mask = np.zeros_like(self.current_image.seg_map, dtype=np.uint8)
        mask[np.where(self.current_image.seg_map == 0)] = 255

        result = self.anime[index]
        result = transforms.ToPILImage()(result)
        result = result.resize(self.cropped_image_shape)
        bg_image = self.current_image.original_image
        combined_image = Image.new('RGB', bg_image.size, (255, 255, 255))
        combined_image.paste(result, self.first_anchor)
        combined_image.paste(bg_image, (0, 0), Image.fromarray(mask.astype(np.uint8)))
        self.result = ImageTk.PhotoImage(utils.resize_image(combined_image))
        self.draw_photo(self.result)

    def change_index(self, pos):
        new_pos = self.index + pos
        if new_pos < 0:
            self.index = self.batch_size - 1
        elif new_pos >= self.batch_size:
            self.index = 0
        else:
            self.index = new_pos
        self.draw_generated_image(self.index)

    def save_result(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*"))
        )

        if file_path:
            self.success_text = self.canvas.create_text(630, 100, text="Download Successful!",
                                                        font=("MontserratRoman Bold", 25 * -1), fill="#4BB543")
            self.canvas.bind("<Button-1>", lambda event: self.canvas.delete(self.success_text))
            self.canvas.focus_set()
            self.canvas.bind("<Key>", lambda event: self.canvas.delete(self.success_text))
            result = transforms.ToPILImage()(self.anime[self.index])
            result.save(file_path)


class Toonify(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Toonify Generator")
        self.geometry(f"{WIDTH}x{HEIGHT}")
        self.resizable(False, False)
        self.configure(bg="#FFFFFF")
        self.current_frame = None
        self.switch_frame(StartingPage)

    def switch_frame(self, frame_class, **kwargs):
        new_frame = frame_class(self, self.switch_frame, **kwargs)
        if self.current_frame is not None:
            self.current_frame.destroy()
        self.current_frame = new_frame
        self.current_frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    segmenter, processer = utils.load_segmenter()
    app = Toonify()
    app.mainloop()
