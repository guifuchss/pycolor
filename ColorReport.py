import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import cv2
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import colorsys
import os
import sys
import subprocess
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from PIL import Image, ImageTk, ImageDraw, ImageFont

class ColorPaletteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ColorReport")
        
        # Center the window on the screen
        window_width = 1200    # Define the desired width of the window
        window_height = 900  # Define the desired height of the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        self.root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
        
        # Add an icon to the left side of the title
        #icon = tk.PhotoImage(file='/Users/guilhermefuchs/PythonProjects/ColorReport/ID/icon.png')  # Make sure to provide the correct path to your icon file
        #self.root.iconphoto(False, icon)
        
        # Variables
        self.image_path = None
        self.num_bars = tk.IntVar(value=7)
        self.palette_height = tk.IntVar(value=200)
        self.palette_type = tk.StringVar(value="Predominance")
        self.sorting_option = tk.StringVar(value="Predominance (max to min)")
        self.display_image = None
        self.kmeans = None
        self.palette = None
        self.manual_palette_colors = []
        self.is_manual_palette = False
        self.original_image_path = None
        self.version_image_paths = []
        self.current_version_index = -1
        
        # Create UI elements
        self.create_widgets()
        self.setup_event_handlers()

    def create_widgets(self):
        frame_left = ttk.Frame(self.root)
        frame_left.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        frame_right = ttk.Frame(self.root)
        frame_right.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Adjust the size of the frame to avoid overlapping
        frame_left.config(width=400)

        ttk.Label(frame_left, text="Upload an Image:").grid(row=0, column=0, pady=10, sticky="w")
        ttk.Button(frame_left, text="Browse", command=self.upload_image).grid(row=0, column=1, pady=10, sticky="w")

        ttk.Button(frame_left, text="Create Palette", command=self.create_palette).grid(row=8, column=0, columnspan=2, pady=10)

        ttk.Label(frame_left, text="Types of Palette:").grid(row=5, column=0, pady=10, sticky="w")
        palette_types = ["Predominance", "Saturation", "Brightness", "Darkness"]
        self.palette_type_combo = ttk.Combobox(frame_left, textvariable=self.palette_type, values=palette_types)
        self.palette_type_combo.grid(row=5, column=1, pady=10, sticky="w")

        ttk.Label(frame_left, text="Number of Bars:").grid(row=1, column=0, pady=10, sticky="w")
        self.num_bars_entry = ttk.Entry(frame_left, textvariable=self.num_bars)
        self.num_bars_entry.grid(row=1, column=1, pady=10, sticky="w")

        ttk.Label(frame_left, text="Palette Height:").grid(row=4, column=0, pady=10, sticky="w")
        self.palette_height_slider = ttk.Scale(frame_left, from_=0, to_=1000, orient="horizontal", variable=self.palette_height)
        self.palette_height_slider.set(200)  # Default value
        self.palette_height_slider.grid(row=4, column=1, pady=10, sticky="w")

        self.palette_height_entry = ttk.Entry(frame_left, textvariable=self.palette_height, width=5)
        self.palette_height_entry.grid(row=4, column=2, pady=10, sticky="w")

        ttk.Label(frame_left, text="Sorting Option:").grid(row=7, column=0, pady=10, sticky="w")
        sorting_options = [
            "Predominance (max to min)",
            "Predominance (min to max)",
            "Saturation (most to least)",
            "Saturation (least to most)",
            "Brightness (most to least)",
            "Darkness (most to least)",
        ]
        self.sorting_option_combo = ttk.Combobox(frame_left, textvariable=self.sorting_option, values=sorting_options)
        self.sorting_option_combo.grid(row=7, column=1, pady=10, sticky="w")

        # Add checkbox for RGB values display
        self.show_rgb_values = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_left, text="Show RGB Values", variable=self.show_rgb_values).grid(row=8, column=0, columnspan=2, pady=10)

        # Order of buttons as requested
        ttk.Button(frame_left, text="Create Automatic Palette", command=self.create_palette).grid(row=9, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Manual Palette", command=self.activate_manual_palette).grid(row=10, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Save Image Palette", command=self.save_image_palette).grid(row=11, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Generate Color Wheel", command=self.generate_color_wheel).grid(row=12, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Create Mosaic", command=self.create_mosaic).grid(row=13, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Frame Report", command=self.frame_report).grid(row=14, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Full Report", command=self.run_full_report_script).grid(row=15, column=0, columnspan=2, pady=10)
        ttk.Button(frame_left, text="Image Wipe", command=self.run_image_wipe).grid(row=16, column=0, columnspan=2, pady=10)

        # Image display area
        self.image_label = ttk.Label(frame_right)
        self.image_label.grid(row=1, column=0, padx=10, pady=10)

        # Custom button style
        style = ttk.Style()
        style.configure("Custom.TButton", foreground="white", background="blue")


    def setup_event_handlers(self):
        self.num_bars.trace_add('write', lambda *args: self.safe_update(self.create_palette))
        self.palette_height_slider.bind("<Motion>", self.update_preview)  # Use bind instead of trace_add for Scale widget
        self.palette_height_entry.bind("<Return>", self.update_preview)  # Update on Enter key press
        self.sorting_option.trace_add('write', lambda *args: self.safe_update(self.create_palette))
        self.show_rgb_values.trace_add('write', lambda *args: self.safe_update(self.update_preview))

    def safe_update(self, func):
        try:
            func()
        except tk.TclError:
            pass

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.kmeans = None  # Reset K-means result when a new image is uploaded
            self.display_image_preview()

    def create_palette(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        try:
            num_bars = int(self.num_bars.get())
        except tk.TclError:
            num_bars = 5  # Default value if empty

        try:
            palette_height = int(self.palette_height.get())
        except tk.TclError:
            palette_height = 50  # Default value if empty

        palette_type = self.palette_type.get()
        sorting_option = self.sorting_option.get()

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if palette_type == "Darkness":
            pixels = self.extract_dark_colors(image)
        elif palette_type == "Saturation":
            pixels = self.extract_saturated_colors(image)
        elif palette_type == "Brightness":
            pixels = self.extract_bright_colors(image)
        else:
            pixels = image.reshape(-1, 3)

        # Check if pixels array is empty
        if pixels.size == 0:
            messagebox.showerror("No Colors Found", "No colors found for the selected palette type and threshold.")
            return

        self.kmeans = KMeans(n_clusters=num_bars, n_init=10)
        self.kmeans.fit(pixels)
        colors = self.kmeans.cluster_centers_

        if palette_type == "Saturation":
            colors = self.sort_colors_by_saturation(colors, reverse=True)
        elif palette_type == "Brightness":
            colors = self.sort_colors_by_brightness(colors, reverse=True)
        elif palette_type == "Darkness":
            colors = self.sort_colors_by_darkness(colors)

        self.palette = self.create_palette_image(colors, image.shape[1], palette_height)
        self.is_manual_palette = False  # Mark that we are using the automatic palette
        self.update_preview()

    def display_image_preview(self):
        if not self.image_path:
            return

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(image)

        combined_image_pil = self.display_image.resize((image.shape[1] // 3, image.shape[0] // 3), Image.LANCZOS)
        combined_image_tk = ImageTk.PhotoImage(combined_image_pil)
        self.image_label.configure(image=combined_image_tk)
        self.image_label.image = combined_image_tk

    def update_preview(self, *args, palette_type='automatic'):
        if not self.image_path:
            return

        try:
            num_bars = int(self.num_bars.get())
        except tk.TclError:
            num_bars = 5  # Default value if empty

        try:
            palette_height = int(self.palette_height.get())
        except tk.TclError:
            palette_height = 50  # Default value if empty

        sorting_option = self.sorting_option.get()

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_manual_palette:
            colors = self.manual_palette_colors
        else:
            if self.kmeans is None or self.num_bars.get() != num_bars or self.sorting_option.get() != sorting_option:
                self.create_palette()
            else:
                self.palette = self.create_palette_from_kmeans(image.shape[1], palette_height)
            colors = self.kmeans.cluster_centers_

        if sorting_option == "Saturation (most to least)":
            colors = self.sort_colors_by_saturation(colors, reverse=True)
        elif sorting_option == "Brightness (most to least)":
            colors = self.sort_colors_by_brightness(colors, reverse=True)
        elif sorting_option == "Darkness (most to least)":
            colors = self.sort_colors_by_darkness(colors)

        self.palette = self.create_palette_image(colors, image.shape[1], palette_height)

        combined_image = self.combine_image_and_palette(image, np.array(self.palette))
        combined_image_pil = Image.fromarray(combined_image).resize((combined_image.shape[1] // 3, combined_image.shape[0] // 3), Image.LANCZOS)

        combined_image_tk = ImageTk.PhotoImage(combined_image_pil)
        self.image_label.configure(image=combined_image_tk)
        self.image_label.image = combined_image_tk

    def activate_manual_palette(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        self.manual_palette_colors = []
        self.is_manual_palette = True
        self.image_label.bind("<Button-1>", self.get_color_manual)
        self.update_preview(palette_type='manual')

    def get_color_manual(self, event):
        x = event.x * 3  # Ajuste a escala conforme necessário
        y = event.y * 3  # Ajuste a escala conforme necessário
        rgb = self.display_image.getpixel((x, y))
        self.manual_palette_colors.append(rgb)
        self.update_preview(palette_type='manual')

    @staticmethod
    def rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb

    def save_image_palette(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        try:
            num_bars = int(self.num_bars.get())
        except tk.TclError:
            num_bars = 5  # Default value if empty

        try:
            palette_height = int(self.palette_height.get())
        except tk.TclError:
            palette_height = 50  # Default value if empty

        sorting_option = self.sorting_option.get()

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_manual_palette:
            colors = self.manual_palette_colors
        else:
            if self.kmeans is None or self.num_bars.get() != num_bars or self.sorting_option.get() != sorting_option:
                self.create_palette()
            else:
                self.palette = self.create_palette_from_kmeans(image.shape[1], palette_height)
            colors = self.kmeans.cluster_centers_

        self.palette = self.create_palette_image(colors, image.shape[1], palette_height)
        combined_image = self.combine_image_and_palette(image, np.array(self.palette))
        combined_image_pil = Image.fromarray(combined_image)

        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")], title="Save Image Palette")
        if save_path:
            combined_image_pil.save(save_path, format="JPEG", dpi=(300, 300))
            messagebox.showinfo("Image Palette Saved", f"Image Palette saved to {save_path}")

    def generate_color_wheel(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        if self.palette is None:
            messagebox.showwarning("No Palette", "Please create a palette first.")
            return

        color_wheel_image = self.create_color_wheel_image(self.palette, 800)
        combined_image = self.combine_wheel_and_palette(color_wheel_image, self.palette, (1920, 1080))
        combined_image.show()

        save_path_2 = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")], title="Save Color Wheel + Palette")
        if save_path_2:
            combined_image.save(save_path_2, format="JPEG", dpi=(300, 300))
            messagebox.showinfo("Color Wheel + Palette Saved", f"Color Wheel + Palette saved to {save_path_2}")

    def create_palette_from_kmeans(self, image_width, palette_height):
        bar_height = int(palette_height)
        palette = np.zeros([bar_height, image_width, 3], dtype=np.uint8)

        counter = np.bincount(self.kmeans.labels_)
        count_labels = list(range(len(counter)))
        total_pixels = sum(counter)
        widths = [round(image_width * counter[i] / total_pixels) for i in count_labels]

        current_x = 0
        for i in count_labels:
            next_x = current_x + widths[i]
            palette[:, current_x:next_x, :] = self.kmeans.cluster_centers_[i].astype(int)
            current_x = next_x

        return palette

    def calculate_brightness(self, color):
        if isinstance(color, tuple):
            color = np.array(color)
        r, g, b = color / 255.0
        return (r + g + b) / 3

    def calculate_saturation(self, color):
        if isinstance(color, tuple):
            color = np.array(color)
        r, g, b = color / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return s

    def calculate_hue(self, color):
        if isinstance(color, tuple):
            color = np.array(color)
        r, g, b = color / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h

    def sort_colors_by_brightness(self, colors, reverse=False):
        brightness = [self.calculate_brightness(color) for color in colors]
        sorted_colors = [color for _, color in sorted(zip(brightness, colors), reverse=reverse)]
        return np.array(sorted_colors)

    def sort_colors_by_darkness(self, colors):
        return self.sort_colors_by_brightness(colors, reverse=True)

    def sort_colors_by_saturation(self, colors, reverse=False):
        hsv_colors = [colorsys.rgb_to_hsv(*(np.array(color) / 255.0)) for color in colors]
        sorted_colors = sorted(hsv_colors, key=lambda x: x[1], reverse=reverse)
        sorted_colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*color)) for color in sorted_colors]
        return np.array(sorted_colors)
    
    def extract_dark_colors(self, image, threshold=50):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Create a mask to filter out bright areas
        mask = gray_image < threshold
        # Apply the mask to get dark colors
        dark_colors = image[mask]
        return dark_colors
    
    def extract_saturated_colors(self, image, threshold=0.5):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv_image[:, :, 1] / 255.0  # Normalize saturation to range [0, 1]
        mask = saturation > threshold
        saturated_colors = image[mask]
        return saturated_colors
    
    def extract_bright_colors(self, image, threshold=200):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image > threshold
        bright_colors = image[mask]
        return bright_colors

    def calculate_average_brightness(self, palette):
        brightness_values = [self.calculate_brightness(color) for color in palette[0].reshape(-1, 3)]
        return np.mean(brightness_values)

    def calculate_average_saturation(self, palette):
        saturation_values = [self.calculate_saturation(color) for color in palette[0].reshape(-1, 3)]
        return np.mean(saturation_values)

    def create_palette_image(self, colors, image_width, palette_height):
        bar_height = int(palette_height)
        palette = np.zeros([bar_height, image_width, 3], dtype=np.uint8)

        if self.is_manual_palette:
            num_colors = len(colors)
            if num_colors == 0:
                bar_widths = []
            else:
                bar_widths = [image_width // num_colors] * num_colors
        else:
            counter = np.bincount(self.kmeans.labels_)
            total_pixels = sum(counter)
            count_labels = list(range(len(counter)))
            bar_widths = [round(image_width * counter[i] / total_pixels) for i in count_labels]

        current_x = 0
        for i, color in enumerate(colors):
            next_x = current_x + bar_widths[i]
            if isinstance(color, tuple):
                color = np.array(color)
            palette[:, current_x:next_x, :] = color.astype(int)
            
            # Conditionally draw RGB values on the bar
            if self.show_rgb_values.get():
                rgb_text = f"RGB({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})"
                pil_palette = Image.fromarray(palette)
                draw = ImageDraw.Draw(pil_palette)
                
                # Use a smaller font to ensure it fits
                font = ImageFont.load_default()
                
                # Calculate the position for the text using textbbox
                text_bbox = draw.textbbox((0, 0), rgb_text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                
                # Ensure the text is centered in the bar
                text_x = current_x + (bar_widths[i] - text_width) // 2
                text_y = (bar_height - text_height) // 2
                
                # Draw the text on the palette
                draw.text((text_x, text_y), rgb_text, fill="white", font=font)
                palette = np.array(pil_palette)

            current_x = next_x

        return palette
    


    def frame_report(self):
        def show_skip_dialog(title, prompts):
            root = tk.Tk()
            root.withdraw()

            top = tk.Toplevel(root)
            top.title(title)

            results = []

            def submit():
                names = [entry.get() for entry in name_entries]
                results.extend(names)
                top.destroy()

            name_frames = []
            name_entries = []

            for prompt in prompts:
                frame = tk.Frame(top)
                label = tk.Label(frame, text=prompt)
                label.pack(side=tk.LEFT, padx=10)

                entry = tk.Entry(frame)
                entry.pack(side=tk.LEFT)
                name_entries.append(entry)

                frame.pack(pady=10)
                name_frames.append(frame)

            button_frame = tk.Frame(top)

            submit_button = tk.Button(button_frame, text='Submit', command=submit)
            submit_button.pack(side=tk.LEFT)

            button_frame.pack(pady=10)

            top.wait_window()

            return results

        def cluster_image(image, n_clusters):
            image = image[(image > [10, 10, 10]).all(axis=2)]
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            labels = kmeans.fit_predict(pixels)

            counter = np.bincount(labels)
            total_pixels = sum(counter)
            count_labels = list(range(len(counter)))
            count_labels.sort(key=lambda x: counter[x])

            widths = [round(500 * count / total_pixels) for count in counter]

            while sum(widths) < 500:
                widths[widths.index(min(widths))] += 1
            while sum(widths) > 500:
                widths[widths.index(max(widths))] -= 1

            palette = np.zeros([50, 500, 3], dtype=np.uint8)

            current_x = 0
            for i in range(n_clusters):
                next_x = current_x + widths[count_labels[i]]
                palette[:, current_x:next_x, :] = kmeans.cluster_centers_[count_labels[i]].astype(int)
                current_x = next_x

            return palette

        def plot_image(image, gs, title):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax = plt.subplot(gs)
            ax.imshow(image_rgb)
            ax.set_title(title, color='white', fontsize=20)
            ax.axis('off')

        def create_palette(img, n_colors):
            data = np.array(img).reshape((-1, 3))
            kmeans = KMeans(n_clusters=n_colors, n_init=10)
            kmeans.fit(data)
            colors = kmeans.cluster_centers_
            colors = colors.round(0).astype(int)
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_colors = dict(zip(unique, counts))
            sorted_colors = sorted(dominant_colors.items(), key=lambda item: item[1], reverse=True)
            plot_colors = [colors[i] for i, count in sorted_colors]
            counts = [count for i, count in sorted_colors]
            counts = np.array(counts) / sum(counts)
            return plot_colors, counts

        def extract_dominant_color(image_segment):
            data = np.array(image_segment).reshape((-1, 3))
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(data)
            dominant_color = kmeans.cluster_centers_[0]
            dominant_color = dominant_color.round(0).astype(int)
            return dominant_color / 255

        def plot_colors(colors, counts):
            fig, ax = plt.subplots(1, 1, figsize=(10.40, 2), dpi=300, facecolor='#2E2E2E', edgecolor='k')

        def process_image(image_path, save_path):
            width = 8.27  # inches
            height = 11.69  # inches

            img = Image.open(image_path)
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            img_rgb = img.convert('RGB')

            r_avg, g_avg, b_avg = [], [], []

            for column in range(img_rgb.size[0]):
                r, g, b = 0, 0, 0
                for pixel in img_rgb.crop((column, 0, column + 1, img_rgb.size[1])).getdata():
                    r += pixel[0]
                    g += pixel[1]
                    b += pixel[2]
                r_avg.append(r / img_rgb.size[1])
                g_avg.append(g / img_rgb.size[1])
                b_avg.append(b / img_rgb.size[1])

            if not r_avg or not g_avg or not b_avg:
                print(f"Empty channel data for {image_path}. Skipping this image.")
                return

            max_val = max(max(r_avg), max(g_avg), max(b_avg))
            r_avg = [x / max_val * 100 for x in r_avg]
            g_avg = [x / max_val * 100 for x in g_avg]
            b_avg = [x / max_val * 100 for x in b_avg]

            x = np.arange(len(r_avg))

            img_array = np.array(img)

            hsv = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2HSV)
            value = hsv[:, :, 2]
            saturation = hsv[:, :, 1]

            shadow_threshold = 30
            highlight_threshold = 80

            shadows_mask = value < shadow_threshold
            midtones_mask = (value >= shadow_threshold) & (value < highlight_threshold)
            highlights_mask = value >= highlight_threshold

            average_saturation_shadows = np.mean(saturation[shadows_mask])
            average_saturation_midtones = np.mean(saturation[midtones_mask])
            average_saturation_highlights = np.mean(saturation[highlights_mask])

            average_saturations = [average_saturation_shadows, average_saturation_midtones, average_saturation_highlights]

            shadows_color = extract_dominant_color(hsv[shadows_mask])
            midtones_color = extract_dominant_color(hsv[midtones_mask])
            highlights_color = extract_dominant_color(hsv[highlights_mask])

            shadows_color_rgb = mcolors.hsv_to_rgb(shadows_color)
            midtones_color_rgb = mcolors.hsv_to_rgb(midtones_color)
            highlights_color_rgb = mcolors.hsv_to_rgb(highlights_color)

            bar_colors = [shadows_color_rgb, midtones_color_rgb, highlights_color_rgb]

            segments = ['Sombras', 'Meio-Tons', 'Altas-Luzes']

            fig = plt.figure(figsize=(width, height), facecolor='#2E2E2E')
            gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1.2, 1, 1])

            ax0 = plt.subplot(gs[0])
            ax0.imshow(img)
            ax0.axis('off')

            colors, counts = create_palette(img, 10)
            colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

            ax1 = plt.subplot(gs[1])
            start = 0
            for i in range(len(colors)):
                ax1.bar(start, 1, width=counts[i], color=colors[i], align='edge', edgecolor='none')
                start += counts[i]

            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.set_aspect('auto')
            ax1.axis('off')

            ax2 = plt.subplot(gs[2], facecolor='#2E2E2E')
            ax2.plot(x, r_avg, 'r', x, g_avg, 'g', x, b_avg, 'b')
            ax2.set_xlim(left=0)
            ax2.set_ylim(bottom=0)
            ax2.set_title('Waveform', color='white')
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.tick_params(axis='y', colors='white')

            ax3 = plt.subplot(gs[3], facecolor='#2E2E2E')
            for i, color in enumerate(['r', 'g', 'b']):
                ax3.hist(img_array[..., i].ravel(), bins=256, color=color, alpha=0.5)
            ax3.set_xlim(left=0)
            ax3.set_ylim(bottom=0)
            ax3.set_title('Histograma', color='white')
            ax3.tick_params(axis='x', colors='white')
            ax3.tick_params(axis='y', colors='white')
            ax3.set_yticks([])
            ax3.set_yticklabels([])

            plt.subplots_adjust(hspace=0.5)

            plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        # Prompt for prefix
        prefix = simpledialog.askstring("Input", "Enter prefix for saved images:")

        image_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png")], title="Select Images")
        if not image_paths:
            return

        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            save_path = os.path.join(save_dir, f"{prefix}_{filename}")
            process_image(image_path, save_path)
            print(f"Processed {filename}")

        messagebox.showinfo("Frame Reports Saved", "Frame Reports have been saved successfully.")

    def create_mosaic(self):
        image_dir = filedialog.askdirectory(title="Select Image Directory")
        if not image_dir:
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        output_filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")], title="Save Mosaic As")
        if not output_filename:
            return

        self.generate_mosaic(image_dir, output_filename)

    def generate_mosaic(self, image_dir, output_file, output_size=(1920, 1080)):
        image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith(('png', 'jpg', 'jpeg'))]

        if not image_files:
            messagebox.showerror("Error", "No images found in the selected directory.")
            return

        num_images = len(image_files)

        # Calculate the best number of rows and columns for the given number of images
        def best_grid(num_images):
            rows = cols = int(math.sqrt(num_images))
            while rows * cols < num_images:
                cols += 1
            if (rows - 1) * cols >= num_images:
                rows -= 1
            return rows, cols

        rows, cols = best_grid(num_images)

        # Create a new image with the specified output size
        mosaic_image = Image.new('RGB', output_size, (0, 0, 0))

        # Calculate thumbnail size to fit within the output size
        thumbnail_width = output_size[0] // cols
        thumbnail_height = output_size[1] // rows

        for idx, image_file in enumerate(image_files):
            img = Image.open(image_file)

            # Resize image while maintaining aspect ratio
            img.thumbnail((thumbnail_width, thumbnail_height), Image.LANCZOS)

            # Calculate position to center the thumbnail
            x = (idx % cols) * thumbnail_width + (thumbnail_width - img.width) // 2
            y = (idx // cols) * thumbnail_height + (thumbnail_height - img.height) // 2
            mosaic_image.paste(img, (x, y))

        mosaic_image.save(output_file)
        messagebox.showinfo("Success", f"Mosaic created and saved as {output_file}")

    def create_color_wheel_image(self, palette, wheel_size):
        radius = wheel_size // 2
        image = Image.new("RGB", (wheel_size, wheel_size), (46, 46, 46))
        draw = ImageDraw.Draw(image)

        # Calculate average brightness and saturation of the palette
        average_brightness = self.calculate_average_brightness(palette)
        average_saturation = self.calculate_average_saturation(palette)

        # Create the gradient color wheel
        for y in range(wheel_size):
            for x in range(wheel_size):
                dx = x - radius
                dy = y - radius
                distance = math.sqrt(dx * dx + dy * dy)
                if distance <= radius:
                    hue = math.atan2(dy, dx) / (2 * math.pi) + 0.5
                    saturation = distance / radius * average_saturation
                    value = 1.0 * average_brightness
                    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                    draw.point((x, y), fill=(int(r * 255), int(g * 255), int(b * 255)))

        # Add the colors from the palette
        num_colors = len(palette[0]) // 3
        sample_radius = 20

        for color in palette[0].reshape(-1, 3):
            # Convert RGB to HSV to get the hue angle
            r, g, b = color / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            angle = h * 2 * np.pi - np.pi  # Adjust angle for correct placement
            distance = s * radius

            x = int(radius + distance * np.cos(angle))
            y = int(radius + distance * np.sin(angle))

            # Draw sample circle
            draw.ellipse((x - sample_radius, y - sample_radius, x + sample_radius, y + sample_radius), fill=tuple(color), outline="white", width=2)
            # Draw line from the center to the sample with antialiasing
            draw.line((radius, radius, x, y), fill="white", width=2, joint="curve")

        return image

    def combine_wheel_and_palette(self, color_wheel, palette, size):
        combined_image = Image.new("RGB", size, (46, 46, 46))

        # Define the desired size of the color wheel while maintaining proportions
        wheel_diameter = min(color_wheel.width, color_wheel.height)  # Wheel diameter
        color_wheel_resized = color_wheel.resize((wheel_diameter, wheel_diameter), Image.LANCZOS)

        # Calculate the position to center the color wheel
        wheel_center_x = (size[0] - wheel_diameter) // 2
        wheel_center_y = (size[1] - wheel_diameter - 200) // 2

        # Paste the color wheel in the center of the image
        combined_image.paste(color_wheel_resized, (wheel_center_x, wheel_center_y))

        # Paste palette below the color wheel
        palette_pil = Image.fromarray(palette).resize((size[0], 200), Image.LANCZOS)
        combined_image.paste(palette_pil, (0, size[1] - 200))

        return combined_image

    def combine_image_and_palette(self, image, palette):
        combined_height = image.shape[0] + palette.shape[0]
        combined_image = np.zeros((combined_height, image.shape[1], 3), dtype=np.uint8)
        combined_image[:image.shape[0], :image.shape[1], :] = image
        combined_image[image.shape[0]:, :palette.shape[1], :] = palette
        return combined_image

    def run_full_report_script(self):
        script_path = "/Users/guilhermefuchs/PythonProjects/colorReport_8.py"  # Altere para o caminho real do seu script
        subprocess.run([sys.executable, script_path])

    def run_image_wipe(self):
        script_path = "/Users/guilhermefuchs/PythonProjects/ColorReport/image_wipe.py"
        subprocess.run([sys.executable, script_path])     


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorPaletteApp(root)
    root.mainloop()
