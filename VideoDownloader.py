import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import yt_dlp
import os
import subprocess
from PIL import Image, ImageTk
import cv2
import numpy as np


class VideoDownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Downloader")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.6)
        self.root.geometry(f"{window_width}x{window_height}")

        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.url_label = tk.Label(self.left_frame, text="Video URL:")
        self.url_label.grid(row=0, column=0, padx=2, pady=2, sticky='w')

        self.url_entry = tk.Entry(self.left_frame, width=30)
        self.url_entry.grid(row=0, column=1, padx=2, pady=2)

        self.fetch_button = tk.Button(self.left_frame, text="Process Video", command=self.fetch_video_info)
        self.fetch_button.grid(row=0, column=2, padx=2, pady=2)

        self.format_label = tk.Label(self.left_frame, text="Select Format:")
        self.format_label.grid(row=1, column=0, padx=2, pady=2, sticky='w')

        self.format_var = tk.StringVar()
        self.format_menu = tk.OptionMenu(self.left_frame, self.format_var, "")
        self.format_menu.config(width=15)
        self.format_menu.grid(row=1, column=1, padx=2, pady=2, columnspan=2, sticky='we')

        self.codec_label = tk.Label(self.left_frame, text="Select Codec:")
        self.codec_label.grid(row=2, column=0, padx=2, pady=2, sticky='w')

        self.codec_var = tk.StringVar()
        self.codec_menu = tk.OptionMenu(self.left_frame, self.codec_var, "")
        self.codec_menu.config(width=15)
        self.codec_menu.grid(row=2, column=1, padx=2, pady=2, columnspan=2, sticky='we')

        self.transcode_label = tk.Label(self.left_frame, text="Transcode to:")
        self.transcode_label.grid(row=3, column=0, padx=2, pady=2, sticky='w')

        self.transcode_var = tk.StringVar()
        self.transcode_options = ["None", "ProRes 422"]
        self.transcode_menu = tk.OptionMenu(self.left_frame, self.transcode_var, *self.transcode_options)
        self.transcode_var.set("None")
        self.transcode_menu.config(width=15)
        self.transcode_menu.grid(row=3, column=1, padx=2, pady=2, columnspan=2, sticky='we')

        self.remove_bars_var = tk.BooleanVar()
        self.remove_bars_checkbutton = tk.Checkbutton(self.left_frame, text="Remove Black Bars", variable=self.remove_bars_var)
        self.remove_bars_checkbutton.grid(row=4, column=0, padx=2, pady=2, columnspan=3, sticky='w')

        self.directory_button = tk.Button(self.left_frame, text="Select Save Directory", command=self.select_directory)
        self.directory_button.grid(row=5, column=0, padx=2, pady=2, columnspan=3, sticky='we')

        self.selected_directory_label = tk.Label(self.left_frame, text="No directory selected")
        self.selected_directory_label.grid(row=6, column=0, padx=2, pady=2, columnspan=3, sticky='we')

        self.download_button = tk.Button(self.left_frame, text="Download", command=self.download_video)
        self.download_button.grid(row=7, column=0, padx=2, pady=2, columnspan=3, sticky='we')

        self.export_cut_button = tk.Button(self.left_frame, text="Export Cut", command=self.export_cut)
        self.export_cut_button.grid(row=8, column=0, padx=2, pady=2, columnspan=3, sticky='we')

        self.import_button = tk.Button(self.left_frame, text="Import Video", command=self.import_video)
        self.import_button.grid(row=9, column=0, padx=2, pady=2, columnspan=3, sticky='we')

        self.save_directory = ""
        self.video_info = []
        self.in_point = None
        self.out_point = None

        self.format_var.trace_add('write', self.update_codecs)

        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_frame_container = tk.Frame(self.right_frame, bg="black", width=640, height=360)
        self.video_frame_container.pack_propagate(False)
        self.video_frame_container.pack(pady=10)

        self.video_frame = tk.Label(self.video_frame_container, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.right_frame)
        self.control_frame.pack()

        self.mark_in_button = tk.Button(self.control_frame, text="Mark In", command=lambda: self.mark_in_point(None))
        self.mark_in_button.pack(side=tk.LEFT)

        self.mark_out_button = tk.Button(self.control_frame, text="Mark Out", command=lambda: self.mark_out_point(None))
        self.mark_out_button.pack(side=tk.LEFT)

        self.grab_still_button = tk.Button(self.control_frame, text="Grab Still", command=self.grab_still)
        self.grab_still_button.pack(side=tk.LEFT)

        self.export_frames_button = tk.Button(self.control_frame, text="Export Frames", command=self.export_frames)
        self.export_frames_button.pack(side=tk.LEFT)

        self.timeline_frame = tk.Frame(self.right_frame)
        self.timeline_frame.pack(fill=tk.X)

        self.timeline = ttk.Scale(self.timeline_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_timeline)
        self.timeline.pack(fill=tk.X)

        self.timecode_label = tk.Label(self.timeline_frame, text="00:00:00:00")
        self.timecode_label.pack()

        root.bind("<KeyPress>", self.key_press)

        self.video_path = ""
        self.frames_dir = "frames"
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_positions = []

        self.cap = None
        self.playing = False
        self.progress_window = None  # Initialize progress_window

    def key_press(self, event):
        if event.char == 'i':
            self.mark_in_point(event)
        elif event.char == 'o':
            self.mark_out_point(event)
        elif event.char == 's':
            self.grab_still()
        elif event.char == 'e':
            self.export_frames()
        elif event.char == ' ':
            self.toggle_play_pause(event)
        elif event.char == 'j':
            self.rewind(event)
        elif event.char == 'l':
            self.fast_forward(event)

    def select_directory(self):
        self.save_directory = filedialog.askdirectory()
        if self.save_directory:
            self.selected_directory_label.config(text=self.save_directory)
        else:
            self.selected_directory_label.config(text="No directory selected")

    def mark_in_point(self, event):
        if self.cap and self.cap.isOpened():
            self.in_point = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.add_timeline_marker(self.in_point, "green")       

    def mark_out_point(self, event):
        if self.cap and self.cap.isOpened():
            self.out_point = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.add_timeline_marker(self.out_point, "red")      

    def show_progress_window(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Progress")
        
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f'{width}x{height}+{x}+{y}')
        
        progress_label = tk.Label(progress_window, text="Processing...")
        progress_label.pack(pady=20)
        
        progress_window.update()
        return progress_window
    
    def progress_hook(self, d):
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            speed = d.get('speed_str', 'N/A')
            self.update_progress(percent, eta, speed)
        elif d['status'] == 'finished':
            self.update_progress("100%", "0s", "Completed")

    def update_progress(self, percent, eta, speed):
        if self.progress_window and self.progress_var:
            self.progress_var.set(f"Downloaded: {percent} | ETA: {eta} | Speed: {speed}")
            self.progress_window.update_idletasks()

    def create_progress_window(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Download Progress")

        width = 400
        height = 100
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f'{width}x{height}+{x}+{y}')

        self.progress_var = tk.StringVar()
        self.progress_var.set("Starting download...")
        progress_label = tk.Label(progress_window, textvariable=self.progress_var)
        progress_label.pack(pady=20)

        progress_window.update()
        return progress_window      

    def fetch_video_info(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a video URL")
            return

        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'nocheckcertificate': True,
            'postprocessors': [{'key': 'FFmpegMetadata'}],
        }

        if url.startswith("https:"):
            url_http = url.replace("https:", "http:")
        else:
            url_http = url

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info_dict = ydl.extract_info(url, download=False)
                except yt_dlp.utils.DownloadError as e:
                    if 'TLS fingerprint' in str(e):
                        info_dict = ydl.extract_info(url_http, download=False)
                    else:
                        raise e
                formats = info_dict.get('formats', [])

            self.video_info = formats
            self.populate_format_options(formats)
        except yt_dlp.utils.DownloadError as e:
            messagebox.showerror("Error", f"Failed to fetch video info: {e}")
        except yt_dlp.utils.ExtractorError as e:
            messagebox.showerror("Error", f"Extractor error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def populate_format_options(self, formats):
        format_options = sorted(set(f['ext'] for f in formats if f['ext'] and f.get('height') == 1080))
        self.update_menu(self.format_menu, self.format_var, format_options)

    def update_codecs(self, *args):
        format_choice = self.format_var.get()
        if not format_choice:
            return

        codec_options = sorted(set(f['vcodec'] for f in self.video_info if f['ext'] == format_choice and 'vcodec' in f and f['vcodec'] and f.get('height') == 1080))
        self.update_menu(self.codec_menu, self.codec_var, codec_options)

    def update_menu(self, menu, var, options):
        menu['menu'].delete(0, 'end')
        for option in options:
            menu['menu'].add_command(label=option, command=tk._setit(var, option))
        if options:
            var.set(options[0])
        else:
            var.set('')

    def add_timeline_marker(self, frame_pos, color):
        mark = tk.Label(self.timeline_frame, text="|", fg=color)
        mark.place(relx=frame_pos / self.total_frames, rely=0.5, anchor=tk.CENTER)        

    def trim_video(self, input_file, in_frame, out_frame, output_file):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        in_time = in_frame / fps
        out_time = out_frame / fps

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss", f"{in_time:.3f}",
            "-i", input_file,
            "-to", f"{out_time - in_time:.3f}",
            "-c", "copy",
            output_file
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            messagebox.showinfo("Success", f"Video trimmed and saved as {output_file}")
            self.transcode_video(output_file)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to trim video: {e}")

    def download_video(self):
        url = self.url_entry.get().strip()
        format_choice = self.format_var.get()
        codec_choice = self.codec_var.get()

        if not url:
            messagebox.showerror("Error", "Please enter a video URL")
            return

        if not self.save_directory:
            messagebox.showerror("Error", "Please select a save directory")
            return

        selected_format = None
        for f in self.video_info:
            if (f['ext'] == format_choice and
                    'vcodec' in f and f['vcodec'] == codec_choice and
                    f.get('height') == 1080):
                selected_format = f['format_id']
                break

        if not selected_format:
            messagebox.showerror("Error", "No matching format found")
            return

        output_path = os.path.join(self.save_directory, '%(title)s.%(ext)s')
        ydl_opts = {
            'format': selected_format,
            'outtmpl': output_path,
            'nocheckcertificate': True,
            'progress_hooks': [self.progress_hook],
            'postprocessors': [{'key': 'FFmpegMetadata'}],
        }

        if url.startswith("https:"):
            url_http = url.replace("https:", "http:")
        else:
            url_http = url

        self.progress_window = self.create_progress_window()  # Initialize progress_window correctly

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info_dict = ydl.extract_info(url, download=True)
                except yt_dlp.utils.DownloadError as e:
                    if 'TLS fingerprint' in str(e):
                        info_dict = ydl.extract_info(url_http, download=True)
                    else:
                        raise e
                downloaded_file = ydl.prepare_filename(info_dict)
                self.video_path = downloaded_file
                if self.in_point is not None and self.out_point is not None:
                    self.trim_video(downloaded_file, self.in_point, self.out_point)
                else:
                    self.video_path = downloaded_file
                    self.transcode_video(downloaded_file)
                    self.load_video(downloaded_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download video: {e}")
        finally:
            if self.progress_window:
                self.progress_window.destroy()

    def import_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi")])
        if file_path:
            self.video_path = file_path
            self.load_video(file_path)
        else:
            messagebox.showerror("Error", "No video file selected")

    def load_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file")
            return

        self.playing = False
        self.update_video_frame()

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.timeline.config(to=self.total_frames - 1)
        self.frame_positions = []

    def transcode_video(self, input_file):
        transcode_choice = self.transcode_var.get()
        if transcode_choice == "None":
            self.progress_var.set("Download completed!")
            self.progress_window.update()
            return

        output_file = os.path.splitext(input_file)[0] + "_prores.mov"
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_file,
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-c:a", "copy",
            output_file
        ]

        self.progress_var.set("Transcoding...")
        self.progress_window.update()

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            self.progress_var.set("Transcoding completed!")
            self.progress_window.update()
        except subprocess.CalledProcessError as e:
            self.progress_var.set(f"Failed to transcode video: {e}")
            self.progress_window.update()

    def update_video_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        frame_width = self.video_frame_container.winfo_width()
        frame_height = self.video_frame_container.winfo_height()
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        if self.playing:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame < self.total_frames:
                self.timeline.set(current_frame)
                timecode = self.format_timecode(current_frame / self.cap.get(cv2.CAP_PROP_FPS))
                self.timecode_label.config(text=timecode)
                self.root.after(33, self.update_video_frame)
            else:
                self.pause_video()

    def rewind(self, event=None):
        self.playback_speed /= 2
        if self.playing:
            self.update_video_frame()

    def fast_forward(self, event=None):
        self.playback_speed *= 2
        if self.playing:
            self.update_video_frame()

    def grab_still(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "No video loaded")
            return

        frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if ret:
            filename = os.path.join(self.frames_dir, f"frame_{frame_pos}.jpg")
            cv2.imwrite(filename, frame)
            self.frame_positions.append(frame_pos)
            self.add_timeline_marker(frame_pos, "blue")
            self.update_timeline_marks()

    def remove_black_bars(self, image):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            image_np_cropped = image_np[y:y+h, x:x+w]
            image_cropped = Image.fromarray(cv2.cvtColor(image_np_cropped, cv2.COLOR_BGR2RGB))
            return image_cropped
        return image
    
    def export_cut(self):
        if self.in_point is None or self.out_point is None:
            messagebox.showerror("Error", "Please set both In and Out points")
            return

        if not self.video_path:
            messagebox.showerror("Error", "No video loaded to cut")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="Save Video Cut As"
        )
        
        if not save_path:
            return

        if os.path.exists(save_path):
            if not messagebox.askyesno("Overwrite Confirmation", f"The file '{save_path}' already exists. Do you want to overwrite it?"):
                return

        self.progress_var = tk.StringVar()
        self.progress_window = self.create_progress_window()

        try:
            self.trim_video(self.video_path, self.in_point, self.out_point, save_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cut video: {e}")
        finally:
            if self.progress_window:
                self.progress_window.destroy()

    def export_frames(self):
        if not self.frame_positions:
            messagebox.showerror("Error", "No frames to export")
            return

        save_dir = filedialog.askdirectory()
        if not save_dir:
            return

        export_settings = ExportSettingsDialog(self.root)
        self.root.wait_window(export_settings.top)

        if not export_settings.ok_pressed:
            return

        prefix = export_settings.prefix
        export_format = export_settings.export_format
        remove_bars = self.remove_bars_var.get()

        for i, pos in enumerate(self.frame_positions):
            src = os.path.join(self.frames_dir, f"frame_{pos}.jpg")
            if os.path.exists(src):
                dst_filename = f"{prefix}_{i+1}.{export_format}"
                dst = os.path.join(save_dir, dst_filename)
                
                image = Image.open(src)
                if remove_bars:
                    image = self.remove_black_bars(image)
                if export_format.lower() == "jpg":
                    image.save(dst, "JPEG", quality=100)
                elif export_format.lower() == "png":
                    image.save(dst, "PNG")
                elif export_format.lower() == "tiff":
                    image.save(dst, "TIFF")

        messagebox.showinfo("Success", "Frames exported successfully")

    def update_timeline(self, value):
        if not self.cap or not self.cap.isOpened():
            return

        frame_pos = int(float(value))
        if frame_pos < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.update_video_frame()

            timecode = self.format_timecode(frame_pos / self.cap.get(cv2.CAP_PROP_FPS))
            self.timecode_label.config(text=timecode)

    def update_timeline_marks(self):
        for widget in self.timeline_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.cget("text") == "|":
                widget.destroy()

        for pos in self.frame_positions:
            mark = tk.Label(self.timeline_frame, text="|", fg="blue")
            mark.place(relx=pos / self.total_frames, rely=0.5, anchor=tk.CENTER)

        if self.in_point is not None:
            mark = tk.Label(self.timeline_frame, text="|", fg="green")
            mark.place(relx=self.in_point / self.total_frames, rely=0.5, anchor=tk.CENTER)

        if self.out_point is not None:
            mark = tk.Label(self.timeline_frame, text="|", fg="red")
            mark.place(relx=self.out_point / self.total_frames, rely=0.5, anchor=tk.CENTER)

    def format_timecode(self, seconds):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frames = int((seconds - int(seconds)) * fps)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}:{frames:02}"

class ExportSettingsDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Export Settings")
        
        self.top.update_idletasks()
        width = self.top.winfo_width()
        height = self.top.winfo_height()
        x = (self.top.winfo_screenwidth() // 2) - (width // 2)
        y = (self.top.winfo_screenheight() // 2) - (height // 2)
        self.top.geometry(f'{width}x{height}+{x}+{y}')
        
        self.prefix_label = tk.Label(self.top, text="Prefix:")
        self.prefix_label.pack(pady=5)
        self.prefix_entry = tk.Entry(self.top)
        self.prefix_entry.pack(pady=5)
        
        self.format_label = tk.Label(self.top, text="Format:")
        self.format_label.pack(pady=5)
        self.format_var = tk.StringVar(value="jpg")
        self.format_menu = ttk.Combobox(self.top, textvariable=self.format_var, values=["jpg", "png", "tiff"])
        self.format_menu.pack(pady=5)
        
        self.ok_button = tk.Button(self.top, text="OK", command=self.on_ok)
        self.ok_button.pack(pady=5)
        
        self.ok_pressed = False
    
    def on_ok(self):
        self.prefix = self.prefix_entry.get().strip()
        self.export_format = self.format_var.get().strip()
        self.ok_pressed = True
        self.top.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDownloaderApp(root)
    root.mainloop()
