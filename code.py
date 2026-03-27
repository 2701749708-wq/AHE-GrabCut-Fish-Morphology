import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import platform

K = np.array([[643.1389, 0, 0], [0, 642.0301, 0], [320.6402, 227.6466, 1]], dtype=np.float32)
dist = np.array([0.1836, -0.3672, 0, 0, 0], dtype=np.float32)

measurement_points = []
processed_image = None
processed_left_image = None
processed_right_image = None
original_image_path = ""
is_stereo_image = False
current_view = "single"
drawing = False
measure_line_id = None
distance_text_id = None
temp_line_id = None


class StyleConfig:
    PRIMARY_COLOR = "#3A86FF"
    SECONDARY_COLOR = "#8338EC"
    ACCENT_COLOR = "#FF006E"
    BACKGROUND_COLOR = "#FFFFFF"
    PANEL_COLOR = "#F8F9FA"
    TEXT_COLOR = "#212529"
    TEXT_LIGHT = "#6C757D"
    BORDER_COLOR = "#DEE2E6"
    BUTTON_HOVER = "#2667FF"
    MEASURE_LINE = "#FFBE0B"
    MEASURE_POINT = "#FF006E"
    LEFT_VIEW_COLOR = "#34D399"
    RIGHT_VIEW_COLOR = "#6366F1"

    FONT_FAMILY = "Microsoft YaHei UI"
    FONT_SIZE_SMALL = 10
    FONT_SIZE_NORMAL = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_TITLE = 18
    FONT_SIZE_HEAD = 16

    PADDING_XS = 3
    PADDING_SMALL = 8
    PADDING_NORMAL = 12
    PADDING_LARGE = 16
    PADDING_XL = 20
    BORDER_RADIUS = 10
    BUTTON_HEIGHT = 36
    IMAGE_MAX_HEIGHT = 650
    BUTTON_WIDTH = 18


def fix_image_path(path):
    if not path:
        return ""

    path = os.path.normpath(path)

    if platform.system() == "Windows":
        path = os.path.abspath(path)
        try:
            path = path.encode('utf-8').decode('utf-8')
        except:
            pass

    return path


def split_stereo_image(image):
    height, width = image.shape[:2]
    mid_width = width // 2
    left_img = image[:, :mid_width]
    right_img = image[:, mid_width:]
    return left_img, right_img


def process_single_image(image_path):
    try:
        fixed_path = fix_image_path(image_path)
        if not os.path.exists(fixed_path):
            raise Exception(f"文件不存在: {fixed_path}")

        stream = open(fixed_path, 'rb')
        bytes_data = bytearray(stream.read())
        np_arr = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception(f"无法读取图片: {fixed_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        mask = np.zeros(enhanced_image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, enhanced_image.shape[1] - 40, enhanced_image.shape[0] - 40)

        cv2.grabCut(enhanced_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented_image = enhanced_image * mask2[:, :, np.newaxis]

        gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_segmented, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return segmented_image

    except Exception as e:
        messagebox.showerror("错误", f"图片处理失败: {e}")
        return None


def adaptive_histogram_equalization(image_path):
    try:
        fixed_path = fix_image_path(image_path)
        if not os.path.exists(fixed_path):
            raise Exception(f"文件不存在: {fixed_path}")

        stream = open(fixed_path, 'rb')
        bytes_data = bytearray(stream.read())
        np_arr = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise Exception(f"无法读取图片: {fixed_path}")

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)

        output_path = fixed_path.rsplit('.', 1)[0] + '_enhanced.' + fixed_path.rsplit('.', 1)[1]
        cv2.imencode(os.path.splitext(output_path)[1], enhanced_image)[1].tofile(output_path)

        return output_path, enhanced_image
    except Exception as e:
        messagebox.showerror("错误", f"直方图均衡化失败: {e}")
        return None, None


def grab_cut_segmentation(image_path):
    try:
        fixed_path = fix_image_path(image_path)
        if not os.path.exists(fixed_path):
            raise Exception(f"文件不存在: {fixed_path}")

        stream = open(fixed_path, 'rb')
        bytes_data = bytearray(stream.read())
        np_arr = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception(f"无法读取图片: {fixed_path}")

        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, img.shape[1] - 40, img.shape[0] - 40)

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]

        output_path = fixed_path.rsplit('.', 1)[0] + '_segmented.' + fixed_path.rsplit('.', 1)[1]
        cv2.imencode(os.path.splitext(output_path)[1], img)[1].tofile(output_path)

        return output_path, img
    except Exception as e:
        messagebox.showerror("错误", f"GrabCut分割失败: {e}")
        return None, None


def outline_largest_contour(image_path):
    try:
        fixed_path = fix_image_path(image_path)
        if not os.path.exists(fixed_path):
            raise Exception(f"文件不存在: {fixed_path}")

        stream = open(fixed_path, 'rb')
        bytes_data = bytearray(stream.read())
        np_arr = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception(f"无法读取图片: {fixed_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = fixed_path.rsplit('.', 1)[0] + '_outlined.' + fixed_path.rsplit('.', 1)[1]
        cv2.imencode(os.path.splitext(output_path)[1], image)[1].tofile(output_path)

        return output_path, image
    except Exception as e:
        messagebox.showerror("错误", f"轮廓检测失败: {e}")
        return None, None


class ImageProcessorUI:
    def __init__(self, root):
        self.root = root
        self.style = StyleConfig()

        self.root.title("水产生物轮廓无损测量视觉系统V1.0 ")
        self.root.geometry("1400x950")
        self.root.configure(bg=self.style.BACKGROUND_COLOR)
        self.root.minsize(1200, 800)

        self.image_tk = None
        self.image_origin = None
        self.left_image_origin = None
        self.right_image_origin = None
        self.scale_factor = 1.0
        self.measure_mode = False
        self.image_x = 0
        self.image_y = 0
        self.image_width = 0
        self.image_height = 0

        self.setup_style()

        self.create_widgets()

        self.bind_events()

        self.update_view_buttons_visibility()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Primary.TButton',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL, 'bold'),
                        padding=(15, 8),
                        borderwidth=0,
                        relief='flat')

        style.map('Primary.TButton',
                  background=[('active', self.style.BUTTON_HOVER),
                              ('!active', self.style.PRIMARY_COLOR)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        style.configure('Secondary.TButton',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        padding=(15, 8),
                        borderwidth=0,
                        relief='flat')

        style.map('Secondary.TButton',
                  background=[('active', '#702ED8'),
                              ('!active', self.style.SECONDARY_COLOR)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        style.configure('Danger.TButton',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        padding=(15, 8),
                        borderwidth=0,
                        relief='flat')

        style.map('Danger.TButton',
                  background=[('active', '#E0005E'),
                              ('!active', self.style.ACCENT_COLOR)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        style.configure('Left.TButton',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        padding=(10, 5),
                        borderwidth=0,
                        relief='flat')

        style.map('Left.TButton',
                  background=[('active', '#10B981'),
                              ('!active', self.style.LEFT_VIEW_COLOR)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        style.configure('Right.TButton',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        padding=(10, 5),
                        borderwidth=0,
                        relief='flat')

        style.map('Right.TButton',
                  background=[('active', '#4F46E5'),
                              ('!active', self.style.RIGHT_VIEW_COLOR)],
                  foreground=[('active', 'white'), ('!active', 'white')])

        style.configure('Title.TLabel',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_TITLE, 'bold'),
                        foreground=self.style.TEXT_COLOR,
                        background=self.style.BACKGROUND_COLOR)

        style.configure('Header.TLabel',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_HEAD, 'bold'),
                        foreground=self.style.TEXT_COLOR,
                        background=self.style.PANEL_COLOR)

        style.configure('Normal.TLabel',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        foreground=self.style.TEXT_COLOR,
                        background=self.style.PANEL_COLOR)

        style.configure('Light.TLabel',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_SMALL),
                        foreground=self.style.TEXT_LIGHT,
                        background=self.style.PANEL_COLOR)

        style.configure('Custom.TEntry',
                        font=(self.style.FONT_FAMILY, self.style.FONT_SIZE_NORMAL),
                        padding=self.style.PADDING_XS,
                        relief='flat',
                        borderwidth=1)

        style.configure('Panel.TFrame',
                        background=self.style.PANEL_COLOR,
                        relief='flat',
                        borderwidth=1)

        style.configure('Card.TFrame',
                        background=self.style.BACKGROUND_COLOR,
                        relief='raised',
                        borderwidth=1)

    def create_widgets(self):
        main_container = ttk.Frame(self.root, style='Card.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True,
                            padx=self.style.PADDING_XL,
                            pady=self.style.PADDING_XL)

        title_frame = ttk.Frame(main_container, style='Card.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, self.style.PADDING_LARGE))

        title_label = ttk.Label(title_frame, text="水产生物轮廓无损测量视觉系统 ", style='Title.TLabel')
        title_label.pack(pady=self.style.PADDING_NORMAL)

        control_panel = ttk.Frame(main_container, style='Panel.TFrame')
        control_panel.pack(fill=tk.X, pady=(0, self.style.PADDING_LARGE),
                           ipady=self.style.PADDING_NORMAL,
                           padx=self.style.PADDING_NORMAL)

        path_frame = ttk.Frame(control_panel, style='Panel.TFrame')
        path_frame.pack(fill=tk.X, pady=(0, self.style.PADDING_NORMAL))

        ttk.Label(path_frame, text="图片类型：", style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 10))

        self.image_type_var = tk.StringVar(value="single")
        single_radio = ttk.Radiobutton(path_frame, text="单目图片",
                                       variable=self.image_type_var,
                                       value="single",
                                       command=self.switch_image_type)
        single_radio.pack(side=tk.LEFT, padx=(0, 15))

        stereo_radio = ttk.Radiobutton(path_frame, text="双目图片",
                                       variable=self.image_type_var,
                                       value="stereo",
                                       command=self.switch_image_type)
        stereo_radio.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(path_frame, text="图片路径：", style='Header.TLabel').pack(side=tk.LEFT, padx=(0, 10))

        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var,
                               style='Custom.TEntry', width=70)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(path_frame, text="浏览选择",
                                command=self.browse_image,
                                style='Primary.TButton', width=self.style.BUTTON_WIDTH)
        browse_btn.pack(side=tk.LEFT)

        btn_frame = ttk.Frame(control_panel, style='Panel.TFrame')
        btn_frame.pack(fill=tk.X, expand=True)

        self.process_btn = ttk.Button(btn_frame, text="处理图像",
                                      command=self.process_image,
                                      style='Primary.TButton', width=self.style.BUTTON_WIDTH)
        self.process_btn.grid(row=0, column=0, padx=self.style.PADDING_NORMAL, pady=5, sticky='ew')

        self.left_view_btn = ttk.Button(btn_frame, text="显示左视图",
                                        command=lambda: self.switch_view("left"),
                                        style='Left.TButton',
                                        width=12)
        self.left_view_btn.grid(row=0, column=1, padx=self.style.PADDING_SMALL, pady=5, sticky='ew')

        self.right_view_btn = ttk.Button(btn_frame, text="显示右视图",
                                         command=lambda: self.switch_view("right"),
                                         style='Right.TButton',
                                         width=12)
        self.right_view_btn.grid(row=0, column=2, padx=self.style.PADDING_SMALL, pady=5, sticky='ew')

        self.measure_btn = ttk.Button(btn_frame, text="开启测量",
                                      command=self.toggle_measure_mode,
                                      style='Secondary.TButton',
                                      state=tk.DISABLED,
                                      width=self.style.BUTTON_WIDTH)
        self.measure_btn.grid(row=0, column=3, padx=self.style.PADDING_NORMAL, pady=5, sticky='ew')

        self.clear_btn = ttk.Button(btn_frame, text="清空内容",
                                    command=self.clear_display,
                                    style='Danger.TButton',
                                    width=self.style.BUTTON_WIDTH)
        self.clear_btn.grid(row=0, column=4, padx=self.style.PADDING_NORMAL, pady=5, sticky='ew')

        self.status_var = tk.StringVar(value="状态：未加载图片")
        status_label = ttk.Label(btn_frame, textvariable=self.status_var, style='Light.TLabel')
        status_label.grid(row=0, column=5, padx=self.style.PADDING_NORMAL, pady=5, sticky='e')

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=0)
        btn_frame.columnconfigure(2, weight=0)
        btn_frame.columnconfigure(3, weight=1)
        btn_frame.columnconfigure(4, weight=1)
        btn_frame.columnconfigure(5, weight=3)

        image_container = ttk.Frame(main_container, style='Card.TFrame')
        image_container.pack(fill=tk.BOTH, expand=True,
                             ipady=self.style.PADDING_SMALL,
                             padx=self.style.PADDING_SMALL)

        self.image_canvas = tk.Canvas(image_container,
                                      bg='white',
                                      highlightthickness=2,
                                      highlightbackground=self.style.BORDER_COLOR,
                                      relief='flat')
        self.image_canvas.pack(fill=tk.BOTH, expand=True,
                               padx=self.style.PADDING_SMALL,
                               pady=self.style.PADDING_SMALL)

        info_frame = ttk.Frame(main_container, style='Panel.TFrame')
        info_frame.pack(fill=tk.X, pady=(self.style.PADDING_NORMAL, 0),
                        ipady=self.style.PADDING_SMALL)

        self.view_info_var = tk.StringVar(value="当前视图：单目")
        view_label = ttk.Label(info_frame, textvariable=self.view_info_var, style='Normal.TLabel')
        view_label.pack(side=tk.LEFT, pady=self.style.PADDING_SMALL, padx=self.style.PADDING_NORMAL)

        self.measure_info_var = tk.StringVar(value="测量信息：未进行测量")
        info_label = ttk.Label(info_frame, textvariable=self.measure_info_var,
                               style='Normal.TLabel')
        info_label.pack(side=tk.RIGHT, pady=self.style.PADDING_SMALL, padx=self.style.PADDING_NORMAL)

    def update_view_buttons_visibility(self):
        image_type = self.image_type_var.get()
        if image_type == "stereo":
            self.left_view_btn.config(state=tk.DISABLED)
            self.right_view_btn.config(state=tk.DISABLED)
            self.left_view_btn.grid()
            self.right_view_btn.grid()
        else:
            self.left_view_btn.grid_remove()
            self.right_view_btn.grid_remove()

    def bind_events(self):
        self.image_canvas.bind('<Button-1>', self.canvas_click)
        self.image_canvas.bind('<Motion>', self.canvas_motion)
        self.image_canvas.bind('<ButtonRelease-1>', self.canvas_release)

        self.root.bind('<Configure>', self.on_window_resize)

        self.root.bind('<Return>', lambda e: self.process_image())

    def switch_image_type(self):
        self.update_view_buttons_visibility()

        image_type = self.image_type_var.get()
        if image_type == "single":
            self.status_var.set("状态：已切换为单目图片模式")
            self.view_info_var.set("当前视图：单目")
        else:
            self.status_var.set("状态：已切换为双目图片模式（将自动分割左右视图）")
            self.view_info_var.set("当前视图：双目（待处理）")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title=f"选择{('单目' if self.image_type_var.get() == 'single' else '双目')}图片文件",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("所有图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif *.webp"),
                ("JPEG图片", "*.jpg *.jpeg"),
                ("PNG图片", "*.png"),
                ("BMP图片", "*.bmp"),
                ("TIFF图片", "*.tif *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            fixed_path = fix_image_path(file_path)
            self.path_var.set(fixed_path)
            global original_image_path
            original_image_path = fixed_path

            image_type = self.image_type_var.get()
            self.status_var.set(
                f"状态：已选择{('单目' if image_type == 'single' else '双目')}图片 - {os.path.basename(fixed_path)}")

    def process_image(self):
        global processed_image, processed_left_image, processed_right_image, is_stereo_image

        file_path = self.path_var.get().strip()
        if not file_path:
            messagebox.showwarning("警告", "请先输入或选择图片路径！")
            return

        fixed_path = fix_image_path(file_path)

        if not os.path.exists(fixed_path):
            messagebox.showerror("错误", f"文件不存在：{fixed_path}")
            return

        try:
            self.root.config(cursor="wait")
            self.root.update()

            image_type = self.image_type_var.get()
            is_stereo_image = (image_type == "stereo")

            if is_stereo_image:
                stream = open(fixed_path, 'rb')
                bytes_data = bytearray(stream.read())
                np_arr = np.asarray(bytes_data, dtype=np.uint8)
                stereo_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if stereo_image is None:
                    raise Exception("无法读取双目图片")

                left_img, right_img = split_stereo_image(stereo_image)

                self.left_image_origin = left_img.copy()
                self.right_image_origin = right_img.copy()

                self.status_var.set("状态：正在处理左视图...")
                self.root.update()
                processed_left_image = process_single_image_from_array(left_img, fixed_path, "_left")

                self.status_var.set("状态：正在处理右视图...")
                self.root.update()
                processed_right_image = process_single_image_from_array(right_img, fixed_path, "_right")

                processed_image = processed_left_image
                self.image_origin = left_img.copy()
                self.current_view = "left"

                self.left_view_btn.config(state=tk.NORMAL)
                self.right_view_btn.config(state=tk.NORMAL)

                self.display_image(processed_image)
                self.view_info_var.set("当前视图：双目-左视图")
                self.status_var.set("状态：双目图片处理完成 - 已分割并处理左右视图")

            else:
                enhanced_path, _ = adaptive_histogram_equalization(fixed_path)
                if enhanced_path is None:
                    raise Exception("直方图均衡化失败")

                segmented_path, _ = grab_cut_segmentation(enhanced_path)
                if segmented_path is None:
                    raise Exception("GrabCut分割失败")

                outlined_path, final_img = outline_largest_contour(segmented_path)
                if outlined_path is None:
                    raise Exception("轮廓检测失败")

                processed_image = final_img
                self.image_origin = final_img.copy()
                self.current_view = "single"

                self.display_image(final_img)
                self.view_info_var.set("当前视图：单目")
                self.status_var.set(f"状态：单目图片处理完成 - {os.path.basename(fixed_path)}")

            self.measure_btn.config(state=tk.NORMAL)

            messagebox.showinfo("成功", f"{('单目' if not is_stereo_image else '双目')}图片处理完成！")

        except Exception as e:
            messagebox.showerror("错误", f"图像处理失败: {str(e)}")
        finally:
            self.root.config(cursor="")

    def switch_view(self, view_type):
        global processed_image

        if not is_stereo_image:
            return

        self.current_view = view_type

        if view_type == "left":
            processed_image = processed_left_image
            self.image_origin = self.left_image_origin.copy()
            self.view_info_var.set("当前视图：双目-左视图")
        else:
            processed_image = processed_right_image
            self.image_origin = self.right_image_origin.copy()
            self.view_info_var.set("当前视图：双目-右视图")

        self.clear_measure_drawings()
        self.display_image(processed_image)
        self.status_var.set(f"状态：已切换到{('左' if view_type == 'left' else '右')}视图")

    def display_image(self, img):
        self.image_canvas.delete("all")

        if img is None:
            return

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width < 100:
            canvas_width = 800
        if canvas_height < 100:
            canvas_height = 600

        img_height, img_width = img.shape[:2]

        scale_w = canvas_width / img_width
        scale_h = canvas_height / self.style.IMAGE_MAX_HEIGHT
        self.scale_factor = min(scale_w, scale_h, 1.0)

        self.image_width = int(img_width * self.scale_factor)
        self.image_height = int(img_height * self.scale_factor)

        resized_img = cv2.resize(img, (self.image_width, self.image_height))

        self.image_x = (canvas_width - self.image_width) // 2
        self.image_y = (canvas_height - self.image_height) // 2

        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.image_tk = ImageTk.PhotoImage(pil_img)

        self.image_canvas.create_image(self.image_x, self.image_y,
                                       anchor=tk.NW,
                                       image=self.image_tk)

        self.image_canvas.update_idletasks()

    def toggle_measure_mode(self):
        global measurement_points
        measurement_points = []

        if not self.measure_mode:
            self.measure_mode = True
            self.measure_btn.config(text="关闭测量")
            view_text = f"{('双目-' + self.current_view + '视图' if is_stereo_image else '单目视图')}"
            self.status_var.set(f"状态：测量模式已开启 - 点击{view_text}选择两点测量距离")
            self.measure_info_var.set("测量信息：请点击图片选择第一个点")
        else:
            self.measure_mode = False
            self.measure_btn.config(text="开启测量")
            self.status_var.set("状态：测量模式已关闭")
            self.measure_info_var.set("测量信息：测量已结束")
            self.clear_measure_drawings()

    def is_inside_image(self, x, y):
        return (self.image_x <= x <= self.image_x + self.image_width and
                self.image_y <= y <= self.image_y + self.image_height)

    def canvas_click(self, event):
        global measurement_points

        if not self.measure_mode or processed_image is None:
            return

        if not self.is_inside_image(event.x, event.y):
            return

        img_x = (event.x - self.image_x) / self.scale_factor
        img_y = (event.y - self.image_y) / self.scale_factor

        measurement_points.append((img_x, img_y))

        self.image_canvas.create_oval(event.x - 4, event.y - 4,
                                      event.x + 4, event.y + 4,
                                      fill=self.style.MEASURE_POINT,
                                      outline="white",
                                      width=2,
                                      tags="measure_point")

        if len(measurement_points) == 2:
            self.calculate_and_display_distance()

    def calculate_and_display_distance(self):
        global measurement_points

        (x1, y1), (x2, y2) = measurement_points

        cx1 = self.image_x + int(x1 * self.scale_factor)
        cy1 = self.image_y + int(y1 * self.scale_factor)
        cx2 = self.image_x + int(x2 * self.scale_factor)
        cy2 = self.image_y + int(y2 * self.scale_factor)

        self.clear_measure_drawings()

        self.image_canvas.create_line(cx1, cy1, cx2, cy2,
                                      fill=self.style.MEASURE_LINE,
                                      width=3,
                                      tags="measure_line")

        pixel_distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        Z = 1
        point1 = np.array([[x1], [y1], [1]])
        point2 = np.array([[x2], [y2], [1]])
        point1_camera = Z * np.linalg.inv(K) @ point1
        point2_camera = Z * np.linalg.inv(K) @ point2
        physical_distance = np.linalg.norm(point1_camera - point2_camera)

        text_x = (cx1 + cx2) // 2
        text_y = (cy1 + cy2) // 2 - 15

        distance_text = f"像素: {pixel_distance:.2f} | 物理: {physical_distance:.2f} mm"

        self.image_canvas.create_text(text_x, text_y,
                                      text=distance_text,
                                      fill="black",
                                      font=(self.style.FONT_FAMILY, 11, 'bold'),
                                      tags="measure_text_border")

        self.image_canvas.create_text(text_x - 1, text_y - 1,
                                      text=distance_text,
                                      fill=self.style.MEASURE_LINE,
                                      font=(self.style.FONT_FAMILY, 11, 'bold'),
                                      tags="measure_text")

        view_text = f"{('双目-' + self.current_view + '视图' if is_stereo_image else '单目视图')}"
        self.measure_info_var.set(
            f"测量信息（{view_text}）：像素距离 {pixel_distance:.2f} | 物理距离 {physical_distance:.2f} mm")

        measurement_points = []

    def canvas_motion(self, event):
        if not self.measure_mode or len(measurement_points) != 1 or processed_image is None:
            return

        self.image_canvas.delete("temp_line")

        if not self.is_inside_image(event.x, event.y):
            return

        (x1, y1) = measurement_points[0]
        cx1 = self.image_x + int(x1 * self.scale_factor)
        cy1 = self.image_y + int(y1 * self.scale_factor)

        self.image_canvas.create_line(cx1, cy1, event.x, event.y,
                                      fill=self.style.MEASURE_LINE,
                                      width=2,
                                      dash=(3, 3),
                                      tags="temp_line")

    def canvas_release(self, event):
        pass

    def clear_measure_drawings(self):
        self.image_canvas.delete("measure_line")
        self.image_canvas.delete("measure_text")
        self.image_canvas.delete("measure_text_border")
        self.image_canvas.delete("temp_line")
        self.image_canvas.delete("measure_point")

    def clear_display(self):
        global processed_image, processed_left_image, processed_right_image, is_stereo_image

        self.image_canvas.delete("all")

        self.path_var.set("")

        processed_image = None
        processed_left_image = None
        processed_right_image = None
        original_image_path = ""
        is_stereo_image = False
        measurement_points = []

        self.measure_mode = False
        self.image_tk = None
        self.image_origin = None
        self.left_image_origin = None
        self.right_image_origin = None
        self.scale_factor = 1.0
        self.current_view = "single"

        self.measure_btn.config(state=tk.DISABLED, text="开启测量")
        self.left_view_btn.config(state=tk.DISABLED)
        self.right_view_btn.config(state=tk.DISABLED)
        self.status_var.set("状态：未加载图片")
        self.view_info_var.set("当前视图：单目")
        self.measure_info_var.set("测量信息：未进行测量")

        messagebox.showinfo("提示", "已清空所有内容！")

    def on_window_resize(self, event):
        if processed_image is not None and self.image_origin is not None:
            self.root.after(100, lambda: self.display_image(processed_image))


def process_single_image_from_array(image, base_path, suffix):
    try:
        temp_path = base_path.rsplit('.', 1)[0] + suffix + '.' + base_path.rsplit('.', 1)[1]
        cv2.imencode(os.path.splitext(temp_path)[1], image)[1].tofile(temp_path)

        enhanced_path, enhanced_img = adaptive_histogram_equalization(temp_path)
        if enhanced_path is None:
            raise Exception("直方图均衡化失败")

        enhanced_color = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

        segmented_path, segmented_img = grab_cut_segmentation(enhanced_path)
        if segmented_path is None:
            raise Exception("GrabCut分割失败")

        outlined_path, final_img = outline_largest_contour(segmented_path)
        if outlined_path is None:
            raise Exception("轮廓检测失败")

        for temp_file in [temp_path, enhanced_path, segmented_path, outlined_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return final_img

    except Exception as e:
        messagebox.showerror("错误", f"处理分割后的视图失败: {e}")
        return None


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

    root = tk.Tk()
    app = ImageProcessorUI(root)
    root.mainloop()